from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from utilts import DETECTOR, TRAIN


class RQRDetector(nn.Module):
    def __init__(self, encoder_path):
        super().__init__()
        self.num_labels = DETECTOR["num_labels"]
        self.quality_dim = DETECTOR["quality_dim"]
        self.projection_dim = DETECTOR["projection_dim"]
        self.weight_normalization = DETECTOR["weight_normalization"]
        self.detach_weights = DETECTOR["detach_weights"]

        self.bert = AutoModel.from_pretrained(encoder_path)
        hidden = self.bert.config.hidden_size

        self.quality_net = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(DETECTOR["dropout"]),
            nn.Linear(256, self.quality_dim),
            nn.Sigmoid(),
        )
        self.quality_classifier = nn.Sequential(
            nn.Linear(hidden + self.quality_dim, 256),
            nn.ReLU(),
            nn.Dropout(DETECTOR["dropout"]),
            nn.Linear(256, self.num_labels),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, self.projection_dim)
        )
        self.reason_projection = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, self.projection_dim)
        )

        if self.quality_dim == 3:
            init = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float32)
        else:
            init = torch.full((self.quality_dim,), 1.0 / self.quality_dim, dtype=torch.float32)
        self.quality_logits = nn.Parameter(torch.log(init))
        self.temperature = nn.Parameter(torch.tensor(float(DETECTOR["temperature"])))

    @property
    def quality_weights(self):
        return F.softmax(self.quality_logits, dim=-1)

    def extract_reason(self, hidden_states, reason_start_idx):
        reps = []
        for i in range(hidden_states.size(0)):
            start = int(reason_start_idx[i].item())
            segment = hidden_states[i, start:, :]
            reps.append(segment.mean(dim=0) if segment.size(0) else hidden_states[i, 0, :])
        return torch.stack(reps)

    def contrastive_weights(self, q):
        if self.weight_normalization == "mean":
            weights = q / (q.mean() + 1e-8)
        else:
            weights = F.softmax(q / 0.1, dim=0) * q.size(0)
        return weights.detach() if self.detach_weights else weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        reason_start_idx=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        pooled = outputs.pooler_output
        reason_rep = self.extract_reason(hidden_states, reason_start_idx)

        quality_scores = self.quality_net(pooled)
        q = (quality_scores * self.quality_weights).sum(dim=1)
        logits = self.quality_classifier(torch.cat([pooled, quality_scores], dim=-1))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if self.training:
                z_cls = F.normalize(self.text_projection(pooled), dim=-1)
                z_rea = F.normalize(self.reason_projection(reason_rep), dim=-1)
                similarity = torch.matmul(z_cls, z_rea.T) / torch.clamp(
                    self.temperature, min=0.1
                )
                exp_sim = torch.exp(similarity.clamp(-50, 50))
                p_ii = exp_sim.diag() / (exp_sim.sum(dim=1) + 1e-8)
                weights = self.contrastive_weights(q)
                loss = loss + TRAIN["lambda_con"] * (-(weights * torch.log(p_ii + 1e-8)).mean())

        return SequenceClassifierOutput(loss=loss, logits=logits)

    @classmethod
    def from_checkpoint(cls, checkpoint, encoder_path):
        model = cls(encoder_path=encoder_path)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        return model

    @torch.no_grad()
    def predict(self, batch, device):
        self.eval()
        moved = {k: v.to(device) for k, v in batch.items()}
        outputs = self(
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            token_type_ids=moved["token_type_ids"],
            reason_start_idx=moved["reason_start_idx"],
        )
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, pred = probs.max(dim=-1)

        hidden = self.bert(
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            token_type_ids=moved["token_type_ids"],
            return_dict=True,
        )
        scores = self.quality_net(hidden.pooler_output)
        q = (scores * self.quality_weights).sum(dim=1)
        return {
            "pred": pred.cpu().tolist(),
            "confidence": confidence.cpu().tolist(),
            "quality": q.cpu().tolist(),
            "probs": probs.cpu().tolist(),
        }


def save_checkpoint(model, path, epoch):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, path)
