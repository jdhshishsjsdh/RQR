import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import ReasoningDataset
from detector_func import RQRDetector, save_checkpoint
from utilts import DATASETS, DETECTOR, OUTPUT_DIR, RUN_DIR, TRAIN, pick_device, set_seed


def metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, labels=[0, 1]
    )
    return {
        "acc": float(accuracy_score(true_labels, predictions) * 100),
        "real_f1": float(f1[0] * 100),
        "fake_f1": float(f1[1] * 100),
        "real_p": float(precision[0] * 100),
        "real_r": float(recall[0] * 100),
        "fake_p": float(precision[1] * 100),
        "fake_r": float(recall[1] * 100),
    }


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    losses = []
    predictions = []
    true_labels = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        optimizer.zero_grad()
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        losses.append(float(outputs.loss.item()))
        predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
        true_labels.extend(batch["labels"].cpu().tolist())
    return float(np.mean(losses)), float(accuracy_score(true_labels, predictions) * 100)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
        true_labels.extend(batch["labels"].cpu().tolist())
    return metrics(true_labels, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weibo", choices=sorted(DATASETS))
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--test-jsonl", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    train_jsonl = args.train_jsonl or str(
        OUTPUT_DIR / ("%s_train_reasoning.jsonl" % args.dataset)
    )
    test_jsonl = args.test_jsonl or str(
        OUTPUT_DIR / ("%s_test_reasoning.jsonl" % args.dataset)
    )
    output_dir = args.output or str(RUN_DIR / args.dataset)

    set_seed(TRAIN["seed"])
    device = pick_device(TRAIN["device"])
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR["encoder"])

    train_set = ReasoningDataset(train_jsonl, tokenizer, cfg["max_len"])
    test_set = ReasoningDataset(test_jsonl, tokenizer, cfg["max_len"])
    train_loader = DataLoader(
        train_set, batch_size=TRAIN["batch_size"], shuffle=True, collate_fn=train_set.collate
    )
    test_loader = DataLoader(
        test_set, batch_size=TRAIN["batch_size"], shuffle=False, collate_fn=test_set.collate
    )

    model = RQRDetector(encoder_path=DETECTOR["encoder"]).to(device)
    optimizer = AdamW(model.parameters(), lr=TRAIN["lr"])
    total_steps = len(train_loader) * TRAIN["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAIN["warmup_steps"],
        num_training_steps=total_steps,
    )

    best_acc = 0.0
    for epoch in range(1, TRAIN["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        result = evaluate(model, test_loader, device)
        print(
            "epoch %2d  loss %.4f  train_acc %.2f  test_acc %.2f  fake_f1 %.2f  real_f1 %.2f"
            % (epoch, train_loss, train_acc, result["acc"], result["fake_f1"], result["real_f1"])
        )
        save_checkpoint(model, "%s/last.pt" % output_dir, epoch)
        if result["acc"] > best_acc:
            best_acc = result["acc"]
            save_checkpoint(model, "%s/best.pt" % output_dir, epoch)
    print("best test accuracy %.2f -> %s" % (best_acc, output_dir))


if __name__ == "__main__":
    main()
