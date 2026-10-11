import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utilts import read_jsonl


@dataclass
class NewsItem:
    uid: str
    index: int
    text: str
    image_files: list
    label: int
    label_str: str


def load_news_csv(csv_path, label_map, limit=None):
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        if limit is not None and len(items) >= limit:
            break
        label = int(row["label"])
        items.append(
            NewsItem(
                uid=str(row["post_id"]),
                index=len(items),
                text=str(row["content"]),
                image_files=ast.literal_eval(row["image_files"]),
                label=label,
                label_str=label_map[str(label)],
            )
        )
    return items


def load_image(images_dir, filename):
    return Image.open(Path(images_dir) / filename).convert("RGB")


class ReasoningDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records = read_jsonl(jsonl_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "uid": record["uid"],
            "news_text": record["news_text"],
            "reasons": [str(r) for r in record["reasons"]],
            "label": 1 if record["true_label"] == "FAKE" else 0,
        }

    def collate(self, batch):
        texts = []
        starts = []
        labels = []
        for item in batch:
            texts.append(item["news_text"] + "[SEP]" + " ".join(item["reasons"]))
            starts.append(len(self.tokenizer.tokenize(item["news_text"])) + 1)
            labels.append(item["label"])

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"],
            "reason_start_idx": torch.tensor(starts, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
