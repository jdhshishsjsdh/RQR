import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RUN_DIR = ROOT / "runs"

STAGE1 = {
    "base_url": "",
    "api_key_env": "",
    "model": "",
    "vision": True,
    "temperature": 0.1,
    "top_p": 1.0,
    "request_interval": 1.0,
}

STAGE2 = {
    "base_url": "",
    "api_key_env": "",
    "model": "",
    "vision": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "request_interval": 1.0,
}

DATASETS = {
    "weibo": {
        "train_csv": str(DATA_DIR / "weibo" / "train.csv"),
        "test_csv": str(DATA_DIR / "weibo" / "test.csv"),
        "images": str(DATA_DIR / "weibo" / "images"),
        "max_len": 512,
        "label_map": {"0": "REAL", "1": "FAKE"},
    },
    "fakeddit": {
        "train_csv": str(DATA_DIR / "fakeddit" / "train.csv"),
        "test_csv": str(DATA_DIR / "fakeddit" / "test.csv"),
        "images": str(DATA_DIR / "fakeddit" / "images"),
        "max_len": 128,
        "label_map": {"0": "REAL", "1": "FAKE"},
    },
}

DETECTOR = {
    "encoder": "",
    "max_len": 512,
    "num_labels": 2,
    "dropout": 0.1,
    "quality_dim": 3,
    "projection_dim": 128,
    "temperature": 0.07,
    "weight_normalization": "mean",
    "detach_weights": True,
}

TRAIN = {
    "batch_size": 16,
    "lr": 2e-5,
    "epochs": 20,
    "lambda_con": 0.1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "warmup_steps": 0,
    "device": "auto",
}


def stage_config(stage):
    return STAGE1 if stage == "stage1" else STAGE2


def get_api_key(stage):
    return os.environ[stage_config(stage)["api_key_env"]]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


