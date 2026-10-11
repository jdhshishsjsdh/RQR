import argparse
import json

import torch
from transformers import AutoTokenizer

from detector_func import RQRDetector
from extract_func import extract_visual_facts
from llm_client import build_client
from reason_func import judge_veracity
from utilts import DATASETS, DETECTOR, OUTPUT_DIR, pick_device


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_detector(news_text, reasons, checkpoint, device_name):
    device = pick_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR["encoder"])
    model = RQRDetector.from_checkpoint(checkpoint, DETECTOR["encoder"]).to(device)

    encoded = tokenizer(
        [news_text + "[SEP]" + " ".join(reasons)],
        truncation=True,
        padding="max_length",
        max_length=DETECTOR["max_len"],
        return_tensors="pt",
    )
    batch = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded["token_type_ids"],
        "reason_start_idx": torch.tensor(
            [len(tokenizer.tokenize(news_text)) + 1], dtype=torch.long
        ),
    }
    result = model.predict(batch, device)
    return {
        "label": "FAKE" if result["pred"][0] == 1 else "REAL",
        "confidence": float(result["confidence"][0]),
        "quality": float(result["quality"][0]),
        "prob_real": float(result["probs"][0][0]),
        "prob_fake": float(result["probs"][0][1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fakeddit", choices=sorted(DATASETS))
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--facts-file", default=None)
    parser.add_argument("--reasons-file", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.facts_file:
        facts = load_json(args.facts_file)
    else:
        facts = extract_visual_facts(build_client("stage1"), args.image)

    if args.reasons_file:
        judgement = load_json(args.reasons_file)
    else:
        judgement = judge_veracity(build_client("stage2"), args.text, facts)
    reasons = judgement["reasons"]

    detection = run_detector(args.text, reasons, args.checkpoint, args.device)

    print("news text : %s" % args.text)
    print("stage1    : %d entities, %d unary, %d binary, %d negated" % (
        len(facts["entities"]),
        len(facts["unary_facts"]),
        len(facts["binary_facts"]),
        len(facts["negated_facts"]),
    ))
    print("stage2    : %s" % judgement["final_answer"])
    for index, reason in enumerate(reasons, 1):
        print("            %d. %s" % (index, reason))
    print("detector  : %s  confidence %.4f  quality %.4f" % (
        detection["label"], detection["confidence"], detection["quality"]
    ))
