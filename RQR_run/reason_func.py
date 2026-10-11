import argparse
import time

import cot_prompt_list
from data_utils import load_news_csv
from llm_client import build_client, chat_json
from resp_process import normalize_judgement
from utilts import DATASETS, OUTPUT_DIR, append_jsonl, read_jsonl
from vision_utils import text_message

SYSTEM_PROMPT = "You are a fake news judge. Return JSON only."


def judge_veracity(client, news_text, visual_facts):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                text_message(cot_prompt_list.build_stage2_prompt(visual_facts, news_text))
            ],
        },
    ]
    return normalize_judgement(chat_json(client, "stage2", messages))


def run_batch(dataset, split, limit=None, sleep=1.0, facts_path=None, out_path=None):
    cfg = DATASETS[dataset]
    csv_path = cfg["train_csv"] if split == "train" else cfg["test_csv"]
    items = load_news_csv(csv_path, cfg["label_map"], limit)
    facts_path = facts_path or str(OUTPUT_DIR / ("%s_%s_facts.jsonl" % (dataset, split)))
    facts_by_uid = {rec["uid"]: rec["visual_facts"] for rec in read_jsonl(facts_path)}
    out_path = out_path or str(OUTPUT_DIR / ("%s_%s_reasoning.jsonl" % (dataset, split)))
    client = build_client("stage2")
    open(out_path, "w").close()

    for item in items:
        record = {"uid": item.uid, "news_text": item.text, "true_label": item.label_str}
        record.update(judge_veracity(client, item.text, facts_by_uid[item.uid]))
        append_jsonl(out_path, record)
        time.sleep(sleep)
    print("%d records -> %s" % (len(items), out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fakeddit", choices=sorted(DATASETS))
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--facts", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_batch(args.dataset, args.split, args.limit, args.sleep, args.facts, args.out)


if __name__ == "__main__":
    main()
