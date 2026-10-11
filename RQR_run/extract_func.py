import argparse
import time

import cot_prompt_list
from data_utils import load_image, load_news_csv
from llm_client import build_client, chat_json
from resp_process import clean_facts
from utilts import DATASETS, OUTPUT_DIR, append_jsonl
from vision_utils import encode_image_to_base64, image_message, text_message

SYSTEM_PROMPT = (
    "You are a structured visual fact extractor. Your task is to analyze the image "
    "and output all observable facts in strict JSON format. Do not infer anything "
    "that is not explicitly shown in the image."
)


def extract_visual_facts(client, image):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                text_message(cot_prompt_list.STAGE1_VISUAL_FACTS),
                image_message(encode_image_to_base64(image)),
            ],
        },
    ]
    return clean_facts(chat_json(client, "stage1", messages))


def run_batch(dataset, split, limit=None, sleep=1.0, out_path=None):
    cfg = DATASETS[dataset]
    csv_path = cfg["train_csv"] if split == "train" else cfg["test_csv"]
    items = load_news_csv(csv_path, cfg["label_map"], limit)
    out_path = out_path or str(OUTPUT_DIR / ("%s_%s_facts.jsonl" % (dataset, split)))
    client = build_client("stage1")
    open(out_path, "w").close()

    for item in items:
        image = load_image(cfg["images"], item.image_files[0])
        record = {
            "uid": item.uid,
            "news_text": item.text,
            "true_label": item.label_str,
            "visual_facts": extract_visual_facts(client, image),
        }
        append_jsonl(out_path, record)
        time.sleep(sleep)
    print("%d records -> %s" % (len(items), out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fakeddit", choices=sorted(DATASETS))
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_batch(args.dataset, args.split, args.limit, args.sleep, args.out)


if __name__ == "__main__":
    main()
