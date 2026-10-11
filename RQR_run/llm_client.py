import json

from openai import OpenAI

from utilts import get_api_key, stage_config


def build_client(stage):
    cfg = stage_config(stage)
    return OpenAI(api_key=get_api_key(stage), base_url=cfg["base_url"])


def chat_json(client, stage, messages):
    cfg = stage_config(stage)
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
