import json
import os


def append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    return records
