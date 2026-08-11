from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic broad prompts for base-behavior replay.")
    parser.add_argument("--examples", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    stream = stream.shuffle(seed=20260905, buffer_size=10_000)
    rows = []
    seen = set()
    for item in stream:
        user = next((message["content"] for message in item["messages"] if message["role"] == "user"), None)
        if not isinstance(user, str) or not 80 <= len(user) <= 1200:
            continue
        digest = hashlib.sha256(user.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rows.append({"id": f"replay-prompt-{digest}", "messages": [{"role": "user", "content": user}]})
        if len(rows) >= args.examples:
            break
    if len(rows) != args.examples:
        raise RuntimeError(f"only extracted {len(rows)} of {args.examples} replay prompts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"examples": len(rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
