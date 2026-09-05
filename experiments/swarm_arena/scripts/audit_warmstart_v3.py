from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from renderers import Qwen3Renderer, Qwen3RendererConfig
from transformers import AutoTokenizer


def load_split(root: Path, split: str):
    return load_dataset("parquet", data_files=str(root / f"{split}.parquet"), split="train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a local replay-protected warm-start dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    renderer = Qwen3Renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False))
    ids = []
    report = {"splits": {}, "max_tokens": args.max_tokens}
    for split in ("train", "validation"):
        dataset = load_split(args.dataset, split)
        sources: Counter[str] = Counter()
        lengths = []
        for row in dataset:
            ids.append(row["id"])
            sources[row["source"]] += 1
            if not row["messages"] or row["messages"][-1]["role"] != "assistant":
                raise AssertionError(f"{row['id']} does not end in an assistant response")
            lengths.append(len(renderer.render_ids(row["messages"], add_generation_prompt=False)))
        if max(lengths) > args.max_tokens:
            raise AssertionError(f"{split} has {max(lengths)} tokens, above {args.max_tokens}")
        report["splits"][split] = {
            "examples": len(dataset),
            "by_source": dict(sorted(sources.items())),
            "min_tokens": min(lengths),
            "max_tokens_observed": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
        }
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate row ids across dataset splits")
    report["ids_sha256"] = hashlib.sha256("".join(sorted(ids)).encode()).hexdigest()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
