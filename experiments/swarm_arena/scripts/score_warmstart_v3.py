from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from renderers import Qwen3Renderer, Qwen3RendererConfig
from swarm_ctf_eval.warmstart_v3 import validate_warmstart_response
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Score the frozen warm-start v3 validation split.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    renderer = Qwen3Renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False))
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to("cuda")
    model.eval()
    dataset = load_dataset("parquet", data_files=str(args.dataset / "validation.parquet"), split="train")
    rows = []
    counts: Counter[str] = Counter()
    grouped: dict[str, Counter[str]] = {}
    group_sizes: Counter[str] = Counter()
    for start in range(0, len(dataset), args.batch_size):
        batch = [dict(dataset[index]) for index in range(start, min(start + args.batch_size, len(dataset)))]
        prompts = [renderer.render_ids(row["messages"][:-1], add_generation_prompt=True) for row in batch]
        encoded = tokenizer.pad([{"input_ids": ids} for ids in prompts], padding=True, return_tensors="pt").to("cuda")
        generated = model.generate(
            **encoded,
            max_new_tokens=224,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_width = encoded["input_ids"].shape[1]
        for row, tokens in zip(batch, generated, strict=True):
            raw = tokenizer.decode(tokens[prompt_width:], skip_special_tokens=True).strip()
            metrics = validate_warmstart_response(row, raw)
            metadata = json.loads(row["metadata_json"])
            group = metadata.get("phase", row["source"])
            grouped.setdefault(group, Counter())
            group_sizes[group] += 1
            for key, value in metrics.items():
                counts[key] += int(value)
                grouped[group][key] += int(value)
            rows.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "group": group,
                    "response": raw,
                    "target": row["messages"][-1]["content"],
                    **metrics,
                }
            )
    metrics = ("schema_valid", "grounded", "legal", "exact")
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": str(args.dataset),
        "examples": len(rows),
        **{key: counts[key] / len(rows) for key in metrics},
        "groups": {
            group: {"examples": group_sizes[group], **{key: values[key] / group_sizes[group] for key in metrics}}
            for group, values in sorted(grouped.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
