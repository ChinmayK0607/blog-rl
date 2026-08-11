from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from swarm_ctf_eval.sft_metrics import validate_dataset_response
from transformers import AutoModelForCausalLM, AutoTokenizer


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def score(
    model_path: str,
    adapter_path: str | None,
    dataset_id: str,
    split: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model = model.to("cuda")
    model.eval()
    dataset = load_dataset(dataset_id, split=split)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    phase_metrics: dict[str, Counter[str]] = {
        "BROADCAST": Counter(),
        "ACT": Counter(),
    }
    for start in range(0, len(dataset), batch_size):
        batch = [dict(dataset[index]) for index in range(start, min(start + batch_size, len(dataset)))]
        prompts = [render_prompt(tokenizer, row["messages"][:-1]) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        generated = model.generate(
            **encoded,
            max_new_tokens=224,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for index, row in enumerate(batch):
            raw = tokenizer.decode(
                generated[index, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            result = validate_dataset_response(row, raw)
            phase = row["metadata"]["phase"]
            phase_counts[phase] += 1
            for key, value in result.items():
                counts[key] += int(value)
                phase_metrics[phase][key] += int(value)
            rows.append(
                {
                    "id": row["id"],
                    "phase": phase,
                    "response": raw,
                    "target": row["messages"][-1]["content"],
                    **result,
                }
            )
    total = len(rows)
    summary: dict[str, Any] = {
        "model": model_path,
        "adapter": adapter_path,
        "dataset": dataset_id,
        "split": split,
        "examples": total,
        **{key: counts[key] / total for key in ("schema_valid", "supported", "legal", "ordered_exact", "exact")},
    }
    for phase in ("BROADCAST", "ACT"):
        summary[phase.lower()] = {
            "examples": phase_counts[phase],
            **{
                key: phase_metrics[phase][key] / max(1, phase_counts[phase])
                for key in ("schema_valid", "supported", "legal", "ordered_exact", "exact")
            },
        }
    summary["selection_score"] = statistics.mean([summary["broadcast"]["exact"], summary["act"]["exact"]])
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", default="CK0607/swarm-arena-sft-v2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = score(args.model, args.adapter, args.dataset, args.split, args.batch_size)
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
