from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from renderers import Qwen3Renderer, Qwen3RendererConfig
from swarm_ctf_eval.regression import FROZEN_REGRESSION_CASES, summarize_regression_rows, validate_response
from swarm_ctf_eval.regression_v2 import (
    FROZEN_REGRESSION_V2_CASES,
    summarize_v2_rows,
    validate_v2_response,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def score(
    model_path: str,
    adapter_path: str | None,
    batch_size: int,
    device: str,
    suite: str = "v1",
) -> tuple[list[dict], dict]:
    cases = FROZEN_REGRESSION_CASES if suite == "v1" else FROZEN_REGRESSION_V2_CASES
    validator = validate_response if suite == "v1" else validate_v2_response
    summarizer = summarize_regression_rows if suite == "v1" else summarize_v2_rows
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    renderer = Qwen3Renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False))
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
    )
    if adapter_path:
        # Text-only scoring: do not import an incompatible optional torchvision.
        from transformers.utils import import_utils

        import_utils._torchvision_available = False
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model = model.to(device)
    model.eval()
    rows = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        prompt_ids = [renderer.render_ids(list(case.messages), add_generation_prompt=True) for case in batch]
        encoded = tokenizer.pad(
            [{"input_ids": input_ids} for input_ids in prompt_ids], padding=True, return_tensors="pt"
        ).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_width = encoded["input_ids"].shape[1]
        for case, output in zip(batch, generated, strict=True):
            raw = tokenizer.decode(output[prompt_width:], skip_special_tokens=True).strip()
            rows.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "response": raw,
                    "expected": case.expected,
                    **validator(case, raw),
                }
            )
    summary = {
        "model": model_path,
        "adapter": adapter_path,
        "device": device,
        **summarizer(rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the frozen non-arena regression suite.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--suite", choices=("v1", "v2"), default="v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = score(args.model, args.adapter, args.batch_size, resolve_device(args.device), args.suite)
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
