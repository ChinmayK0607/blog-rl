from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from renderers import Qwen3Renderer, Qwen3RendererConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic base-model preservation responses.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-tokens", type=int, default=1400)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    renderer = Qwen3Renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()
    prompts = load_rows(args.prompts)
    output_rows = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        rendered = [renderer.render_ids(row["messages"], add_generation_prompt=True) for row in batch]
        encoded = tokenizer.pad(
            [{"input_ids": ids} for ids in rendered],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_width = encoded["input_ids"].shape[1]
        for row, tokens in zip(batch, generated, strict=True):
            completion = tokens[prompt_width:]
            eos_seen = tokenizer.eos_token_id in completion.tolist()
            response = tokenizer.decode(completion, skip_special_tokens=True).strip()
            messages = [*row["messages"], {"role": "assistant", "content": response}]
            token_count = len(renderer.render_ids(messages, add_generation_prompt=False))
            payload = json.dumps(messages, sort_keys=True, separators=(",", ":"))
            output_rows.append(
                {
                    "id": hashlib.sha256(f"base-replay:{payload}".encode()).hexdigest(),
                    "messages": messages,
                    "source": "base_behavior_replay",
                    "metadata_json": json.dumps(
                        {
                            "dataset_version": "arena-warmstart-v3",
                            "model": args.model,
                            "source_prompt_id": row["id"],
                            "token_count": token_count,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "truncated": not eos_seen or token_count > args.max_sequence_tokens or not response,
                }
            )
        if start == 0 or (start // args.batch_size + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "processed": len(output_rows),
                        "usable": sum(not row["truncated"] for row in output_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "examples": len(output_rows),
                "usable": sum(not row["truncated"] for row in output_rows),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
