from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
from renderers import Qwen3Renderer, Qwen3RendererConfig
from swarm_ctf_eval.episode_protocol import episode_broadcast_prompt
from swarm_ctf_eval.rl_v3 import ArenaRLEnv
from swarm_ctf_eval.structured_protocol import completion_allowed_token_ids, protocol_choices
from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def generate(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt_ids: list[int],
    choices: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    response = await client.post(
        "/inference/v1/generate",
        json={
            "model": model,
            "token_ids": prompt_ids,
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 128,
                "logprobs": 1,
                "seed": seed,
                "structured_outputs": {"choice": list(choices)},
            },
        },
    )
    response.raise_for_status()
    return response.json()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe exact legal-choice constrained vLLM rollouts.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual_sha256 = sha256_file(args.adapter / "adapter_model.safetensors")
    if actual_sha256 != args.adapter_sha256:
        raise ValueError(f"adapter checksum mismatch: {actual_sha256}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = Qwen3Renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False))
    env = ArenaRLEnv(seed=17, size=12)
    env.reset()
    messages, _ = episode_broadcast_prompt(env, "blue-0")
    choices = protocol_choices(messages)
    prompt_ids = renderer.render_ids(messages, add_generation_prompt=True)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=120.0) as client:
        response = await generate(
            client,
            model=args.model,
            prompt_ids=prompt_ids,
            choices=choices,
            seed=20260813,
        )

    choice = response["choices"][0]
    completion_ids = list(choice["token_ids"])
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    choice_token_ids = [
        [*tokenizer.encode(value, add_special_tokens=False), tokenizer.eos_token_id]
        for value in choices
    ]
    result = {
        "adapter_sha256": actual_sha256,
        "choice_count": len(choices),
        "completion": completion,
        "completion_ids": completion_ids,
        "completion_logprobs": [item["logprob"] for item in choice["logprobs"]["content"]],
        "prompt_ids": prompt_ids,
        "raw_response": response,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        result["allowed_token_ids"] = completion_allowed_token_ids(completion_ids, choice_token_ids)
    except ValueError:
        matching_text = [value for value in choices if value == completion]
        print(
            json.dumps(
                {
                    "completion": completion,
                    "completion_ids": completion_ids,
                    "matching_choice": matching_text,
                    "isolated_ids": tokenizer.encode(completion, add_special_tokens=False),
                    "raw_output": str(args.output),
                },
                sort_keys=True,
            )
        )
        raise
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "choice_count": len(choices),
                "completion": completion,
                "completion_tokens": len(completion_ids),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
