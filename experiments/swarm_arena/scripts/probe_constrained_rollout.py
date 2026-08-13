from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
from renderers import Qwen3Renderer, Qwen3RendererConfig
from swarm_ctf_eval.episode import EMPTY_BROADCAST
from swarm_ctf_eval.episode_protocol import episode_action_prompt, episode_broadcast_prompt
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
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    actual_sha256 = sha256_file(args.adapter / "adapter_model.safetensors")
    if actual_sha256 != args.adapter_sha256:
        raise ValueError(f"adapter checksum mismatch: {actual_sha256}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = Qwen3Renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False))

    rows = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=120.0) as client:
        for sample_index in range(args.samples):
            seed = 17 + sample_index // 2
            agent_id = f"blue-{sample_index % 4}"
            env = ArenaRLEnv(seed=seed, size=12)
            env.reset()
            if sample_index % 2 == 0:
                phase = "BROADCAST"
                messages, _ = episode_broadcast_prompt(env, agent_id, permutation=sample_index)
            else:
                phase = "ACT"
                env.broadcast_phase({agent: EMPTY_BROADCAST for agent in env._require_state().agents})
                messages, _ = episode_action_prompt(env, agent_id, permutation=sample_index)
            choices = protocol_choices(messages)
            prompt_ids = renderer.render_ids(messages, add_generation_prompt=True)
            response = await generate(
                client,
                model=args.model,
                prompt_ids=prompt_ids,
                choices=choices,
                seed=20260813 + sample_index,
            )
            choice = response["choices"][0]
            completion_ids = list(choice["token_ids"])
            choice_token_ids = [
                [*tokenizer.encode(value, add_special_tokens=False), tokenizer.eos_token_id]
                for value in choices
            ]
            rows.append(
                {
                    "agent_id": agent_id,
                    "allowed_token_ids": completion_allowed_token_ids(completion_ids, choice_token_ids),
                    "choice_count": len(choices),
                    "completion": tokenizer.decode(completion_ids, skip_special_tokens=False),
                    "completion_ids": completion_ids,
                    "completion_logprobs": [item["logprob"] for item in choice["logprobs"]["content"]],
                    "phase": phase,
                    "prompt_ids": prompt_ids,
                    "seed": seed,
                }
            )

    result = {"adapter_sha256": actual_sha256, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "action_samples": sum(row["phase"] == "ACT" for row in rows),
                "broadcast_samples": sum(row["phase"] == "BROADCAST" for row in rows),
                "completion_tokens": sum(len(row["completion_ids"]) for row in rows),
                "output": str(args.output),
                "samples": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
