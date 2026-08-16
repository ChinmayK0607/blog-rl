from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from probe_live_rollout import replace_adapter
from swarm_ctf_eval.arena_protocol import parse_broadcast
from swarm_ctf_eval.episode import EpisodeConfig
from swarm_ctf_eval.episode_protocol import episode_action_prompt, episode_broadcast_prompt
from swarm_ctf_eval.live_rl_rollout import PolicyEndpoint, VLLMChoiceGenerator
from swarm_ctf_eval.rl_v3 import ArenaRLEnv
from swarm_ctf_eval.structured_protocol import protocol_choices


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a fresh, exact-runtime, four-policy numerical-parity probe."
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-name", default="swarm-parity-probe")
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 8 or args.samples % 8:
        parser.error("samples must be a positive multiple of eight and at least eight")
    if len(args.base_url) != 3:
        parser.error("runtime parity capture requires exactly three rollout servers")
    from probe_constrained_rollout import sha256_file

    actual_adapter_sha256 = sha256_file(args.adapter / "adapter_model.safetensors")
    if actual_adapter_sha256 != args.adapter_sha256:
        raise ValueError("runtime parity adapter checksum mismatch")
    base_urls = tuple(args.base_url)
    await replace_adapter(base_urls, args.adapter_name, args.adapter)

    samples = []
    async with VLLMChoiceGenerator(args.tokenizer) as generator:
        for sample_index in range(args.samples):
            seed = 7_100_000 + sample_index // 8
            agent_index = (sample_index // 2) % 4
            agent_id = f"blue-{agent_index}"
            phase = "BROADCAST" if sample_index % 2 == 0 else "ACT"
            env = ArenaRLEnv(
                seed=seed,
                size=12 + sample_index % 2,
                config=EpisodeConfig(
                    horizon=4 + sample_index % 2,
                    communication_cost=0.0,
                    invalid_broadcast_cost=0.0,
                    invalid_action_cost=0.0,
                ),
            )
            env.reset(seed)
            if phase == "BROADCAST":
                messages, _ = episode_broadcast_prompt(
                    env, agent_id, permutation=101 + sample_index
                )
            else:
                broadcasts = {}
                for current_agent in sorted(env.observations()):
                    current_messages, _ = episode_broadcast_prompt(
                        env, current_agent, permutation=211 + sample_index
                    )
                    parsed = parse_broadcast(
                        protocol_choices(current_messages)[0],
                        env._require_state(),
                        current_agent,
                    )
                    if not parsed.valid or parsed.value is None:
                        raise RuntimeError("could not construct parity-probe broadcasts")
                    broadcasts[current_agent] = parsed.value
                env.broadcast_phase(broadcasts)
                messages, _ = episode_action_prompt(
                    env, agent_id, permutation=307 + sample_index
                )
            endpoint = PolicyEndpoint(
                policy_id=f"blue-policy-{agent_index}",
                revision="runtime-parity-probe",
                model_name=args.adapter_name,
                base_urls=(base_urls[sample_index % len(base_urls)],),
            )
            completion = await generator.generate(
                endpoint,
                messages,
                sampling_key=f"runtime-parity:{sample_index}:{phase}:{agent_id}",
            )
            samples.append(
                {
                    "decision_id": f"runtime-parity-{sample_index}",
                    "game_id": f"runtime-parity-seed-{seed}",
                    "replica_index": 0,
                    "agent_id": agent_id,
                    "policy_id": endpoint.policy_id,
                    "policy_slot": agent_index,
                    "phase": phase,
                    "turn": 0,
                    "prompt_ids": list(completion.prompt_ids),
                    "completion_ids": list(completion.completion_ids),
                    "completion_logprobs": list(completion.logprobs),
                    "allowed_token_ids": [
                        list(row) for row in completion.allowed_token_ids
                    ],
                    "serving_allowed_logprobs": [
                        [list(value) for value in row]
                        for row in completion.serving_allowed_logprobs
                    ],
                    "request_sha256": completion.request_sha256,
                    "server_url": endpoint.base_urls[0],
                }
            )

    result = {
        "version": "arena-runtime-parity-probe-v1",
        "adapter_sha256": actual_adapter_sha256,
        "servers": len(base_urls),
        "base_urls": list(base_urls),
        "samples": samples,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite runtime probe: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(samples),
                "completion_tokens": sum(
                    len(row["completion_ids"]) for row in samples
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
