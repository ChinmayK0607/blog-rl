from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from swarm_ctf_eval.arena_protocol import parse_action, parse_broadcast
from swarm_ctf_eval.episode import EpisodeConfig
from swarm_ctf_eval.episode_protocol import episode_action_prompt, episode_broadcast_prompt
from swarm_ctf_eval.live_rl_rollout import PolicyEndpoint, VLLMChoiceGenerator
from swarm_ctf_eval.rl_v3 import ArenaRLEnv
from swarm_ctf_eval.structured_protocol import protocol_choices


async def replace_adapter(base_urls: tuple[str, ...], name: str, path: Path) -> None:
    async with httpx.AsyncClient(timeout=120.0) as client:
        unloads = await asyncio.gather(
            *(
                client.post(
                    f"{base_url.rstrip('/')}/v1/unload_lora_adapter",
                    json={"lora_name": name},
                )
                for base_url in base_urls
            )
        )
        for response in unloads:
            if response.status_code not in {200, 404}:
                response.raise_for_status()
        loads = await asyncio.gather(
            *(
                client.post(
                    f"{base_url.rstrip('/')}/v1/load_lora_adapter",
                    json={"lora_name": name, "lora_path": str(path)},
                )
                for base_url in base_urls
            )
        )
        for response in loads:
            response.raise_for_status()
        registries = await asyncio.gather(
            *(client.get(f"{base_url.rstrip('/')}/v1/models") for base_url in base_urls)
        )
    expected_path = str(path.resolve())
    for response in registries:
        response.raise_for_status()
        matches = [row for row in response.json()["data"] if row["id"] == name]
        if len(matches) != 1 or matches[0].get("root") != expected_path:
            raise RuntimeError(f"LoRA registry did not bind {name} to {expected_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe exact live Swarm Arena serving paths.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-name", default="swarm-probe")
    parser.add_argument("--base-url", action="append", required=True)
    args = parser.parse_args()

    base_urls = tuple(args.base_url)
    await replace_adapter(base_urls, args.adapter_name, args.adapter)
    env = ArenaRLEnv(
        seed=7_000_003,
        size=12,
        config=EpisodeConfig(
            horizon=2,
            communication_cost=0.0,
            invalid_broadcast_cost=0.0,
            invalid_action_cost=0.0,
        ),
    )
    env.reset(7_000_003)
    endpoint = PolicyEndpoint(
        "probe-policy",
        "probe-revision",
        args.adapter_name,
        base_urls,
    )
    async with VLLMChoiceGenerator(args.tokenizer) as generator:
        broadcast_messages, _ = episode_broadcast_prompt(env, "blue-0", permutation=17)
        broadcast = await generator.generate(
            endpoint,
            broadcast_messages,
            sampling_key="swarm-probe:blue-0:0:BROADCAST",
        )
        parsed_broadcast = parse_broadcast(broadcast.text, env._require_state(), "blue-0")
        if not parsed_broadcast.valid:
            raise RuntimeError(f"broadcast probe failed: {parsed_broadcast.errors}")

        broadcasts = {}
        for agent_id in sorted(env.observations()):
            messages, _ = episode_broadcast_prompt(env, agent_id, permutation=19)
            parsed = parse_broadcast(protocol_choices(messages)[0], env._require_state(), agent_id)
            if not parsed.valid or parsed.value is None:
                raise RuntimeError(f"could not construct legal broadcast for {agent_id}")
            broadcasts[agent_id] = parsed.value
        env.broadcast_phase(broadcasts)
        action_messages, displayed = episode_action_prompt(env, "blue-0", permutation=23)
        action = await generator.generate(
            endpoint,
            action_messages,
            sampling_key="swarm-probe:blue-0:0:ACT",
        )
        parsed_action = parse_action(action.text, displayed)
        if not parsed_action.valid:
            raise RuntimeError(f"action probe failed: {parsed_action.errors}")

    print(
        json.dumps(
            {
                "status": "passed",
                "servers": len(base_urls),
                "broadcast": {
                    "text": broadcast.text,
                    "prompt_tokens": len(broadcast.prompt_ids),
                    "completion_tokens": len(broadcast.completion_ids),
                },
                "action": {
                    "text": action.text,
                    "prompt_tokens": len(action.prompt_ids),
                    "completion_tokens": len(action.completion_ids),
                    "legal_choices": len(displayed),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
