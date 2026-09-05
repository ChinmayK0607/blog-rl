from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from swarm_ctf_eval.arena_protocol import Broadcast, parse_broadcast
from swarm_ctf_eval.broadcast_priority import (
    PROMPT_VARIANTS,
    apply_prompt_variant,
    summarize_priority_rows,
)
from swarm_ctf_eval.communication_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.episode import EpisodeConfig
from swarm_ctf_eval.episode_protocol import episode_broadcast_prompt
from swarm_ctf_eval.live_rl_rollout import PolicyEndpoint, VLLMChoiceGenerator
from swarm_ctf_eval.rl_v3 import ArenaRLEnv


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe non-leaking broadcast fact-priority instructions."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(PROMPT_VARIANTS),
    )
    args = parser.parse_args()
    variants = tuple(args.variant or PROMPT_VARIANTS)
    manifest = json.loads((args.data_dir / "train.json").read_text(encoding="utf-8"))
    if args.pair_offset < 0:
        parser.error("--pair-offset must be nonnegative")
    if args.pairs < 1 or args.pair_offset + args.pairs > len(manifest["pairs"]):
        parser.error("--pair-offset and --pairs must select a nonempty manifest slice")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    endpoint = PolicyEndpoint(
        "broadcast-priority-diagnostic",
        args.policy_revision,
        args.model_name,
        (args.base_url,),
    )
    config = EpisodeConfig(
        horizon=2,
        communication_cost=0.0,
        invalid_broadcast_cost=0.0,
        invalid_action_cost=0.0,
    )
    rows = []
    async with VLLMChoiceGenerator(args.tokenizer) as generator:
        selected_pairs = manifest["pairs"][
            args.pair_offset : args.pair_offset + args.pairs
        ]
        for pair_index, pair in enumerate(selected_pairs, start=args.pair_offset):
            scenario = reconstruct_manifest_scenario(pair["critical"])
            env = ArenaRLEnv(size=scenario.size, config=config)
            env.reset_from_state(scenario.state)
            base_messages, _ = episode_broadcast_prompt(
                env,
                scenario.sender,
                permutation=pair_index,
            )
            target_observation = scenario.state.knowledge[scenario.sender][
                scenario.target
            ].to_dict()
            for repetition in range(args.repetitions):
                sampling_key = (
                    f"broadcast-priority-v1:pair-{pair_index}:rep-{repetition}"
                )
                messages_by_variant = {
                    variant: apply_prompt_variant(base_messages, variant)
                    for variant in variants
                }
                completions = await asyncio.gather(
                    *(
                        generator.generate(
                            endpoint,
                            messages_by_variant[variant],
                            sampling_key=sampling_key,
                        )
                        for variant in variants
                    )
                )
                for variant, completion in zip(variants, completions, strict=True):
                    parsed = parse_broadcast(
                        completion.text,
                        scenario.state,
                        scenario.sender,
                    )
                    valid = parsed.valid and isinstance(parsed.value, Broadcast)
                    broadcast = parsed.value if valid else None
                    facts = broadcast.facts if isinstance(broadcast, Broadcast) else ()
                    rows.append(
                        {
                            "pair_index": pair_index,
                            "repetition": repetition,
                            "variant": variant,
                            "sender": scenario.sender,
                            "receiver": scenario.receiver,
                            "target": scenario.target,
                            "target_observation": target_observation,
                            "protocol_valid": valid,
                            "errors": list(parsed.errors),
                            "target_fact_present": any(
                                fact.node == scenario.target for fact in facts
                            ),
                            "fact_count": len(facts),
                            "broadcast": (
                                broadcast.to_dict()
                                if isinstance(broadcast, Broadcast)
                                else None
                            ),
                            "sampling_key": sampling_key,
                            "prompt_sha256": hashlib.sha256(
                                json.dumps(
                                    messages_by_variant[variant],
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest(),
                            "request_sha256": completion.request_sha256,
                        }
                    )
            print(
                json.dumps(
                    {
                        "completed_pair": pair_index,
                        "completed_pairs": pair_index - args.pair_offset + 1,
                        "total_pairs": args.pairs,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    result = {
        "schema_version": "broadcast-priority-probe-v1",
        "policy_revision": args.policy_revision,
        "pair_offset": args.pair_offset,
        "variants": {variant: PROMPT_VARIANTS[variant] for variant in variants},
        "summary": summarize_priority_rows(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
