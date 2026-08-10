from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .arena import ARENA_VERSION, Action, GameState, NodeObservation, step
from .arena_generation import GENERATOR_VERSION, generate_mechanics_state, generate_state
from .arena_oracle import deterministic_policy, local_policy_action, solve_joint_action
from .arena_protocol import (
    ARENA_PROMPT_VERSION,
    Broadcast,
    action_prompt,
    broadcast_prompt,
    encode_action,
    encode_broadcast,
    parse_action,
    parse_broadcast,
)
from .arena_splits import FROZEN_EVAL_SEEDS


DATASET_VERSION = "arena-sft-v2"


def split_for_seed(seed: int) -> str:
    if seed in FROZEN_EVAL_SEEDS:
        raise ValueError(f"seed {seed} is reserved for frozen evaluation")
    bucket = int(hashlib.sha256(f"{GENERATOR_VERSION}:{seed}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def oracle_broadcast(state: GameState, agent_id: str, intent: Action) -> Broadcast:
    agent = state.agents[agent_id]
    memory = state.knowledge[agent_id]
    candidates = list(memory.values())

    def priority(fact: NodeObservation) -> tuple[int, int, int, str]:
        return (
            int(fact.node == intent.target),
            int(fact.status in {"COMPROMISED", "EXPOSED"} or fact.critical),
            fact.observed_turn,
            fact.node,
        )

    candidates.sort(key=priority, reverse=True)
    # Secure, non-critical friendly nodes rarely change another agent's choice.
    useful = [
        fact
        for fact in candidates
        if fact.node == intent.target
        or fact.owner != agent.team
        or fact.status != "SECURE"
        or fact.critical
    ][:3]
    request = int(agent.resource == 0 and intent.kind == "WAIT")
    declared = None if intent.kind == "WAIT" else intent
    return Broadcast(tuple(useful), declared, request)


def _row_id(messages: list[dict[str, str]], metadata: dict[str, Any]) -> str:
    payload = json.dumps({"messages": messages, "metadata": metadata}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _example(
    prompt: list[dict[str, str]], target: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    messages = [*prompt, {"role": "assistant", "content": target}]
    row = {"messages": messages, "metadata": metadata}
    row["id"] = _row_id(messages, metadata)
    return row


def generate_seed_examples(seed: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    state = generate_state(seed)
    red = deterministic_policy(state, "RED")
    blue_agents = sorted(agent.id for agent in state.agents.values() if agent.team == "BLUE")
    blue_local = {agent_id: local_policy_action(state, agent_id) for agent_id in blue_agents}
    solution = solve_joint_action(state, "BLUE", red)
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    split = split_for_seed(seed)

    broadcasts = {
        agent_id: oracle_broadcast(state, agent_id, action)
        for agent_id, action in blue_local.items()
    }
    intent_counts = Counter(
        (action.kind, action.target)
        for action in blue_local.values()
        if action.kind != "WAIT"
    )
    for index, agent_id in enumerate(sorted(blue_local)):
        prompt, _ = broadcast_prompt(state, agent_id, seed + index)
        target = encode_broadcast(broadcasts[agent_id])
        validation = parse_broadcast(target, state, agent_id)
        if not validation.valid:
            raise AssertionError(f"invalid oracle broadcast for {seed}/{agent_id}: {validation.errors}")
        rows.append(
            _example(
                prompt,
                target,
                {
                    "dataset_version": DATASET_VERSION,
                    "arena_version": ARENA_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "prompt_version": ARENA_PROMPT_VERSION,
                    "seed": seed,
                    "split": split,
                    "phase": "BROADCAST",
                    "agent_id": agent_id,
                    "label_source": "deterministic_observation_policy",
                },
            )
        )

    for index, agent_id in enumerate(sorted(blue_local)):
        acceptable = solution.acceptable_for(agent_id)
        if len(acceptable) != 1:
            rejected["ambiguous_optimal_action"] += 1
            continue
        target_action = next(iter(acceptable))
        if blue_local[agent_id] != target_action:
            # A globally optimal label that contradicts the deterministic
            # prompt-visible policy may depend on hidden information. It is not a
            # safe decentralized SFT target.
            rejected["not_prompt_identifiable"] += 1
            continue
        local_intent = blue_local[agent_id]
        if local_intent.kind != "WAIT" and intent_counts[(local_intent.kind, local_intent.target)] > 1:
            rejected["conflicting_declared_intent"] += 1
            continue
        inbox = [
            {"sender": sender, "broadcast": broadcasts[sender].to_dict()}
            for sender in sorted(broadcasts)
            if sender != agent_id
        ]
        prompt, displayed = action_prompt(state, agent_id, inbox, seed + index)
        target = encode_action(target_action, displayed)
        validation = parse_action(target, displayed)
        if not validation.valid or validation.value != target_action:
            raise AssertionError(f"invalid oracle action for {seed}/{agent_id}: {validation.errors}")
        realized = step(state, {**red, **blue_local}).rewards["BLUE"]
        rows.append(
            _example(
                prompt,
                target,
                {
                    "dataset_version": DATASET_VERSION,
                    "arena_version": ARENA_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "prompt_version": ARENA_PROMPT_VERSION,
                    "seed": seed,
                    "split": split,
                    "phase": "ACT",
                    "agent_id": agent_id,
                    "label_source": "exact_joint_solver_and_prompt_policy_agree",
                    "solver_reward": solution.reward,
                    "policy_reward": realized,
                    "solver_joint_actions_explored": solution.explored,
                    "solver_optimal_count": solution.optimal_count,
                },
            )
        )
    return rows, rejected


def _mechanics_examples(
    start_seed: int, per_kind: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for skill_index, skill in enumerate(("WAIT", "SCAN", "TRANSFER")):
        if per_kind < 3:
            quotas = {"train": per_kind, "validation": 0, "test": 0}
        else:
            holdout = max(1, per_kind // 8)
            quotas = {
                "train": per_kind - 2 * holdout,
                "validation": holdout,
                "test": holdout,
            }
        accepted_by_split: Counter[str] = Counter()
        attempt = 0
        while sum(accepted_by_split.values()) < per_kind:
            if attempt >= max(500, per_kind * 200):
                raise RuntimeError(f"could not generate {per_kind} certified {skill} examples")
            seed = 5_000_000 + start_seed * 10_000 + skill_index * 2_000 + attempt
            attempt += 1
            if seed in FROZEN_EVAL_SEEDS:
                continue
            split = split_for_seed(seed)
            if accepted_by_split[split] >= quotas[split]:
                continue
            state = generate_mechanics_state(seed, skill)
            red = deterministic_policy(state, "RED")
            solution = solve_joint_action(state, "BLUE", red)
            acceptable = solution.acceptable_for("blue-0")
            local = local_policy_action(state, "blue-0")
            if len(acceptable) != 1 or local not in acceptable or local.kind != skill:
                rejected[f"targeted_{skill.lower()}_rejected"] += 1
                continue
            prompt, displayed = action_prompt(state, "blue-0", [], seed)
            target = encode_action(local, displayed)
            parsed = parse_action(target, displayed)
            if not parsed.valid or parsed.value != local:
                raise AssertionError(f"invalid targeted target for {seed}/{skill}")
            rows.append(
                _example(
                    prompt,
                    target,
                    {
                        "dataset_version": DATASET_VERSION,
                        "arena_version": ARENA_VERSION,
                        "generator_version": GENERATOR_VERSION,
                        "prompt_version": ARENA_PROMPT_VERSION,
                        "generator_mode": "targeted_mechanics",
                        "targeted_skill": skill,
                        "seed": seed,
                        "split": split,
                        "phase": "ACT",
                        "agent_id": "blue-0",
                        "label_source": "exact_joint_solver_and_unique_visible_mechanic",
                        "solver_reward": solution.reward,
                        "solver_joint_actions_explored": solution.explored,
                        "solver_optimal_count": solution.optimal_count,
                    },
                )
            )
            accepted_by_split[split] += 1
    return rows, rejected


def generate_dataset(
    start_seed: int,
    seeds: int,
    mechanics_per_kind: int = 0,
    silence_examples: int = 0,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seed_values = range(start_seed, start_seed + seeds)
    if workers == 1:
        generated = map(generate_seed_examples, seed_values)
        for seed_rows, seed_rejected in generated:
            rows.extend(seed_rows)
            rejected.update(seed_rejected)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for seed_rows, seed_rejected in pool.map(generate_seed_examples, seed_values):
                rows.extend(seed_rows)
                rejected.update(seed_rejected)
    if mechanics_per_kind:
        mechanics_rows, mechanics_rejected = _mechanics_examples(start_seed, mechanics_per_kind)
        rows.extend(mechanics_rows)
        rejected.update(mechanics_rejected)
    for index in range(silence_examples):
        seed = 6_000_000 + start_seed * 10_000 + index
        state = generate_mechanics_state(seed, "SILENCE")
        prompt, _ = broadcast_prompt(state, "blue-0", seed)
        target = encode_broadcast(Broadcast((), None, 0))
        parsed = parse_broadcast(target, state, "blue-0")
        if not parsed.valid:
            raise AssertionError(f"invalid silence target for {seed}")
        rows.append(
            _example(
                prompt,
                target,
                {
                    "dataset_version": DATASET_VERSION,
                    "arena_version": ARENA_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "prompt_version": ARENA_PROMPT_VERSION,
                    "generator_mode": "targeted_mechanics",
                    "targeted_skill": "SILENCE",
                    "seed": seed,
                    "split": split_for_seed(seed),
                    "phase": "BROADCAST",
                    "agent_id": "blue-0",
                    "label_source": "exact_empty_broadcast",
                },
            )
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate SFT row IDs")
    by_split = Counter(row["metadata"]["split"] for row in rows)
    by_phase = Counter(row["metadata"]["phase"] for row in rows)
    seed_splits: dict[int, set[str]] = {}
    for row in rows:
        seed_splits.setdefault(row["metadata"]["seed"], set()).add(row["metadata"]["split"])
    if any(len(values) != 1 for values in seed_splits.values()):
        raise AssertionError("seed leaked across splits")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "arena_version": ARENA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "prompt_version": ARENA_PROMPT_VERSION,
        "start_seed": start_seed,
        "num_seeds": seeds,
        "targeted_mechanics_per_kind": mechanics_per_kind,
        "targeted_silence_examples": silence_examples,
        "generation_workers": workers,
        "num_examples": len(rows),
        "examples_by_split": dict(sorted(by_split.items())),
        "examples_by_phase": dict(sorted(by_phase.items())),
        "rejections": dict(sorted(rejected.items())),
        "content_sha256": hashlib.sha256(
            "".join(sorted(row["id"] for row in rows)).encode()
        ).hexdigest(),
    }
    return rows, manifest


def write_dataset(rows: list[dict[str, Any]], manifest: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["metadata"]["split"] == split]
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in subset), encoding="utf-8"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate solver-validated arena SFT examples.")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--mechanics-per-kind", type=int, default=16)
    parser.add_argument("--silence-examples", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/arena_sft_pilot"))
    args = parser.parse_args()
    rows, manifest = generate_dataset(
        args.start_seed,
        args.seeds,
        args.mechanics_per_kind,
        args.silence_examples,
        args.workers,
    )
    write_dataset(rows, manifest, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
