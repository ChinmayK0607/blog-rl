from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .arena import ARENA_VERSION, WAIT, Action, GameState, redundant_agents, step
from .arena_generation import GENERATOR_VERSION, generate_state
from .arena_oracle import deterministic_policy, solve_joint_action
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
from .arena_sft import oracle_broadcast
from .arena_splits import FROZEN_EVAL_CASES, FROZEN_EVAL_MANIFEST_SHA256
from .providers import OpenAICompatibleProvider

EVAL_VERSION = "arena-eval-v2"
CONDITIONS = ("generated", "dropped", "reference", "shuffled")


class ArenaModel(Protocol):
    name: str

    def respond(self, messages: list[dict[str, str]], oracle_target: str) -> str: ...


@dataclass
class OracleArenaModel:
    name: str = "oracle"

    def respond(self, messages: list[dict[str, str]], oracle_target: str) -> str:
        del messages
        return oracle_target


@dataclass
class OpenAIArenaModel:
    provider: OpenAICompatibleProvider
    name: str

    def respond(self, messages: list[dict[str, str]], oracle_target: str) -> str:
        del oracle_target
        return self.provider.generate(None, messages).text  # type: ignore[arg-type]


def _reference_broadcasts(state: GameState, reference: dict[str, Action]) -> dict[str, Broadcast]:
    return {agent_id: oracle_broadcast(state, agent_id, reference[agent_id]) for agent_id in sorted(reference)}


def _inbox(broadcasts: dict[str, Broadcast], receiver: str) -> list[dict[str, Any]]:
    return [
        {"sender": sender, "broadcast": broadcasts[sender].to_dict()}
        for sender in sorted(broadcasts)
        if sender != receiver
    ]


def evaluate_case(
    model: ArenaModel,
    seed: int,
    size: int,
    opponent_style: str,
    shuffled_broadcasts: dict[str, Broadcast],
) -> dict[str, Any]:
    state = generate_state(seed, size)
    opponent_actions = deterministic_policy(state, "RED", opponent_style)
    solution = solve_joint_action(state, "BLUE", opponent_actions)
    reference = dict(solution.canonical_assignment)
    reference_broadcasts = _reference_broadcasts(state, reference)

    generated: dict[str, Broadcast] = {}
    message_rows = []
    for index, agent_id in enumerate(sorted(reference)):
        prompt, _ = broadcast_prompt(state, agent_id, seed + index)
        target = encode_broadcast(reference_broadcasts[agent_id])
        raw = model.respond(prompt, target)
        parsed = parse_broadcast(raw, state, agent_id)
        value = parsed.value if parsed.valid else Broadcast((), None, 0)
        assert isinstance(value, Broadcast)
        generated[agent_id] = value
        message_rows.append(
            {
                "agent_id": agent_id,
                "valid": parsed.valid,
                "errors": list(parsed.errors),
                "raw_response": raw,
            }
        )

    condition_broadcasts = {
        "generated": generated,
        "dropped": {agent_id: Broadcast((), None, 0) for agent_id in reference},
        "reference": reference_broadcasts,
        "shuffled": shuffled_broadcasts,
    }
    condition_rows = []
    semantic_by_agent: dict[str, list[Action | None]] = defaultdict(list)
    for condition in CONDITIONS:
        broadcasts = condition_broadcasts[condition]
        permutations = (0, 1, 2) if condition == "generated" else (0,)
        for permutation in permutations:
            selected: dict[str, Action] = {}
            strict_valid = 0
            action_rows = []
            for index, agent_id in enumerate(sorted(reference)):
                prompt, displayed = action_prompt(
                    state, agent_id, _inbox(broadcasts, agent_id), seed + index + permutation
                )
                target = encode_action(reference[agent_id], displayed)
                raw = model.respond(prompt, target)
                parsed = parse_action(raw, displayed)
                action = parsed.value if parsed.valid else WAIT
                assert isinstance(action, Action)
                selected[agent_id] = action
                strict_valid += int(parsed.valid)
                if condition == "generated":
                    semantic_by_agent[agent_id].append(action if parsed.valid else None)
                action_rows.append(
                    {
                        "agent_id": agent_id,
                        "valid": parsed.valid,
                        "errors": list(parsed.errors),
                        "selected_action": action.to_dict(),
                        "raw_response": raw,
                    }
                )
            outcome = step(state, {**opponent_actions, **selected})
            redundant = redundant_agents(state, {**opponent_actions, **selected}, "BLUE")
            environment_reward = outcome.rewards["BLUE"]
            strict_team = strict_valid == 4
            regret = solution.reward - environment_reward
            condition_rows.append(
                {
                    "condition": condition,
                    "permutation": permutation,
                    "strict_action_rate": strict_valid / 4,
                    "strict_team_protocol": strict_team,
                    "environment_reward": environment_reward,
                    "oracle_reward": solution.reward,
                    "regret": regret,
                    "optimal_outcome": strict_team and regret <= 1e-9,
                    "duplicate_targets": list(outcome.duplicate_targets["BLUE"]),
                    "redundant_agents": list(redundant),
                    "invalid_environment_actions": list(outcome.invalid_agents),
                    "actions": action_rows,
                }
            )

    order_consistent = all(
        len(actions) == 3 and actions[0] is not None and actions.count(actions[0]) == 3
        for actions in semantic_by_agent.values()
    )
    return {
        "eval_version": EVAL_VERSION,
        "arena_version": ARENA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "prompt_version": ARENA_PROMPT_VERSION,
        "manifest_sha256": FROZEN_EVAL_MANIFEST_SHA256,
        "model": model.name,
        "seed": seed,
        "size": size,
        "opponent_style": opponent_style,
        "solver_joint_actions_explored": solution.explored,
        "solver_optimal_count": solution.optimal_count,
        "message_strict_rate": sum(row["valid"] for row in message_rows) / 4,
        "action_order_consistent": order_consistent,
        "messages": message_rows,
        "conditions": condition_rows,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_ci(values: list[float]) -> list[float]:
        mean = statistics.mean(values)
        if len(values) < 2:
            return [mean, mean]
        radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        return [mean - radius, mean + radius]

    def wilson(values: list[bool]) -> list[float]:
        n = len(values)
        p = sum(values) / n
        z = 1.96
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return [center - radius, center + radius]

    main = [condition for row in rows for condition in row["conditions"] if condition["permutation"] == 0]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in main:
        by_condition[row["condition"]].append(row)
    conditions = {}
    for name, group in sorted(by_condition.items()):
        rewards = [float(item["environment_reward"]) for item in group]
        regrets = [float(item["regret"]) for item in group]
        optimal = [bool(item["optimal_outcome"]) for item in group]
        conditions[name] = {
            "strict_action_rate": statistics.mean(item["strict_action_rate"] for item in group),
            "optimal_outcome_rate": statistics.mean(optimal),
            "optimal_outcome_wilson_95": wilson(optimal),
            "mean_environment_reward": statistics.mean(rewards),
            "mean_environment_reward_95": mean_ci(rewards),
            "mean_oracle_regret": statistics.mean(regrets),
            "mean_oracle_regret_95": mean_ci(regrets),
            "nonredundant_joint_action_rate": statistics.mean(not item["redundant_agents"] for item in group),
        }
    generated = conditions["generated"]
    generated_lookup = {
        (row["seed"], row["size"], row["opponent_style"]): next(
            item for item in row["conditions"] if item["condition"] == "generated" and item["permutation"] == 0
        )
        for row in rows
    }
    slices = {}
    for label, keys in {
        "topology_size": sorted({str(row["size"]) for row in rows}),
        "opponent_style": sorted({row["opponent_style"] for row in rows}),
    }.items():
        slices[label] = {}
        for key in keys:
            group = (
                [
                    generated_lookup[(row["seed"], row["size"], row["opponent_style"])]
                    for row in rows
                    if str(row["size"]) == key
                    if label == "topology_size"
                ]
                if label == "topology_size"
                else [
                    generated_lookup[(row["seed"], row["size"], row["opponent_style"])]
                    for row in rows
                    if row["opponent_style"] == key
                ]
            )
            slices[label][key] = {
                "cases": len(group),
                "optimal_outcome_rate": statistics.mean(item["optimal_outcome"] for item in group),
                "mean_oracle_regret": statistics.mean(item["regret"] for item in group),
            }
    return {
        "eval_version": EVAL_VERSION,
        "arena_version": ARENA_VERSION,
        "manifest_sha256": FROZEN_EVAL_MANIFEST_SHA256,
        "model": rows[0]["model"],
        "num_cases": len(rows),
        "message_strict_rate": statistics.mean(row["message_strict_rate"] for row in rows),
        "action_order_consistency_rate": statistics.mean(row["action_order_consistent"] for row in rows),
        "conditions": conditions,
        "generated_slices": slices,
        "generated_minus_dropped_reward": generated["mean_environment_reward"]
        - conditions["dropped"]["mean_environment_reward"],
        "reference_message_headroom": conditions["reference"]["mean_environment_reward"]
        - generated["mean_environment_reward"],
    }


def _evaluate_frozen_case(model: ArenaModel, index: int) -> dict[str, Any]:
    seed, size, style = FROZEN_EVAL_CASES[index]
    shuffled_seed, shuffled_size, _ = FROZEN_EVAL_CASES[(index + 1) % len(FROZEN_EVAL_CASES)]
    shuffled_state = generate_state(shuffled_seed, shuffled_size)
    shuffled_reference = dict(
        solve_joint_action(
            shuffled_state,
            "BLUE",
            deterministic_policy(shuffled_state, "RED", style),
        ).canonical_assignment
    )
    shuffled = _reference_broadcasts(shuffled_state, shuffled_reference)
    # Preserve receiver identities while substituting semantically unrelated
    # broadcasts from the next frozen case.
    remapped = {
        agent_id: shuffled[source]
        for agent_id, source in zip(
            sorted(agent.id for agent in generate_state(seed, size).agents.values() if agent.team == "BLUE"),
            sorted(shuffled),
            strict=True,
        )
    }
    return evaluate_case(model, seed, size, style, remapped)


def run(model: ArenaModel, output_dir: Path, workers: int = 1) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        rows = [_evaluate_frozen_case(model, index) for index in range(len(FROZEN_EVAL_CASES))]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(lambda index: _evaluate_frozen_case(model, index), range(len(FROZEN_EVAL_CASES))))
    summary = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 4v4 arena evaluation.")
    parser.add_argument("--provider", choices=("oracle", "openai", "local-hf"), default="oracle")
    parser.add_argument("--model", default="oracle")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/arena_oracle"))
    args = parser.parse_args()
    if args.provider == "oracle":
        model: ArenaModel = OracleArenaModel()
    elif args.provider == "openai":
        provider = OpenAICompatibleProvider(
            args.base_url, args.model, api_key=args.api_key, temperature=0.0, max_tokens=192
        )
        model = OpenAIArenaModel(provider, args.model)
    else:
        from .local_hf import LocalHFArenaModel

        model = LocalHFArenaModel(args.model, args.adapter)
    if args.provider == "local-hf" and args.workers != 1:
        parser.error("local-hf requires --workers 1; use the OpenAI provider for concurrent serving")
    print(json.dumps(run(model, args.output_dir, args.workers), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
