from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from .arena import WAIT, legal_actions, step
from .arena_generation import generate_state
from .arena_oracle import deterministic_policy, local_policy_action, solve_joint_action
from .arena_splits import FROZEN_EVAL_CASES, FROZEN_EVAL_MANIFEST_SHA256


def run(random_repeats: int = 16) -> dict[str, Any]:
    if random_repeats < 1:
        raise ValueError("random_repeats must be positive")
    per_policy: dict[str, list[dict[str, float | bool]]] = {
        name: [] for name in ("wait", "random_legal", "independent_local", "sequential_heuristic", "centralized_oracle")
    }
    for seed, size, style in FROZEN_EVAL_CASES:
        state = generate_state(seed, size)
        red = deterministic_policy(state, "RED", style)
        solution = solve_joint_action(state, "BLUE", red)
        blue_agents = sorted(agent.id for agent in state.agents.values() if agent.team == "BLUE")
        policies = {
            "wait": {agent_id: WAIT for agent_id in blue_agents},
            "independent_local": {
                agent_id: local_policy_action(state, agent_id) for agent_id in blue_agents
            },
            "sequential_heuristic": deterministic_policy(state, "BLUE"),
            "centralized_oracle": dict(solution.canonical_assignment),
        }
        for name, policy in policies.items():
            reward = step(state, {**red, **policy}).rewards["BLUE"]
            per_policy[name].append(
                {
                    "reward": reward,
                    "regret": solution.reward - reward,
                    "optimal": abs(solution.reward - reward) <= 1e-9,
                }
            )
        random_rewards = []
        for repeat in range(random_repeats):
            rng = random.Random(seed * 10_000 + repeat)
            policy = {
                agent_id: rng.choice(legal_actions(state, agent_id))
                for agent_id in blue_agents
            }
            random_rewards.append(step(state, {**red, **policy}).rewards["BLUE"])
        random_reward = statistics.mean(random_rewards)
        per_policy["random_legal"].append(
            {
                "reward": random_reward,
                "regret": solution.reward - random_reward,
                "optimal": False,
            }
        )

    return {
        "manifest_sha256": FROZEN_EVAL_MANIFEST_SHA256,
        "num_cases": len(FROZEN_EVAL_CASES),
        "random_repeats_per_case": random_repeats,
        "policies": {
            name: {
                "mean_reward": statistics.mean(float(row["reward"]) for row in rows),
                "mean_oracle_regret": statistics.mean(float(row["regret"]) for row in rows),
                "optimal_outcome_rate": statistics.mean(bool(row["optimal"]) for row in rows),
            }
            for name, rows in per_policy.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-learning policies on the frozen arena evaluation.")
    parser.add_argument("--random-repeats", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/arena_v2/baselines.json"))
    args = parser.parse_args()
    summary = run(args.random_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
