from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .arena import WAIT, Action
from .arena_oracle import deterministic_policy, local_policy_action
from .episode import ArenaEpisodeEnv, EpisodeConfig
from .episode_splits import EPISODE_EVAL_CASES, EPISODE_EVAL_MANIFEST_SHA256

POLICIES = ("wait", "independent", "centralized")


def blue_actions(env: ArenaEpisodeEnv, policy: str) -> dict[str, Action]:
    assert env.state is not None
    members = sorted(agent.id for agent in env.state.agents.values() if agent.team == "BLUE")
    if policy == "wait":
        return {agent_id: WAIT for agent_id in members}
    if policy == "independent":
        return {agent_id: local_policy_action(env.state, agent_id) for agent_id in members}
    if policy == "centralized":
        return deterministic_policy(env.state, "BLUE")
    raise ValueError(f"unknown policy: {policy}")


def run_case(case: tuple[int, int, int, str, str], policy: str) -> dict[str, Any]:
    seed, size, horizon, style_before, style_after = case
    env = ArenaEpisodeEnv(seed, size, EpisodeConfig(horizon=horizon))
    env.reset()
    final = None
    for turn in range(horizon):
        env.broadcast_phase({})
        assert env.state is not None
        style = style_before if turn < horizon // 2 else style_after
        actions = {
            **blue_actions(env, policy),
            **deterministic_policy(env.state, "RED", style),
        }
        final = env.advance(actions)
        if final.terminated or final.truncated:
            break
    assert final is not None and (final.terminated or final.truncated)
    return {
        "seed": seed,
        "size": size,
        "horizon": horizon,
        "style_before": style_before,
        "style_after": style_after,
        "policy": policy,
        "terminal_return": final.rewards["BLUE"],
        "communication_spend": final.info["communication_spend"]["BLUE"],
        "invalid_actions": final.info["invalid_actions"]["BLUE"],
        "invalid_broadcasts": final.info["invalid_broadcasts"]["BLUE"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies = {}
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        returns = [float(row["terminal_return"]) for row in group]
        policies[policy] = {
            "cases": len(group),
            "mean_terminal_return": statistics.mean(returns),
            "stdev_terminal_return": statistics.stdev(returns),
            "positive_return_rate": statistics.mean(value > 0 for value in returns),
            "zero_return_rate": statistics.mean(value == 0 for value in returns),
        }
    paired_headroom = []
    lookup = {
        (row["seed"], row["size"], row["horizon"], row["style_before"], row["style_after"], row["policy"]): row
        for row in rows
    }
    for case in EPISODE_EVAL_CASES:
        central = lookup[(*case, "centralized")]["terminal_return"]
        independent = lookup[(*case, "independent")]["terminal_return"]
        paired_headroom.append(float(central) - float(independent))
    mean = statistics.mean(paired_headroom)
    radius = 1.96 * statistics.stdev(paired_headroom) / len(paired_headroom) ** 0.5
    return {
        "episode_version": "arena-episode-v2",
        "manifest_sha256": EPISODE_EVAL_MANIFEST_SHA256,
        "policies": policies,
        "centralized_minus_independent": {
            "mean_terminal_return_difference": mean,
            "mean_terminal_return_difference_95": [mean - radius, mean + radius],
            "positive_case_rate": statistics.mean(value > 0 for value in paired_headroom),
        },
    }


def run() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [run_case(case, policy) for case in EPISODE_EVAL_CASES for policy in POLICIES]
    return rows, summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-free baselines on the frozen RL-native episodes.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = run()
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
