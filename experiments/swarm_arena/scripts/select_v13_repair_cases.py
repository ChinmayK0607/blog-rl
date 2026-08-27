#!/usr/bin/env python3
"""Select compact V13 repair bands from training-only V12 rollout records."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

POLICIES = ("blue-0", "blue-1", "blue-2", "blue-3")
EPSILON = 1e-12


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_class(group: dict[str, Any], replica: dict[str, Any]) -> str:
    action = replica.get("focused_action") or {}
    target = action.get("target")
    scenario = group["scenario"]
    active = scenario.get("active_target")
    alternate = next(
        (candidate for candidate in scenario.get("candidate_targets", []) if candidate != active),
        None,
    )
    if target == active:
        return "active_target"
    if alternate is not None and target == alternate:
        return "alternate_target"
    return "other"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _rate(values: list[bool]) -> float:
    return statistics.fmean(float(value) for value in values) if values else 0.0


def _handoff_rows(
    updates: list[dict[str, Any]],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        for group in update["groups"]:
            scenario = group["scenario"]
            if scenario.get("kind") != kind:
                continue
            key = (
                int(scenario["pair_index"]),
                str(scenario["world"]),
                str(scenario["receiver"]),
            )
            grouped[key].append(group)

    rows = []
    for (pair_index, world, receiver), groups in grouped.items():
        replicas = [replica for group in groups for replica in group["replicas"]]
        classes = [action_class(group, replica) for group in groups for replica in group["replicas"]]
        effect_key = "challenge_effect" if kind == "decoy" else "semantic_effect"
        effects = [
            float(replica[effect_key])
            for replica in replicas
            if replica.get(effect_key) is not None
        ]
        alternate_rate = _rate([value == "alternate_target" for value in classes])
        active_rate = _rate([value == "active_target" for value in classes])
        negative_effect_rate = _rate([value < -EPSILON for value in effects])
        if kind == "decoy":
            priority = alternate_rate + _mean([abs(min(value, 0.0)) for value in effects])
        else:
            priority = (1.0 - active_rate) + _mean([max(value, 0.0) for value in effects]) * 0.05
        rows.append(
            {
                "pair_index": pair_index,
                "world": world,
                "receiver": receiver,
                "groups": len(groups),
                "replicas": len(replicas),
                "active_target_rate": active_rate,
                "alternate_target_rate": alternate_rate,
                "mean_effect": _mean(effects),
                "negative_effect_rate": negative_effect_rate,
                "priority": priority,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["priority"], row["receiver"], row["pair_index"], row["world"]),
    )


def _ordinary_rows(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for update in updates:
        for group in update["groups"]:
            scenario = group["scenario"]
            if scenario.get("source") != "ordinary":
                continue
            receiver = str(scenario["focused_agent"])
            replicas = group["replicas"]
            advantages = [float(replica["advantages"].get(receiver, 0.0)) for replica in replicas]
            actions = {
                (
                    (replica.get("focused_action") or {}).get("type"),
                    (replica.get("focused_action") or {}).get("target"),
                )
                for replica in replicas
            }
            nonzero_rate = _rate([abs(value) > EPSILON for value in advantages])
            rows.append(
                {
                    "seed": int(scenario["seed"]),
                    "receiver": receiver,
                    "stage": scenario.get("curriculum_stage"),
                    "opponent_family": scenario.get("opponent", {}).get("family"),
                    "nonzero_advantage_rate": nonzero_rate,
                    "mean_absolute_advantage": _mean([abs(value) for value in advantages]),
                    "distinct_actions": len(actions),
                    "priority": nonzero_rate + min(len(actions), 4) * 0.05,
                }
            )
    return sorted(rows, key=lambda row: (-row["priority"], row["receiver"], row["seed"]))


def _take_per_receiver(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = []
    for policy in POLICIES:
        policy_rows = [row for row in rows if row["receiver"] == policy]
        if len(policy_rows) < count:
            raise ValueError(f"only {len(policy_rows)} eligible cases for {policy}; need {count}")
        selected.extend(policy_rows[:count])
    return sorted(selected, key=lambda row: (row["receiver"], -row["priority"], row.get("seed", -1)))


def select_repair_cases(
    updates: list[dict[str, Any]],
    *,
    window_updates: int,
    challenge_per_receiver: int,
    critical_per_receiver: int,
    ordinary_per_receiver: int,
) -> dict[str, Any]:
    if not updates:
        raise ValueError("progress contains no updates")
    steps = [int(update["step"]) for update in updates]
    if steps != list(range(len(updates))):
        raise ValueError("progress steps must be contiguous from zero")
    if window_updates < 1 or window_updates > len(updates):
        raise ValueError("window_updates must fit inside durable progress")
    window = updates[-window_updates:]
    challenges = _handoff_rows(window, kind="decoy")
    critical = _handoff_rows(window, kind="critical")
    ordinary = _ordinary_rows(window)
    return {
        "version": "arena-rl-v13-training-only-repair-selection-v1",
        "scope": "training-only V12 rollout band; never a held-out or selection result",
        "source": {
            "durable_updates": len(updates),
            "window_first_step": int(window[0]["step"]),
            "window_last_step": int(window[-1]["step"]),
            "window_updates": window_updates,
        },
        "selected": {
            "challenge_repair": _take_per_receiver(challenges, challenge_per_receiver),
            "critical_rehearsal": _take_per_receiver(critical, critical_per_receiver),
            "ordinary_replay_anchors": _take_per_receiver(ordinary, ordinary_per_receiver),
        },
        "diagnostics": {
            "challenge_candidates": len(challenges),
            "critical_candidates": len(critical),
            "ordinary_candidates": len(ordinary),
        },
        "admission": {
            "status": "interim_only",
            "reason": "regenerate from the completed V12 progress record before freezing V13",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-updates", type=int, default=40)
    parser.add_argument("--challenge-per-receiver", type=int, default=6)
    parser.add_argument("--critical-per-receiver", type=int, default=4)
    parser.add_argument("--ordinary-per-receiver", type=int, default=8)
    args = parser.parse_args()
    updates = json.loads(args.progress.read_text(encoding="utf-8"))
    result = select_repair_cases(
        updates,
        window_updates=args.window_updates,
        challenge_per_receiver=args.challenge_per_receiver,
        critical_per_receiver=args.critical_per_receiver,
        ordinary_per_receiver=args.ordinary_per_receiver,
    )
    result["source"]["progress_sha256"] = file_sha256(args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
