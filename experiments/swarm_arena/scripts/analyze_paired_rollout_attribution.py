from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


STEP_PATTERN = re.compile(r":step-(\d+):")


def _actions(turn: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {agent_id: action for agent_id, action in turn["actions"]}


def _sequence(turns: list[dict[str, Any]], agent_id: str) -> list[dict[str, Any]]:
    return [_actions(turn)[agent_id] for turn in turns]


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonzero = [row for row in rows if row["effect_nonzero"]]
    positive = [row for row in rows if row["effect_positive"]]
    return {
        "replicas": len(rows),
        "mean_message_effect": _mean(rows, "message_effect"),
        "nonzero_effect_rate": _rate(rows, "effect_nonzero"),
        "positive_effect_rate": _rate(rows, "effect_positive"),
        "trained_receiver_first_action_changed_rate": _rate(
            rows, "trained_receiver_first_action_changed"
        ),
        "receiver_action_sequence_changed_rate": _rate(
            rows, "receiver_action_sequence_changed"
        ),
        "other_blue_action_sequence_changed_rate": _rate(
            rows, "other_blue_action_sequence_changed"
        ),
        "red_action_sequence_changed_rate": _rate(rows, "red_action_sequence_changed"),
        "normal_receiver_target_rate": _rate(rows, "normal_receiver_target"),
        "dropped_receiver_target_rate": _rate(rows, "dropped_receiver_target"),
        "normal_target_dropped_not_rate": _rate(rows, "normal_target_dropped_not"),
        "dropped_target_normal_not_rate": _rate(rows, "dropped_target_normal_not"),
        "nonzero_effect": {
            "count": len(nonzero),
            "trained_receiver_first_action_unchanged_rate": _rate(
                nonzero, "trained_receiver_first_action_unchanged"
            ),
            "trained_receiver_first_action_unchanged_but_other_blue_changed_rate": _rate(
                nonzero,
                "trained_receiver_first_action_unchanged_but_other_blue_changed",
            ),
            "normal_target_dropped_not_rate": _rate(nonzero, "normal_target_dropped_not"),
            "dropped_target_normal_not_rate": _rate(nonzero, "dropped_target_normal_not"),
        },
        "positive_effect": {
            "count": len(positive),
            "trained_receiver_first_action_unchanged_rate": _rate(
                positive, "trained_receiver_first_action_unchanged"
            ),
            "trained_receiver_first_action_unchanged_but_other_blue_changed_rate": _rate(
                positive,
                "trained_receiver_first_action_unchanged_but_other_blue_changed",
            ),
            "normal_target_dropped_not_rate": _rate(positive, "normal_target_dropped_not"),
            "dropped_target_normal_not_rate": _rate(positive, "dropped_target_normal_not"),
        },
    }


def analyze(evidence_path: Path, progress_path: Path) -> dict[str, Any]:
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    scenarios = {
        group["game_id"]: group["scenario"]
        for update in progress
        for group in update["groups"]
    }
    rows: list[dict[str, Any]] = []
    with evidence_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)["payload"]
            group_id = payload["group_id"]
            scenario = scenarios[group_id]
            match = STEP_PATTERN.search(group_id)
            if match is None:
                raise ValueError(f"missing step in group id: {group_id}")
            step = int(match.group(1))
            receiver = payload["focused_agent"]
            target = scenario["active_target"]
            for replica in payload["replicas"]:
                normal_turns = replica["replay"]["turns"]
                dropped_turns = replica["dropped_replay"]["turns"]
                normal_first = _actions(normal_turns[0])
                dropped_first = _actions(dropped_turns[0])
                normal_receiver = normal_first[receiver]
                dropped_receiver = dropped_first[receiver]
                receiver_first_changed = normal_receiver != dropped_receiver
                receiver_sequence_changed = _sequence(
                    normal_turns, receiver
                ) != _sequence(dropped_turns, receiver)
                other_blue_changed = any(
                    _sequence(normal_turns, agent_id)
                    != _sequence(dropped_turns, agent_id)
                    for agent_id in normal_first
                    if agent_id.startswith("blue-") and agent_id != receiver
                )
                red_changed = any(
                    _sequence(normal_turns, agent_id)
                    != _sequence(dropped_turns, agent_id)
                    for agent_id in normal_first
                    if agent_id.startswith("red-")
                )
                effect = float(replica["replay"]["terminal_return"]) - float(
                    replica["dropped_replay"]["terminal_return"]
                )
                normal_target = normal_receiver.get("target") == target
                dropped_target = dropped_receiver.get("target") == target
                rows.append(
                    {
                        "step": step,
                        "window": f"{step // 10 * 10}-{step // 10 * 10 + 9}",
                        "pair_index": int(scenario["pair_index"]),
                        "world": scenario["world"],
                        "receiver": receiver,
                        "message_effect": effect,
                        "effect_nonzero": abs(effect) > 1e-12,
                        "effect_positive": effect > 1e-12,
                        "trained_receiver_first_action_changed": receiver_first_changed,
                        "trained_receiver_first_action_unchanged": not receiver_first_changed,
                        "receiver_action_sequence_changed": receiver_sequence_changed,
                        "other_blue_action_sequence_changed": other_blue_changed,
                        "red_action_sequence_changed": red_changed,
                        "trained_receiver_first_action_unchanged_but_other_blue_changed": (
                            not receiver_first_changed and other_blue_changed
                        ),
                        "normal_receiver_target": normal_target,
                        "dropped_receiver_target": dropped_target,
                        "normal_target_dropped_not": normal_target and not dropped_target,
                        "dropped_target_normal_not": dropped_target and not normal_target,
                    }
                )

    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_window_pair_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[row["window"]].append(row)
        pair_world = f"pair-{row['pair_index']}/{row['world']}"
        by_pair_world[pair_world].append(row)
        by_window_pair_world[f"{row['window']}/{pair_world}"].append(row)
    return {
        "version": "paired-rollout-attribution-diagnosis-v1",
        "evidence_path": str(evidence_path),
        "progress_path": str(progress_path),
        "overall": _summarize(rows),
        "by_window": {key: _summarize(value) for key, value in sorted(by_window.items())},
        "by_pair_world": {
            key: _summarize(value) for key, value in sorted(by_pair_world.items())
        },
        "by_window_pair_world": {
            key: _summarize(value)
            for key, value in sorted(by_window_pair_world.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose causal attribution in paired normal/drop rollout evidence."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.evidence, args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
