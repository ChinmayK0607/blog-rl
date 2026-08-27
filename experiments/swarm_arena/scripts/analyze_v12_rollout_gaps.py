#!/usr/bin/env python3
"""Summarize compact V12 rollout/evaluation evidence without loading raw traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

EPSILON = 1e-12


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.fmean(rows) if rows else None


def rate(values: Iterable[bool]) -> float | None:
    rows = list(values)
    return statistics.fmean(float(value) for value in rows) if rows else None


def kind(group: dict[str, Any]) -> str:
    scenario = group["scenario"]
    return str(scenario.get("kind", "ordinary" if scenario.get("source") == "ordinary" else "unknown"))


def action_class(group: dict[str, Any], replica: dict[str, Any]) -> str:
    action = replica.get("focused_action") or {}
    target = action.get("target")
    active = group["scenario"].get("active_target")
    candidates = group["scenario"].get("candidate_targets") or []
    alternate = next((value for value in candidates if value != active), None)
    if target == active:
        return "active_target"
    if alternate is not None and target == alternate:
        return "alternate_target"
    if target is None:
        return "no_target"
    return "other_target"


def effect(group_kind: str, replica: dict[str, Any]) -> float | None:
    key = "challenge_effect" if group_kind == "decoy" else "semantic_effect"
    value = replica.get(key)
    return float(value) if value is not None else None


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    replicas = [
        (group, replica)
        for group in groups
        for replica in group["replicas"]
    ]
    effects = [
        value
        for group, replica in replicas
        if (value := effect(kind(group), replica)) is not None
    ]
    action_classes = Counter(action_class(group, replica) for group, replica in replicas)
    action_types = Counter(
        (replica.get("focused_action") or {}).get("type", "NONE")
        for _, replica in replicas
    )
    focused_advantages = []
    for group, replica in replicas:
        focused = group["scenario"].get("focused_agent")
        if focused is not None:
            focused_advantages.append(float(replica["advantages"].get(focused, 0.0)))
    group_effects = []
    uniform_effect_groups = 0
    for group in groups:
        values = [
            value
            for replica in group["replicas"]
            if (value := effect(kind(group), replica)) is not None
        ]
        if values:
            group_effects.append(statistics.fmean(values))
            if max(values) - min(values) <= EPSILON:
                uniform_effect_groups += 1
    target_effects: dict[str, list[float]] = defaultdict(list)
    for group, replica in replicas:
        value = effect(kind(group), replica)
        if value is not None:
            target_effects[action_class(group, replica)].append(value)
    return {
        "groups": len(groups),
        "replicas": len(replicas),
        "mean_terminal_return": mean(
            float(replica["return"]) for _, replica in replicas
        ),
        "effect": {
            "mean": mean(effects),
            "nonzero_rate": rate(abs(value) > EPSILON for value in effects),
            "positive_rate": rate(value > EPSILON for value in effects),
            "negative_rate": rate(value < -EPSILON for value in effects),
            "positive_group_rate": rate(value > EPSILON for value in group_effects),
            "uniform_replica_group_rate": (
                uniform_effect_groups / len(group_effects) if group_effects else None
            ),
        },
        "focused_advantage": {
            "mean_absolute": mean(abs(value) for value in focused_advantages),
            "nonzero_rate": rate(abs(value) > EPSILON for value in focused_advantages),
        },
        "focused_action_class_counts": dict(sorted(action_classes.items())),
        "focused_action_class_rates": {
            key: value / len(replicas) for key, value in sorted(action_classes.items())
        },
        "focused_action_type_counts": dict(sorted(action_types.items())),
        "effect_by_action_class": {
            key: {
                "replicas": len(values),
                "mean": mean(values),
                "positive_rate": rate(value > EPSILON for value in values),
                "negative_rate": rate(value < -EPSILON for value in values),
            }
            for key, values in sorted(target_effects.items())
        },
    }


def group_breakdown(
    updates: list[dict[str, Any]],
    key_fn: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        for group in update["groups"]:
            grouped[str(key_fn(update, group))].append(group)
    return {key: summarize_groups(value) for key, value in sorted(grouped.items())}


def training_report(updates: list[dict[str, Any]]) -> dict[str, Any]:
    if not updates:
        raise ValueError("progress contains no durable updates")
    steps = [int(update["step"]) for update in updates]
    if steps != list(range(len(updates))):
        raise ValueError("progress steps must be contiguous from zero")
    all_groups = [group for update in updates for group in update["groups"]]
    return {
        "durable_updates": len(updates),
        "latest_step": steps[-1],
        "policy_revision": updates[-1]["policy_revision"],
        "overall": summarize_groups(all_groups),
        "by_kind": group_breakdown(updates, lambda _update, group: kind(group)),
        "by_20_update_window": group_breakdown(
            updates,
            lambda update, group: f"{int(update['step']) // 20 * 20:03d}-"
            f"{int(update['step']) // 20 * 20 + 19:03d}/{kind(group)}",
        ),
        "by_stage": group_breakdown(
            updates,
            lambda _update, group: f"{group['scenario'].get('curriculum_stage')}/{kind(group)}",
        ),
        "by_receiver": group_breakdown(
            updates,
            lambda _update, group: f"{group['scenario'].get('focused_agent')}/{kind(group)}",
        ),
        "by_20_update_window_receiver": group_breakdown(
            updates,
            lambda update, group: f"{int(update['step']) // 20 * 20:03d}-"
            f"{int(update['step']) // 20 * 20 + 19:03d}/"
            f"{group['scenario'].get('focused_agent')}/{kind(group)}",
        ),
        "by_opponent_family": group_breakdown(
            updates,
            lambda _update, group: f"{group['scenario'].get('opponent', {}).get('family')}/{kind(group)}",
        ),
        "by_horizon": group_breakdown(
            updates,
            lambda _update, group: f"{group['scenario'].get('scheduled_horizon')}/{kind(group)}",
        ),
        "by_world": group_breakdown(
            updates,
            lambda _update, group: f"{group['scenario'].get('world')}/{kind(group)}",
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            "/".join(
                (
                    str(row["policy_variant"]),
                    str(row["suite"]),
                    str(row["condition"]),
                )
            )
        ].append(row)
    return {
        "rows": len(rows),
        "cells": {
            key: {
                "rows": len(value),
                "mean_terminal_return": mean(float(row["terminal_return"]) for row in value),
                "mean_duplicate_target_turn_rate": mean(
                    float(row["duplicate_target_turn_rate"]) for row in value
                ),
                "mean_communication_spend": mean(
                    float(row["communication_spend"]) for row in value
                ),
                "mean_nonempty_messages": mean(
                    float(row["messages_nonempty"]) for row in value
                ),
                "critical_capture_rate": rate(bool(row["critical_capture"]) for row in value),
                "action_validity": mean(float(row["action_protocol_rate"]) for row in value),
                "broadcast_validity": mean(
                    float(row["broadcast_protocol_rate"])
                    for row in value
                    if row["broadcast_protocol_rate"] is not None
                ),
                "grounding": mean(
                    float(row["broadcast_grounded_rate"])
                    for row in value
                    if row["broadcast_grounded_rate"] is not None
                ),
            }
            for key, value in sorted(grouped.items())
        },
    }


def parse_eval_arg(value: str) -> tuple[int, Path]:
    step, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("evaluation must be STEP=ROWS_JSONL")
    return int(step), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--evaluation", action="append", default=[], type=parse_eval_arg)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    updates = json.loads(args.progress.read_text(encoding="utf-8"))
    evaluation = {
        str(step): {
            **summarize_eval_rows(read_jsonl(path)),
            "rows_sha256": file_sha256(path),
        }
        for step, path in args.evaluation
    }
    report = {
        "version": "v12-compact-rollout-gap-audit-v1",
        "scope": (
            "interim training-distribution and compact development evidence; "
            "not a held-out result or checkpoint selection"
        ),
        "progress_sha256": file_sha256(args.progress),
        "training": training_report(updates),
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
