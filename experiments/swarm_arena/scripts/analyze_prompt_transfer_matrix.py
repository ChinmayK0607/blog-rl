from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def _bootstrap_mean(
    values: list[float], *, trials: int = 20_000, seed: int = 0
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    samples = sorted(
        statistics.fmean(values[generator.randrange(len(values))] for _ in values)
        for _ in range(trials)
    )
    return [samples[int(0.025 * (trials - 1))], samples[int(0.975 * (trials - 1))]]


def _mean_ci(values: list[float], *, seed: int) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "bootstrap_95": _bootstrap_mean(values, seed=seed),
        "positive_rate": statistics.fmean(value > 0 for value in values),
    }


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1:
        raise ValueError("Wilson interval requires observations")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return [center - margin / denominator, center + margin / denominator]


def _binary_summary(values: list[bool]) -> dict[str, Any]:
    successes = sum(values)
    return {
        "successes": successes,
        "total": len(values),
        "rate": successes / len(values),
        "wilson_95": _wilson(successes, len(values)),
    }


def _mcnemar_two_sided(normal_only: int, dropped_only: int) -> float:
    discordant = normal_only + dropped_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(normal_only, dropped_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["pair_index"]), str(row["world"]), int(row["repeat"])


def _condition_map(
    rows: list[dict[str, Any]], *, kind: str
) -> dict[tuple[int, str, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] != kind:
            continue
        grouped.setdefault(_key(row), {})[str(row["condition"])] = row
    required = {"normal", "dropped", "sender_shuffled"}
    incomplete = [key for key, conditions in grouped.items() if set(conditions) != required]
    if incomplete:
        raise ValueError(f"incomplete intervention cells: {incomplete[:3]}")
    return grouped


def _return_effects(
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]],
    right: str,
) -> dict[tuple[int, str, int], float]:
    return {
        key: float(conditions["normal"]["terminal_return"])
        - float(conditions[right]["terminal_return"])
        for key, conditions in grouped.items()
    }


def _scenario_means(
    effects: dict[tuple[int, str, int], float]
) -> list[float]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for (pair_index, world, _), value in effects.items():
        grouped.setdefault((pair_index, world), []).append(value)
    return [statistics.fmean(grouped[key]) for key in sorted(grouped)]


def _effect_summary(
    effects: dict[tuple[int, str, int], float], *, seed: int
) -> dict[str, Any]:
    sampling = list(effects.values())
    scenarios = _scenario_means(effects)
    return {
        "sampling_units": _mean_ci(sampling, seed=seed),
        "pair_world_units": _mean_ci(scenarios, seed=seed + 1),
    }


def _target_summary(
    critical: dict[tuple[int, str, int], dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    normal = {
        key: bool(conditions["normal"]["receiver_target_action"])
        for key, conditions in critical.items()
    }
    dropped = {
        key: bool(conditions["dropped"]["receiver_target_action"])
        for key, conditions in critical.items()
    }
    normal_only = sum(normal[key] and not dropped[key] for key in normal)
    dropped_only = sum(dropped[key] and not normal[key] for key in normal)
    return {
        "normal": _binary_summary(list(normal.values())),
        "dropped": _binary_summary(list(dropped.values())),
        "normal_correct_dropped_wrong": normal_only,
        "normal_wrong_dropped_correct": dropped_only,
        "paired_accuracy_delta": statistics.fmean(
            float(normal[key]) - float(dropped[key]) for key in normal
        ),
        "mcnemar_exact_two_sided": _mcnemar_two_sided(normal_only, dropped_only),
    }


def _summarize_cell(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    critical = _condition_map(rows, kind="critical")
    decoy = _condition_map(rows, kind="decoy")
    critical_drop = _return_effects(critical, "dropped")
    critical_shuffle = _return_effects(critical, "sender_shuffled")
    decoy_drop = _return_effects(decoy, "dropped")
    if set(critical_drop) != set(decoy_drop):
        raise ValueError("critical and decoy keys do not match")
    specificity = {
        key: critical_drop[key] - decoy_drop[key] for key in critical_drop
    }
    by_pair_world: dict[str, Any] = {}
    for pair_index, world in sorted({(key[0], key[1]) for key in critical}):
        selected = {
            key: conditions
            for key, conditions in critical.items()
            if key[:2] == (pair_index, world)
        }
        by_pair_world[f"pair-{pair_index}/{world}"] = {
            "normal_minus_dropped_return": _effect_summary(
                {key: critical_drop[key] for key in selected}, seed=seed + pair_index
            ),
            "target_selection": _target_summary(selected),
        }
    protocol_fields = ("broadcast_valid", "broadcast_grounded", "action_valid")
    return {
        "rows": len(rows),
        "normal_minus_dropped_return": _effect_summary(critical_drop, seed=seed),
        "normal_minus_shuffled_return": _effect_summary(
            critical_shuffle, seed=seed + 10
        ),
        "critical_minus_decoy_drop_specificity": _effect_summary(
            specificity, seed=seed + 20
        ),
        "target_selection": _target_summary(critical),
        "normal_sender_target_fact": _binary_summary(
            [
                bool(conditions["normal"]["sender_target_fact"])
                for conditions in critical.values()
            ]
        ),
        "protocol": {
            field: _binary_summary([bool(row[field]) for row in rows])
            for field in protocol_fields
        },
        "by_pair_world": by_pair_world,
    }


def _parse_binding(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def _load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (path / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return manifest, rows


def _comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    left_critical = _condition_map(left, kind="critical")
    right_critical = _condition_map(right, kind="critical")
    left_decoy = _condition_map(left, kind="decoy")
    right_decoy = _condition_map(right, kind="decoy")
    if not (
        set(left_critical) == set(right_critical) == set(left_decoy) == set(right_decoy)
    ):
        raise ValueError("comparison cells do not share identical paired keys")
    left_drop = _return_effects(left_critical, "dropped")
    right_drop = _return_effects(right_critical, "dropped")
    left_decoy_drop = _return_effects(left_decoy, "dropped")
    right_decoy_drop = _return_effects(right_decoy, "dropped")
    keys = set(left_drop)
    return {
        "difference_in_normal_minus_dropped": _effect_summary(
            {key: left_drop[key] - right_drop[key] for key in keys}, seed=seed
        ),
        "difference_in_specificity": _effect_summary(
            {
                key: (left_drop[key] - left_decoy_drop[key])
                - (right_drop[key] - right_decoy_drop[key])
                for key in keys
            },
            seed=seed + 1,
        ),
        "normal_target_accuracy_difference": _mean_ci(
            [
                float(left_critical[key]["normal"]["receiver_target_action"])
                - float(right_critical[key]["normal"]["receiver_target_action"])
                for key in sorted(keys)
            ],
            seed=seed + 2,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the powered 4B prompt-transfer evaluation matrix."
    )
    parser.add_argument("--cell", action="append", type=_parse_binding, required=True)
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(dict(args.cell)) != len(args.cell):
        parser.error("cell names must be unique")
    loaded = {name: _load_rows(path) for name, path in args.cell}
    comparisons: dict[str, Any] = {}
    for index, value in enumerate(args.comparison):
        label, separator, operands = value.partition("=")
        left, comma, right = operands.partition(",")
        if not separator or not comma or left not in loaded or right not in loaded:
            parser.error("comparison must be LABEL=LEFT,RIGHT using named cells")
        comparisons[label] = {
            "left": left,
            "right": right,
            **_comparison(loaded[left][1], loaded[right][1], seed=100 + index * 10),
        }
    report = {
        "version": "prompt-transfer-matrix-analysis-v1",
        "cells": {
            name: {
                "manifest": manifest,
                "summary": _summarize_cell(rows, seed=index * 1000),
            }
            for index, (name, (manifest, rows)) in enumerate(sorted(loaded.items()))
        },
        "comparisons": comparisons,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
