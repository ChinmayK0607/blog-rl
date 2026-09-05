from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean_interval(
    values: list[float],
    *,
    trials: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty paired endpoint")
    mean = statistics.fmean(values)
    if len(values) == 1:
        interval = [mean, mean]
    else:
        generator = random.Random(seed)
        samples = sorted(
            statistics.fmean(values[generator.randrange(len(values))] for _ in values)
            for _ in range(trials)
        )
        interval = [
            samples[int(0.025 * (trials - 1))],
            samples[int(0.975 * (trials - 1))],
        ]
    return {
        "independent_units": len(values),
        "mean": mean,
        "interval_95": interval,
        "positive_rate": statistics.fmean(value > 0 for value in values),
        "values": values,
    }


def _paired_effects(
    rows: list[dict[str, Any]],
    *,
    field: str,
    left: str,
    right: str,
) -> dict[str, list[float]]:
    cells: dict[tuple[int, str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if int(row["repeat"]) >= 4:
            continue
        key = (
            int(row["pair_index"]),
            str(row["kind"]),
            str(row["world"]),
            int(row["repeat"]),
        )
        cells[key][str(row["condition"])] = float(row[field])
    per_pair: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (pair_index, kind, _, _), conditions in cells.items():
        if left not in conditions or right not in conditions:
            raise ValueError(f"incomplete paired cell for {pair_index}/{kind}: {conditions}")
        per_pair[(pair_index, kind)].append(conditions[left] - conditions[right])
    result: dict[str, list[float]] = defaultdict(list)
    pair_ids = sorted({pair_index for pair_index, _ in per_pair})
    for pair_index in pair_ids:
        for kind in ("critical", "decoy"):
            result[kind].append(statistics.fmean(per_pair[(pair_index, kind)]))
        result["critical_minus_decoy"].append(
            result["critical"][-1] - result["decoy"][-1]
        )
    return dict(result)


def _selection(summary: dict[str, Any]) -> dict[str, Any]:
    critical = [
        cell
        for cell in summary["cells"]
        if cell["kind"] == "critical" and cell["condition"] == "generated"
    ]
    bands: dict[str, list[dict[str, Any]]] = {
        "primary_receiver_band": [],
        "hard_reserve": [],
        "easy_stabilizers": [],
        "low_signal": [],
    }
    for cell in critical:
        compact = {
            "pair_index": cell["pair_index"],
            "world": cell["world"],
            "capture_pass_at_1": cell["capture_pass_at_1"],
            "capture_pass_at_4": cell["capture_pass_at_4"],
            "capture_pass_at_8": cell["capture_pass_at_8"],
            "return_contrast_at_4": cell["return_contrast_at_4"],
            "mean_return": cell["mean_return"],
            "best_return_at_4": cell["best_return_at_4"],
        }
        capture = float(cell["capture_pass_at_1"])
        contrast = float(cell["return_contrast_at_4"])
        if contrast >= 0.75 and 0.125 <= capture <= 0.5:
            bands["primary_receiver_band"].append(compact)
        elif contrast >= 0.75 and capture == 0:
            bands["hard_reserve"].append(compact)
        elif contrast >= 0.5 and capture > 0.5:
            bands["easy_stabilizers"].append(compact)
        else:
            bands["low_signal"].append(compact)
    return {
        "status": "exploratory training-split selection; thresholds chosen after this screen",
        "band_definitions": {
            "primary_receiver_band": (
                "return contrast@4 >= 0.75 and target-capture pass@1 between 0.125 and 0.5"
            ),
            "hard_reserve": "return contrast@4 >= 0.75 and target-capture pass@1 = 0",
            "easy_stabilizers": "return contrast@4 >= 0.5 and target-capture pass@1 > 0.5",
            "low_signal": "all remaining screened critical worlds",
        },
        "bands": bands,
        "counts": {name: len(values) for name, values in bands.items()},
        "matched_decoy_rule": (
            "retain the matched decoy for every selected critical pair/world; never train the "
            "critical row without its null control"
        ),
        "sender_rule": (
            "do not select sender-BROADCAST updates from this slice: generated target-fact "
            "coverage is saturated and reference insertion is an identity/null control"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed training-only pass@k screen.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.input_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    endpoints = {}
    fields = {
        "terminal_return": "terminal_return",
        "turn_zero_target_capture": "target_captured_turn_zero",
        "terminal_target_capture": "target_captured",
        "receiver_target_action": "receiver_target_action",
    }
    for name, field in fields.items():
        generated_dropped = _paired_effects(
            rows,
            field=field,
            left="generated",
            right="dropped",
        )
        reference_generated = _paired_effects(
            rows,
            field=field,
            left="reference",
            right="generated",
        )
        endpoints[name] = {
            "generated_minus_dropped": {
                key: _mean_interval(values, seed=index)
                for index, (key, values) in enumerate(sorted(generated_dropped.items()))
            },
            "reference_minus_generated_null": {
                key: _mean_interval(values, seed=100 + index)
                for index, (key, values) in enumerate(sorted(reference_generated.items()))
            },
        }
    report = {
        "schema_version": "arena-training-passk-analysis-v1",
        "manifest_sha256": summary["manifest_sha256"],
        "rows": len(rows),
        "protocol_valid_rate": statistics.fmean(row["protocol_valid"] for row in rows),
        "requests": sum(int(row["requests"]) for row in rows),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
        "rollout_wall_seconds": summary["wall_seconds"],
        "passk_aggregate": summary["aggregate"],
        "paired_endpoints": endpoints,
        "selection": _selection(summary),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
