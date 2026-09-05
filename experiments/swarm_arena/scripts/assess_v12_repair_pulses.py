#!/usr/bin/env python3
"""Apply the predeclared V12 update-40 fail-fast rule without opening frozen data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(summary: dict[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        try:
            value = value[key]
        except (KeyError, TypeError) as error:
            raise ValueError(f"missing pulse metric {'.'.join(keys)}") from error
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"non-numeric pulse metric {'.'.join(keys)}")
    return float(value)


def extract(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "semantic": metric(
            summary,
            "semantic",
            "candidate_critical_normal_minus_target_swapped",
            "mean_difference",
        ),
        "specificity": metric(
            summary,
            "semantic",
            "critical_minus_decoy_semantic_specificity",
            "mean_difference",
        ),
        "ordinary_legacy": metric(
            summary, "capability_rl_minus_sft", "ordinary_legacy", "mean_difference"
        ),
        "ordinary_hard": metric(
            summary, "capability_rl_minus_sft", "ordinary_hard", "mean_difference"
        ),
    }


def assess(baseline: dict[str, Any], checkpoints: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    if [step for step, _ in checkpoints] != [20, 40]:
        raise ValueError("repair screen requires exactly update 20 then update 40")
    base = extract(baseline)
    rows = []
    for step, summary in checkpoints:
        values = extract(summary)
        ordinary_improved = min(values["ordinary_legacy"], values["ordinary_hard"]) > min(
            base["ordinary_legacy"], base["ordinary_hard"]
        )
        robustness_improved = values["specificity"] > base["specificity"]
        rows.append(
            {
                "step": step,
                "metrics": values,
                "ordinary_retention_improved": ordinary_improved,
                "decoy_robustness_improved": robustness_improved,
                "semantic_positive": values["semantic"] > 0,
            }
        )
    stop = all(
        not row["ordinary_retention_improved"]
        and not row["decoy_robustness_improved"]
        and not row["semantic_positive"]
        for row in rows
    )
    return {
        "version": "v12-repair-pulse-screen-v1",
        "frozen_data_opened": False,
        "baseline": base,
        "checkpoints": rows,
        "decision": "stop" if stop else "continue",
        "rule": (
            "stop only if updates 20 and 40 both fail to improve ordinary retention, "
            "fail to improve critical-minus-decoy specificity, and have non-positive semantics"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initializer-summary", type=Path, required=True)
    parser.add_argument("--update20-summary", type=Path, required=True)
    parser.add_argument("--update40-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.initializer_summary, args.update20_summary, args.update40_summary]
    summaries = [json.loads(path.read_text()) for path in paths]
    result = assess(summaries[0], [(20, summaries[1]), (40, summaries[2])])
    result["input_sha256"] = {path.name: sha256(path) for path in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
