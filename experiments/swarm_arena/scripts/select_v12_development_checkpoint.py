#!/usr/bin/env python3
"""Fail-closed V12 selector with prospectively fixed ordinary non-inferiority."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STEPS = (20, 40, 80, 120, 160)
ORDINARY_MARGIN = -0.02


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint(summary: dict[str, Any], first: str, second: str) -> dict[str, Any]:
    try:
        value = summary[first][second]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing selector endpoint {first}.{second}") from error
    if not isinstance(value, dict):
        raise ValueError(f"non-object selector endpoint {first}.{second}")
    return value


def number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"selector field {key} is not numeric")
    return float(value)


def lower_bound(row: dict[str, Any]) -> float:
    interval = row.get("mean_difference_95")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(not isinstance(value, (int, float)) for value in interval)
    ):
        raise ValueError("selector endpoint omits a two-sided mean_difference_95")
    return float(interval[0])


def assess(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("tier") != "selection":
        raise ValueError("summary tier must be selection")
    semantic = endpoint(
        summary, "semantic", "candidate_critical_normal_minus_target_swapped"
    )
    specificity = endpoint(
        summary, "semantic", "critical_minus_decoy_semantic_specificity"
    )
    legacy = endpoint(summary, "capability_rl_minus_sft", "ordinary_legacy")
    hard = endpoint(summary, "capability_rl_minus_sft", "ordinary_hard")
    measurements = {
        "semantic_return": number(semantic, "mean_difference"),
        "semantic_specificity": number(specificity, "mean_difference"),
        "ordinary_legacy_lower_95": lower_bound(legacy),
        "ordinary_hard_lower_95": lower_bound(hard),
    }
    passed = {
        "semantic_return": measurements["semantic_return"] > 0,
        "semantic_specificity": measurements["semantic_specificity"] > 0,
        "ordinary_legacy_noninferiority": (
            measurements["ordinary_legacy_lower_95"] >= ORDINARY_MARGIN
        ),
        "ordinary_hard_noninferiority": (
            measurements["ordinary_hard_lower_95"] >= ORDINARY_MARGIN
        ),
    }
    return {"measurements": measurements, "passed": passed, "eligible": all(passed.values())}


def select(items: list[tuple[int, Path]], design: Path) -> dict[str, Any]:
    observed = [step for step, _ in items]
    if not observed or observed != list(STEPS[: len(observed)]):
        raise ValueError(f"candidate summaries must be an ordered prefix of {list(STEPS)}")
    candidates = []
    selected_step = None
    for step, path in items:
        if not path.is_file():
            raise ValueError(f"missing candidate summary: {path}")
        result = assess(json.loads(path.read_text()))
        candidates.append(
            {"step": step, "summary_sha256": sha256(path), "summary_path": str(path), **result}
        )
        if selected_step is None and result["eligible"]:
            selected_step = step
            break
    return {
        "version": "v12-development-selector-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "frozen_data_opened": False,
        "candidate_order": list(STEPS),
        "design_sha256": sha256(design),
        "ordinary_noninferiority_margin": ORDINARY_MARGIN,
        "selection_rule": (
            "earliest candidate with positive semantic mean, positive specificity mean, "
            "and legacy+hard clustered lower 95% bounds >= -0.02; no fallback"
        ),
        "selected_step": selected_step,
        "frozen_launch_authorized": selected_step is not None,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--summary", action="append", required=True, metavar="STEP:PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = []
    for item in args.summary:
        step, separator, path = item.partition(":")
        if not separator or not step.isdigit() or not path:
            raise ValueError(f"invalid summary binding: {item}")
        items.append((int(step), Path(path)))
    result = select(items, args.design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
