#!/usr/bin/env python3
"""Fail-closed development-only selector for the frozen V11 evaluation.

The V11 design fixes candidate order to 60, 120, 180.  This utility never
opens frozen data and deliberately accepts no fallback rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STEPS = (60, 120, 180)
GATES = {
    "semantic_return": ("semantic", "candidate_critical_normal_minus_target_swapped", "mean_difference", ">", 0.0),
    "semantic_specificity": ("semantic", "critical_minus_decoy_semantic_specificity", "mean_difference", ">", 0.0),
    "ordinary_legacy": ("capability_rl_minus_sft", "ordinary_legacy", "mean_difference", ">=", 0.0),
    "ordinary_hard": ("capability_rl_minus_sft", "ordinary_hard", "mean_difference", ">=", 0.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value(summary: dict[str, Any], gate: tuple[str, str, str, str, float]) -> float:
    first, second, third, _, _ = gate
    try:
        result = summary[first][second][third]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing or ambiguous selector key {first}.{second}.{third}") from error
    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise ValueError(f"non-numeric selector value {first}.{second}.{third}")
    return float(result)


def assess(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("tier") != "selection":
        raise ValueError("summary tier must be selection")
    measurements = {name: value(summary, gate) for name, gate in GATES.items()}
    passed = {
        name: measurements[name] > gate[4] if gate[3] == ">" else measurements[name] >= gate[4]
        for name, gate in GATES.items()
    }
    return {"measurements": measurements, "passed": passed, "eligible": all(passed.values())}


def select(items: list[tuple[int, Path]], design: Path, clarification: Path) -> dict[str, Any]:
    observed_steps = [step for step, _ in items]
    if not observed_steps or observed_steps != list(STEPS[: len(observed_steps)]):
        raise ValueError(f"candidate summaries must be a non-empty ordered prefix of {list(STEPS)}")
    candidates = []
    selected_step: int | None = None
    for step, path in items:
        if not path.is_file():
            raise ValueError(f"missing candidate summary: {path}")
        summary = json.loads(path.read_text())
        assessed = assess(summary)
        entry = {"step": step, "summary_path": str(path), "summary_sha256": sha256(path), **assessed}
        candidates.append(entry)
        if selected_step is None and assessed["eligible"]:
            selected_step = step
    if selected_step is not None:
        for step in STEPS:
            if step > selected_step and step not in observed_steps:
                candidates.append({"step": step, "status": "skipped_after_earliest_selection"})
        for entry in candidates:
            if entry["step"] > selected_step:
                entry["skipped_after_earliest_selection"] = True
    return {
        "version": "v11-development-selector-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "frozen_data_opened": False,
        "candidate_order": list(STEPS),
        "design_path": str(design),
        "design_sha256": sha256(design),
        "clarification_path": str(clarification),
        "clarification_sha256": sha256(clarification),
        "gate_rule": {
            "semantic_return": "semantic.candidate_critical_normal_minus_target_swapped.mean_difference > 0",
            "semantic_specificity": "semantic.critical_minus_decoy_semantic_specificity.mean_difference > 0",
            "ordinary_legacy": "capability_rl_minus_sft.ordinary_legacy.mean_difference >= 0",
            "ordinary_hard": "capability_rl_minus_sft.ordinary_hard.mean_difference >= 0",
        },
        "selection_rule": "earliest candidate in [60,120,180] satisfying all four gates; no fallback",
        "selected_step": selected_step,
        "frozen_launch_authorized": selected_step is not None,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--clarification", type=Path, required=True)
    parser.add_argument("--summary", action="append", required=True, metavar="STEP:PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parsed = []
    for item in args.summary:
        step, sep, path = item.partition(":")
        if not sep or not step.isdigit() or not path:
            raise ValueError(f"invalid --summary {item!r}; expected STEP:PATH")
        parsed.append((int(step), Path(path)))
    result = select(parsed, args.design, args.clarification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
