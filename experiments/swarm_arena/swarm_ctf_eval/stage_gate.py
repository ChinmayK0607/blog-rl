from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STAGE_GATE_VERSION = "arena-rl-stage-gates-v1"
STAGE_GATE_RESULT_VERSION = "arena-rl-stage-gate-result-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage_gates(path: Path) -> dict[str, Any]:
    gates = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in gates.items() if key != "sha256"}
    if gates.get("version") != STAGE_GATE_VERSION:
        raise ValueError("unsupported stage-gate version")
    if gates.get("sha256") != _digest(body):
        raise ValueError("stage-gate body hash mismatch")
    checkpoints = gates.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("stage gates require checkpoint rules")
    parsed_steps = sorted(int(value) for value in checkpoints)
    if [str(value) for value in parsed_steps] != list(checkpoints):
        raise ValueError("stage-gate checkpoints must be sorted canonical integer strings")
    for step, checkpoint in checkpoints.items():
        requirements = checkpoint.get("requirements")
        if int(step) < 1 or not checkpoint.get("stage") or not requirements:
            raise ValueError("each stage gate requires a positive step, stage, and requirements")
        names = [row.get("name") for row in requirements]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("stage-gate requirement names must be non-empty and unique")
        for row in requirements:
            if not isinstance(row.get("path"), list) or not row["path"]:
                raise ValueError("stage-gate requirements need a non-empty JSON path")
            predicates = {key for key in ("minimum", "maximum", "equals") if key in row}
            if len(predicates) != 1:
                raise ValueError("stage-gate requirements need exactly one predicate")
    return gates


def _resolve(document: object, path: list[object]) -> object:
    value = document
    for component in path:
        if isinstance(component, int):
            if not isinstance(value, list):
                raise ValueError(f"stage-gate path expected a list before index {component}")
            value = value[component]
        else:
            if not isinstance(value, dict) or component not in value:
                raise ValueError(f"stage-gate path is missing component {component!r}")
            value = value[component]
    return value


def evaluate_stage_gate(
    gates: dict[str, Any],
    summary: dict[str, Any],
    *,
    step: int,
    summary_sha256: str,
) -> dict[str, Any]:
    checkpoint = gates["checkpoints"].get(str(step))
    if checkpoint is None:
        raise ValueError(f"no predeclared stage gate exists for update {step}")
    results = []
    for requirement in checkpoint["requirements"]:
        observed = _resolve(summary, requirement["path"])
        if "minimum" in requirement:
            passed = float(observed) >= float(requirement["minimum"])
            predicate = {"minimum": requirement["minimum"]}
        elif "maximum" in requirement:
            passed = float(observed) <= float(requirement["maximum"])
            predicate = {"maximum": requirement["maximum"]}
        else:
            passed = observed == requirement["equals"]
            predicate = {"equals": requirement["equals"]}
        results.append(
            {
                "name": requirement["name"],
                "path": requirement["path"],
                "observed": observed,
                **predicate,
                "passed": passed,
            }
        )
    body = {
        "version": STAGE_GATE_RESULT_VERSION,
        "stage_gate_sha256": gates["sha256"],
        "summary_sha256": summary_sha256,
        "step": step,
        "stage": checkpoint["stage"],
        "status": "passed" if all(row["passed"] for row in results) else "failed",
        "requirements": results,
        "on_fail": checkpoint.get("on_fail", "stop_before_next_optimizer_update"),
    }
    return {**body, "sha256": _digest(body)}
