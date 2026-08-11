from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .regression_compare import compare, load_rows

STEP_PATTERN = re.compile(r"^step_(\d+)$")


def select_warm_start_v3(
    validation_root: Path,
    regression_v1_root: Path,
    regression_v2_root: Path,
    base_v1_rows_path: Path,
    base_v2_rows_path: Path,
    protocol_name: str = "swarm-warm-start-v3",
) -> dict[str, Any]:
    base_v1 = load_rows(base_v1_rows_path)
    base_v2 = load_rows(base_v2_rows_path)
    candidates = []
    candidate_dirs = []
    for candidate_dir in validation_root.iterdir():
        match = STEP_PATTERN.fullmatch(candidate_dir.name)
        if match is not None:
            candidate_dirs.append((int(match.group(1)), candidate_dir))
    for step, candidate_dir in sorted(candidate_dirs):
        validation_path = candidate_dir / "summary.json"
        v1_path = regression_v1_root / candidate_dir.name / "rows.jsonl"
        v2_path = regression_v2_root / candidate_dir.name / "rows.jsonl"
        if not all(path.is_file() for path in (validation_path, v1_path, v2_path)):
            continue
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        groups = validation["groups"]
        protocol_gates = {
            "all_schema_valid": validation["schema_valid"] == 1.0,
            "broadcast_grounded": groups["BROADCAST"]["grounded"] >= 0.99,
            "action_legal": groups["ACT"]["legal"] >= 0.99,
        }
        v1 = compare(base_v1, load_rows(v1_path))
        v2 = compare(base_v2, load_rows(v2_path))
        selection_score = (groups["BROADCAST"]["exact"] + groups["ACT"]["exact"]) / 2
        candidates.append(
            {
                "step": step,
                "selection_score": selection_score,
                "protocol_gates": {
                    **protocol_gates,
                    "passed": all(protocol_gates.values()),
                },
                "validation": validation,
                "regression_v1": v1,
                "regression_v2": v2,
                "eligible": (
                    all(protocol_gates.values())
                    and v1["gates"]["passed"]
                    and v2["gates"]["passed"]
                ),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda candidate: (candidate["selection_score"], -candidate["step"]),
        )
        if eligible
        else None
    )
    return {
        "selection_protocol": protocol_name,
        "decision": "adapter" if selected else "base_model",
        "selected_step": selected["step"] if selected else None,
        "reason": (
            "highest protocol exactness among checkpoints passing both paired regression suites"
            if selected
            else "no evaluated adapter passed the protocol and paired regression gates"
        ),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the replay-protected regression-safe warm start."
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--regression-v1-root", type=Path, required=True)
    parser.add_argument("--regression-v2-root", type=Path, required=True)
    parser.add_argument("--base-v1-rows", type=Path, required=True)
    parser.add_argument("--base-v2-rows", type=Path, required=True)
    parser.add_argument("--protocol-name", default="swarm-warm-start-v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_warm_start_v3(
        args.validation_root,
        args.regression_v1_root,
        args.regression_v2_root,
        args.base_v1_rows,
        args.base_v2_rows,
        args.protocol_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
