from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .regression_compare import compare, load_rows

STEP_PATTERN = re.compile(r"^step_(\d+)$")
MIN_BROADCAST_SUPPORTED = 0.95
MIN_ACTION_LEGAL = 0.95


def select_warm_start(
    validation_root: Path,
    regression_root: Path,
    base_rows_path: Path,
) -> dict[str, Any]:
    base_rows = load_rows(base_rows_path)
    candidates = []
    validation_dirs = []
    for validation_dir in validation_root.iterdir():
        match = STEP_PATTERN.fullmatch(validation_dir.name)
        if match is None:
            continue
        validation_dirs.append((int(match.group(1)), validation_dir))
    for step, validation_dir in sorted(validation_dirs):
        validation_path = validation_dir / "validation" / "summary.json"
        regression_path = regression_root / validation_dir.name / "rows.jsonl"
        if not validation_path.is_file() or not regression_path.is_file():
            continue
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        regression = compare(base_rows, load_rows(regression_path))
        protocol_gates = {
            "schema_valid": validation["schema_valid"] == 1.0,
            "broadcast_supported": validation["broadcast"]["supported"] >= MIN_BROADCAST_SUPPORTED,
            "action_legal": validation["act"]["legal"] >= MIN_ACTION_LEGAL,
        }
        candidates.append(
            {
                "step": step,
                "selection_score": float(validation["selection_score"]),
                "broadcast_exact": float(validation["broadcast"]["exact"]),
                "action_exact": float(validation["act"]["exact"]),
                "protocol_gates": {**protocol_gates, "passed": all(protocol_gates.values())},
                "regression": regression,
                "eligible": all(protocol_gates.values()) and regression["gates"]["passed"],
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(eligible, key=lambda item: (item["selection_score"], -item["step"])) if eligible else None
    return {
        "selection_protocol": "swarm-warm-start-v1",
        "decision": "adapter" if selected is not None else "base_model",
        "selected_step": selected["step"] if selected is not None else None,
        "reason": (
            "highest validation protocol score among regression-safe checkpoints"
            if selected is not None
            else "no evaluated adapter passed both protocol and regression gates"
        ),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a regression-safe SFT protocol warm start.")
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--base-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_warm_start(args.validation_root, args.regression_root, args.base_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
