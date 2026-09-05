from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STEP_PATTERN = re.compile(r"^step_(\d+)$")


@dataclass(frozen=True)
class Candidate:
    step: int
    summary_path: Path
    summary: dict[str, Any]
    failures: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.failures

    @property
    def rank(self) -> tuple[float, float, float, float, float, int]:
        # Prefer the earliest checkpoint when all behavioral metrics tie.
        return (
            float(self.summary["selection_score"]),
            float(self.summary["broadcast"]["exact"]),
            float(self.summary["act"]["exact"]),
            float(self.summary["broadcast"]["supported"]),
            float(self.summary["act"]["legal"]),
            -self.step,
        )


def promotion_failures(summary: dict[str, Any]) -> tuple[str, ...]:
    failures = []
    if summary.get("split") != "validation":
        failures.append("split must be validation")
    if float(summary["act"]["schema_valid"]) < 0.995:
        failures.append("action schema validity below 0.995")
    if float(summary["broadcast"]["schema_valid"]) < 0.99:
        failures.append("broadcast schema validity below 0.99")
    if float(summary["broadcast"]["supported"]) < 1.0:
        failures.append("validation contains unsupported broadcast facts")
    if float(summary["act"]["legal"]) < 0.95:
        failures.append("legal action rate below 0.95")
    return tuple(failures)


def load_candidates(results_root: Path) -> list[Candidate]:
    candidates = []
    for step_dir in sorted(results_root.glob("step_*")):
        match = STEP_PATTERN.fullmatch(step_dir.name)
        if match is None:
            continue
        summary_path = step_dir / "validation" / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates.append(
            Candidate(
                step=int(match.group(1)),
                summary_path=summary_path,
                summary=summary,
                failures=promotion_failures(summary),
            )
        )
    if not candidates:
        raise ValueError(f"no checkpoint summaries found under {results_root}")
    example_counts = {int(candidate.summary["examples"]) for candidate in candidates}
    if len(example_counts) != 1:
        raise ValueError(f"checkpoint summaries use different example counts: {sorted(example_counts)}")
    return candidates


def select_checkpoint(candidates: list[Candidate], base_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        reasons = {str(candidate.step): list(candidate.failures) for candidate in candidates}
        raise ValueError(f"no checkpoint passed validation promotion gates: {reasons}")
    selected = max(eligible, key=lambda candidate: candidate.rank)
    rows = []
    for candidate in sorted(candidates, key=lambda item: item.step):
        summary = candidate.summary
        row: dict[str, Any] = {
            "step": candidate.step,
            "summary": str(candidate.summary_path),
            "eligible": candidate.eligible,
            "gate_failures": list(candidate.failures),
            "selection_score": summary["selection_score"],
            "schema_valid": summary["schema_valid"],
            "supported": summary["supported"],
            "legal": summary["legal"],
            "broadcast_exact": summary["broadcast"]["exact"],
            "act_exact": summary["act"]["exact"],
        }
        if base_summary is not None:
            row["selection_score_delta_vs_base"] = float(summary["selection_score"]) - float(
                base_summary["selection_score"]
            )
        rows.append(row)
    return {
        "selection_protocol": "validation-only-behavioral-v1",
        "selection_rule": [
            "pass all fixed protocol gates",
            "maximize phase-balanced exactness",
            "maximize broadcast exactness",
            "maximize action exactness",
            "maximize broadcast supported rate",
            "maximize legal action rate",
            "prefer the earlier step on an exact tie",
        ],
        "selected_step": selected.step,
        "selected_summary": str(selected.summary_path),
        "num_candidates": len(candidates),
        "num_eligible": len(eligible),
        "examples_per_candidate": selected.summary["examples"],
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an SFT checkpoint using validation behavior only.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_summary = (
        json.loads(args.base_summary.read_text(encoding="utf-8")) if args.base_summary is not None else None
    )
    selection = select_checkpoint(load_candidates(args.results_root), base_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
