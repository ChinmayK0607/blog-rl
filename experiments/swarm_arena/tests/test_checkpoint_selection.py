from __future__ import annotations

from pathlib import Path

import pytest

from swarm_ctf_eval.checkpoint_selection import Candidate, promotion_failures, select_checkpoint


def summary(score: float, *, split: str = "validation", supported: float = 1.0) -> dict:
    return {
        "split": split,
        "examples": 275,
        "selection_score": score,
        "schema_valid": 1.0,
        "supported": supported,
        "legal": 1.0,
        "broadcast": {
            "schema_valid": 1.0,
            "supported": supported,
            "legal": supported,
            "exact": score,
        },
        "act": {
            "schema_valid": 1.0,
            "supported": 1.0,
            "legal": 1.0,
            "exact": score,
        },
    }


def candidate(step: int, score: float, *, supported: float = 1.0) -> Candidate:
    value = summary(score, supported=supported)
    return Candidate(step, Path(f"step_{step}/validation/summary.json"), value, promotion_failures(value))


def test_selection_rejects_unsupported_facts_and_uses_best_eligible() -> None:
    result = select_checkpoint([candidate(40, 0.98), candidate(80, 1.0, supported=0.99)])
    assert result["selected_step"] == 40
    assert result["num_eligible"] == 1


def test_selection_prefers_earlier_step_on_exact_tie() -> None:
    result = select_checkpoint([candidate(80, 1.0), candidate(40, 1.0)])
    assert result["selected_step"] == 40


def test_selection_refuses_non_validation_candidates() -> None:
    value = summary(1.0, split="test")
    item = Candidate(40, Path("summary.json"), value, promotion_failures(value))
    with pytest.raises(ValueError, match="no checkpoint passed"):
        select_checkpoint([item])
