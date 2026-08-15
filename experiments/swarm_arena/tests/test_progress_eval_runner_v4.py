from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.run_progress_eval_v4 import (
    TIER_PLANS,
    _handoff_worlds,
    _ordinary_cases,
    _validate_frozen_confirmation,
)


def test_tier_plans_keep_final_large_and_development_small() -> None:
    assert TIER_PLANS["online"].handoff_pairs == 4
    assert TIER_PLANS["selection"].handoff_pairs == 12
    assert TIER_PLANS["frozen"].handoff_pairs == 24
    assert len(TIER_PLANS["frozen"].legacy_option_orders) == 3
    assert TIER_PLANS["online"].critical_conditions == ("normal", "dropped")


def test_frozen_tier_requires_exact_design_digest() -> None:
    design = {"version": "test", "status": "frozen"}
    with pytest.raises(ValueError, match="frozen evaluation requires"):
        _validate_frozen_confirmation("frozen", design, None)
    message = None
    try:
        _validate_frozen_confirmation("frozen", design, "wrong")
    except ValueError as error:
        message = str(error)
    assert message is not None
    confirmation = message.rsplit(" ", 1)[-1]
    _validate_frozen_confirmation("frozen", design, confirmation)


def test_runner_expands_both_handoff_worlds_and_hard_cases() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "rl_v4"
    handoff = json.loads(
        (data_dir / "handoff_development.json").read_text(encoding="utf-8")
    )
    ordinary = json.loads(
        (data_dir / "ordinary_hard_development.json").read_text(encoding="utf-8")
    )
    handoff_rows = _handoff_worlds("online", handoff)
    ordinary_rows = _ordinary_cases("online", ordinary)
    assert len(handoff_rows) == 4 * 2 * 2
    assert len({row[1] for row in handoff_rows}) == 4
    assert {row[2] for row in handoff_rows} == {
        "handoff_critical",
        "handoff_decoy",
    }
    assert len(ordinary_rows) == 8
    assert {row[3] for row in ordinary_rows} == {
        "ordinary_legacy",
        "ordinary_hard",
    }
