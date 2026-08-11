from __future__ import annotations

from swarm_ctf_eval.regression import (
    FROZEN_REGRESSION_CASES,
    REGRESSION_MANIFEST_SHA256,
    summarize_regression_rows,
    validate_response,
)
from swarm_ctf_eval.regression_compare import compare


def test_frozen_regression_manifest_is_balanced_and_unique() -> None:
    assert len(FROZEN_REGRESSION_CASES) == 256
    assert len({row.id for row in FROZEN_REGRESSION_CASES}) == 256
    counts = {}
    for row in FROZEN_REGRESSION_CASES:
        counts[row.category] = counts.get(row.category, 0) + 1
    assert set(counts.values()) == {64}
    assert len(REGRESSION_MANIFEST_SHA256) == 64


def test_response_validation_detects_arena_leakage() -> None:
    case = FROZEN_REGRESSION_CASES[0]
    assert validate_response(case, '{"action_id":"A0"}') == {
        "valid_json": True,
        "exact": False,
        "arena_leakage": True,
    }
    assert validate_response(case, "not json")["valid_json"] is False
    assert validate_response(case, '{"answer":{"facts":[]}}')["arena_leakage"] is True


def test_summary_and_comparison_are_paired() -> None:
    rows = [
        {
            "id": case.id,
            "category": case.category,
            "valid_json": True,
            "exact": True,
            "arena_leakage": False,
        }
        for case in FROZEN_REGRESSION_CASES
    ]
    summary = summarize_regression_rows(rows)
    assert summary["exact"] == 1.0
    result = compare(rows, [dict(row) for row in rows])
    assert result["gates"]["passed"]


def test_comparison_rejects_category_regression() -> None:
    base = []
    adapter = []
    for case in FROZEN_REGRESSION_CASES:
        row = {
            "id": case.id,
            "category": case.category,
            "valid_json": True,
            "exact": True,
            "arena_leakage": False,
        }
        base.append(row)
        adapter.append(dict(row, exact=case.category != "arithmetic"))
    result = compare(base, adapter)
    assert not result["gates"]["every_category_drop_within_0_05"]
    assert not result["gates"]["passed"]
