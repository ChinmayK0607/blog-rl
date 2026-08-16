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
from scripts.log_live_rl_wandb import summarize_evaluation, summarize_logical_update
from swarm_ctf_eval.progress_eval_v5 import summarize_rl_specific_progress_eval


def test_tier_plans_keep_final_large_and_development_small() -> None:
    assert TIER_PLANS["pulse"].handoff_pairs == 1
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


def test_rl_specific_summary_requires_gain_over_sft_and_decoy() -> None:
    rows = []
    common = {
        "opponent_id": "sft",
        "opponent_revision": "sft-revision",
        "side": "BLUE",
        "sampling_key": "sample",
        "messages_nonempty": 1,
        "broadcast_protocol_rate": 1.0,
        "broadcast_grounded_rate": 1.0,
        "action_protocol_rate": 1.0,
    }
    for suite in ("ordinary_legacy", "ordinary_hard"):
        for variant, value in (("candidate_rl", 0.4), ("sft_init", 0.1)):
            rows.append(
                {
                    **common,
                    "independent_id": f"{suite}-unit",
                    "case_id": f"{suite}-case",
                    "suite": suite,
                    "policy_variant": variant,
                    "policy_revision": variant,
                    "condition": "normal",
                    "terminal_return": value,
                }
            )
    for suite in ("handoff_critical", "handoff_decoy"):
        variants = ("candidate_rl", "sft_init") if suite == "handoff_critical" else ("candidate_rl",)
        for variant in variants:
            for condition in ("normal", "dropped"):
                effect = 0.3 if suite == "handoff_critical" and variant == "candidate_rl" else 0.0
                rows.append(
                    {
                        **common,
                        "independent_id": "handoff-unit",
                        "case_id": f"{suite}-case",
                        "suite": suite,
                        "policy_variant": variant,
                        "policy_revision": variant,
                        "condition": condition,
                        "terminal_return": effect if condition == "normal" else 0.0,
                    }
                )
    summary = summarize_rl_specific_progress_eval(rows)
    assert summary["rl_specific_communication_lift"]["mean_difference"] == pytest.approx(0.3)
    assert summary["rl_specific_communication_lift"]["independent_units"] == 1
    assert summary["critical_minus_decoy_specificity"]["mean_difference"] == pytest.approx(0.3)
    metrics = summarize_evaluation(summary)
    assert metrics["eval/rl_specific_communication_lift"] == pytest.approx(0.3)


def test_wandb_controller_summary_exposes_curriculum_and_opponent_metrics() -> None:
    record = {
        "step": 2,
        "groups": [
            {
                "scenario": {
                    "kind": "critical",
                    "curriculum_stage": "handoff",
                    "opponent": {"family": "sft"},
                },
                "replicas": [
                    {"return": 0.2, "advantage": 0.1},
                    {"return": 0.0, "advantage": -0.1},
                ],
            },
            {
                "scenario": {
                    "kind": "decoy",
                    "curriculum_stage": "handoff",
                    "opponent": {"family": "current"},
                },
                "replicas": [
                    {"return": 0.0, "advantage": 0.0},
                    {"return": 0.0, "advantage": 0.0},
                ],
            },
        ],
    }
    metrics = summarize_logical_update(record)
    assert metrics["controller/update"] == 3
    assert metrics["curriculum/stage"] == "handoff"
    assert metrics["curriculum/critical_fraction"] == 0.5
    assert metrics["return/by_opponent/sft"] == 0.1
