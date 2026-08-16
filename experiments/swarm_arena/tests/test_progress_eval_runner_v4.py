from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.log_live_rl_wandb import (
    summarize_evaluation,
    summarize_logical_update,
    watcher_complete,
)
from scripts.run_progress_eval_v4 import (
    TIER_PLANS,
    _digest,
    _distributed_roster,
    _handoff_worlds,
    _import_cached_baseline,
    _ordinary_cases,
    _validate_frozen_confirmation,
)
from scripts.run_staged_pulses import (
    _candidate_models,
    _validate_step_zero_control_config,
    _wait_retained_checkpoints,
)
from swarm_ctf_eval.progress_eval_v5 import summarize_rl_specific_progress_eval


def test_tier_plans_keep_final_large_and_development_small() -> None:
    assert TIER_PLANS["pulse"].legacy_cases == 6
    assert TIER_PLANS["pulse"].hard_cases == 6
    assert TIER_PLANS["pulse"].handoff_pairs == 6
    assert TIER_PLANS["pulse"].sides == ("BLUE", "RED")
    assert TIER_PLANS["online"].handoff_pairs == 4
    assert TIER_PLANS["selection"].handoff_pairs == 12
    assert TIER_PLANS["frozen"].handoff_pairs == 24
    assert len(TIER_PLANS["frozen"].legacy_option_orders) == 3
    assert TIER_PLANS["online"].critical_conditions == ("normal", "dropped")
    assert TIER_PLANS["online"].sides == ("BLUE", "RED")
    distributed = _distributed_roster(
        ("http://actor-1/v1", "http://actor-2/v1", "http://actor-3/v1"),
        ["same-adapter"] * 4,
        "local",
    )
    assert len({id(model) for model in distributed}) == 4
    assert [model.name for model in distributed] == ["same-adapter"] * 4


def test_step_zero_pulse_uses_one_alias_for_exact_harness_control() -> None:
    assert _candidate_models(0, "sft-opponent") == ["sft-opponent"] * 4
    assert _candidate_models(10, "sft-opponent") == [
        "blue-0",
        "blue-1",
        "blue-2",
        "blue-3",
    ]
    with pytest.raises(ValueError, match="cannot be negative"):
        _candidate_models(-1, "sft-opponent")


def test_step_zero_control_requires_identical_four_model_rosters() -> None:
    config = {
        "candidate": {"models": ["sft-opponent"] * 4},
        "baseline": {"models": ["sft-opponent"] * 4},
    }
    _validate_step_zero_control_config(config)
    config["candidate"]["models"][3] = "blue-3"
    with pytest.raises(ValueError, match="rosters must be identical"):
        _validate_step_zero_control_config(config)


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
    handoff = json.loads((data_dir / "handoff_development.json").read_text(encoding="utf-8"))
    ordinary = json.loads((data_dir / "ordinary_hard_development.json").read_text(encoding="utf-8"))
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


def test_cached_baseline_copies_only_required_sft_rows_and_raw_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    raw_records = []
    for index, (suite, case_id, condition) in enumerate(
        (
            ("ordinary_legacy", "legacy", "normal"),
            ("ordinary_hard", "hard", "normal"),
            ("handoff_critical", "critical-left", "normal"),
            ("handoff_critical", "critical-left", "dropped"),
        )
    ):
        evaluation_id = f"cached-{index}"
        raw = {"evaluation_id": evaluation_id, "raw": {"index": index}}
        raw_records.append(raw)
        rows.append(
            {
                "evaluation_id": evaluation_id,
                "raw_sha256": _digest(raw),
                "policy_variant": "sft_init",
                "policy_revision": "sft-revision",
                "suite": suite,
                "case_id": case_id,
                "condition": condition,
                "opponent_id": "sft",
                "side": "BLUE",
            }
        )
    (source / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (source / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_records),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    completed: set[str] = set()
    copied = _import_cached_baseline(
        baseline_rows_path=source / "rows.jsonl",
        rows_path=output / "rows.jsonl",
        raw_path=output / "raw.jsonl",
        completed=completed,
        baseline_revision="sft-revision",
        ordinary_case_ids={"legacy", "hard"},
        critical_case_ids={"critical-left"},
        opponent_ids={"sft"},
        sides={"BLUE"},
        expected_rows=4,
    )
    assert copied == 4
    assert len(completed) == 4
    assert len((output / "rows.jsonl").read_text().splitlines()) == 4
    assert len((output / "raw.jsonl").read_text().splitlines()) == 4


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
    assert summary["handoff_capability_rl_minus_sft"]["mean_difference"] == pytest.approx(0.3)
    assert summary["overall_gameplay_rl_minus_sft"]["mean_difference"] == pytest.approx(0.3)
    metrics = summarize_evaluation(summary)
    assert metrics["eval/rl_specific_communication_lift"] == pytest.approx(0.3)
    assert metrics["eval/overall_gameplay_rl_minus_sft"] == pytest.approx(0.3)


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


def test_wandb_sidecar_waits_for_explicit_final_eval_marker(tmp_path) -> None:
    marker = tmp_path / "COMPLETE"
    assert not watcher_complete({120}, expected_updates=120, finish_marker=marker)
    marker.touch()
    assert watcher_complete({120}, expected_updates=120, finish_marker=marker)

    expected = {}
    for index in range(4):
        checkpoint = tmp_path / f"run_blue_{index}" / "checkpoints" / "step_10"
        for relative in (
            "STABLE",
            "trainer/rank_0.pt",
            "weight/STABLE",
            "weight/adapter_config.json",
        ):
            path = checkpoint / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        adapter = checkpoint / "weight" / "adapter_model.safetensors"
        adapter.write_bytes(f"adapter-{index}".encode())
        expected[f"blue-{index}"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    _wait_retained_checkpoints(
        tmp_path,
        step=10,
        policy_adapter_sha256=expected,
        timeout=0.01,
    )
