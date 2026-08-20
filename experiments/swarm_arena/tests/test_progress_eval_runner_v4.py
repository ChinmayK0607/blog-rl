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
from scripts.run_live_artifact_mirror import (
    MirrorState,
    checkpoint_files,
    compact_files,
    progress_step,
)
from scripts.run_pair7_communication_eval import summarize as summarize_pair7
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
    _validate_pair7_summary,
    _validate_step_zero_control_config,
    _validate_training_pair_summary,
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
                        "critical_capture": (
                            suite == "handoff_critical"
                            and variant == "candidate_rl"
                            and condition == "normal"
                        ),
                        "sender_target_fact": (
                            suite == "handoff_critical" and condition == "normal"
                        ),
                    }
                )
    rows.append(
        {
            **common,
            "independent_id": "zero-budget-unit",
            "case_id": "zero-budget-case",
            "suite": "handoff_critical",
            "policy_variant": "candidate_rl",
            "policy_revision": "candidate_rl",
            "condition": "zero_budget",
            "terminal_return": 0.0,
            "critical_capture": False,
            "sender_target_fact": False,
            "broadcast_protocol_rate": None,
            "broadcast_grounded_rate": None,
        }
    )
    summary = summarize_rl_specific_progress_eval(rows)
    assert summary["rl_specific_communication_lift"]["mean_difference"] == pytest.approx(0.3)
    assert summary["rl_specific_communication_lift"]["independent_units"] == 1
    assert summary["critical_minus_decoy_specificity"]["mean_difference"] == pytest.approx(0.3)
    assert summary["handoff_capability_rl_minus_sft"]["mean_difference"] == pytest.approx(0.3)
    assert summary["overall_gameplay_rl_minus_sft"]["mean_difference"] == pytest.approx(0.3)
    assert summary["communication_mechanism"]["candidate_sender_target_fact_rate"][
        "mean_difference"
    ] == pytest.approx(1.0)
    assert summary["communication_mechanism"]["rl_specific_capture_lift"][
        "mean_difference"
    ] == pytest.approx(1.0)
    assert summary["candidate_protocol"]["broadcast_protocol_rate"] == pytest.approx(1.0)
    assert summary["candidate_protocol_denominators"]["broadcast_protocol_rate"] == {
        "defined_rows": 6,
        "undefined_rows": 1,
    }
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
                    "focused_agent": "blue-1",
                    "focused_phase": "BROADCAST",
                    "opponent": {"family": "sft"},
                },
                "replicas": [
                    {
                        "return": 0.2,
                        "advantages": {"blue-0": 0.0, "blue-1": 0.1},
                    },
                    {
                        "return": 0.0,
                        "advantages": {"blue-0": 0.0, "blue-1": -0.1},
                    },
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
    assert metrics["controller/mean_abs_focused_advantage"] == 0.05
    assert metrics["controller/focused_nonzero_advantage_rate"] == 0.5
    assert metrics["curriculum/focused_broadcast_fraction"] == 0.5
    assert metrics["controller/mean_abs_focused_advantage/broadcast"] == 0.1
    assert metrics["controller/mean_focused_action_diversity"] == 0.0


def test_controller_summary_reports_within_group_focused_action_diversity() -> None:
    record = {
        "step": 0,
        "groups": [
            {
                "scenario": {
                    "kind": "critical",
                    "curriculum_stage": "compact",
                    "focused_agent": "blue-1",
                    "focused_phase": "ACT",
                    "opponent": {"family": "sft"},
                },
                "replicas": [
                    {
                        "return": 0.1,
                        "advantages": {"blue-1": 0.1},
                        "focused_action": {"type": "CAPTURE", "target": "V13"},
                    },
                    {
                        "return": -0.1,
                        "advantages": {"blue-1": -0.1},
                        "focused_action": {"type": "CAPTURE", "target": "V19"},
                    },
                    {
                        "return": 0.1,
                        "advantages": {"blue-1": 0.1},
                        "focused_action": {"type": "CAPTURE", "target": "V13"},
                    },
                    {
                        "return": -0.1,
                        "advantages": {"blue-1": -0.1},
                        "focused_action": {"type": "CAPTURE", "target": "V19"},
                    },
                ],
            }
        ],
    }
    metrics = summarize_logical_update(record)
    assert metrics["controller/mean_focused_action_diversity"] == 0.5
    assert metrics["controller/focused_nonzero_advantage_rate"] == 1.0


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


def test_live_mirror_accepts_only_complete_checksum_bound_checkpoints(tmp_path: Path) -> None:
    expected = {}
    for index in range(4):
        checkpoint = tmp_path / f"run_blue_{index}" / "checkpoints" / "step_10"
        (checkpoint / "weight").mkdir(parents=True)
        (checkpoint / "STABLE").touch()
        (checkpoint / "weight" / "STABLE").touch()
        adapter = checkpoint / "weight" / "adapter_model.safetensors"
        adapter.write_bytes(f"adapter-{index}".encode())
        (checkpoint / "weight" / "adapter_config.json").write_text("{}\n")
        expected[f"blue-{index}"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    files = checkpoint_files(tmp_path, 10, {"policy_adapter_sha256": expected})
    assert len(files) == 8
    assert all("rank_0.pt" not in name for name in files)
    expected["blue-2"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpoint_files(tmp_path, 10, {"policy_adapter_sha256": expected})


def test_live_mirror_progress_and_state_are_resume_safe(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps([{"step": 0}, {"step": 1}]))
    assert progress_step(progress) == 2
    state_path = tmp_path / "state.json"
    state = MirrorState((10, 20), 22, True)
    state.write(state_path)
    assert MirrorState.load(state_path) == state


def test_pair7_summary_separates_communication_lift_from_decoy_tactics(
    tmp_path: Path,
) -> None:
    rows = []
    for kind, normal, dropped in (("critical", 0.8, 0.2), ("decoy", 0.5, 0.4)):
        for condition, value in (
            ("normal", normal),
            ("dropped", dropped),
            ("sender_shuffled", dropped),
            ("target_swapped", dropped),
        ):
            for _world in ("left", "right"):
                rows.append(
                    {
                        "kind": kind,
                        "condition": condition,
                        "terminal_return": value,
                        "receiver_target_action": condition == "normal",
                        "sender_target_fact": condition == "normal",
                        "broadcast_valid": 1.0,
                        "broadcast_grounded": 1.0,
                        "action_valid": 1.0,
                    }
                )
    summary = summarize_pair7(rows)
    assert summary["critical"]["normal_minus_dropped_return"] == pytest.approx(0.6)
    assert summary["specificity"]["critical_minus_decoy_normal_dropped_lift"] == pytest.approx(0.5)
    assert summary["specificity"]["critical_minus_decoy_target_swapped_lift"] == pytest.approx(0.5)
    output = tmp_path / "summary.json"
    output.write_text(json.dumps(summary))
    _validate_pair7_summary(output, repetitions=1)
    metrics = summarize_evaluation(summary)
    assert metrics["eval/train_pair/normal_minus_dropped_return"] == pytest.approx(0.6)
    assert metrics["eval/train_pair/critical_minus_decoy_specificity"] == pytest.approx(0.5)


def test_multipair_summary_preserves_per_pair_signal(tmp_path: Path) -> None:
    rows = []
    for pair_index, lift in ((7, 0.6), (9, 0.2)):
        for kind in ("critical", "decoy"):
            for condition in ("normal", "dropped", "sender_shuffled", "target_swapped"):
                for world in ("left", "right"):
                    terminal_return = (
                        lift
                        if kind == "critical" and condition == "normal"
                        else 0.0
                    )
                    rows.append(
                        {
                            "pair_index": pair_index,
                            "kind": kind,
                            "condition": condition,
                            "world": world,
                            "terminal_return": terminal_return,
                            "receiver_target_action": condition == "normal",
                            "sender_target_fact": condition == "normal",
                            "broadcast_valid": 1.0,
                            "broadcast_grounded": 1.0,
                            "action_valid": 1.0,
                        }
                    )
    summary = summarize_pair7(rows, (7, 9))
    assert summary["version"] == "multipair-semantic-communication-eval-v3"
    assert summary["critical"]["normal_minus_dropped_return"] == pytest.approx(0.4)
    assert summary["by_pair"]["7"]["critical"][
        "normal_minus_dropped_return"
    ] == pytest.approx(0.6)
    assert summary["by_pair"]["9"]["critical"][
        "normal_minus_dropped_return"
    ] == pytest.approx(0.2)
    output = tmp_path / "summary.json"
    output.write_text(json.dumps(summary))
    _validate_training_pair_summary(
        output,
        repetitions=1,
        pair_indices=(7, 9),
    )
    invalid = {**summary, "protocol": {**summary["protocol"], "action_valid_rate": 0.99}}
    output.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="invalid or ungrounded"):
        _validate_training_pair_summary(
            output,
            repetitions=1,
            pair_indices=(7, 9),
        )
    metrics = summarize_evaluation(summary)
    assert metrics["eval/train_pair/normal_minus_dropped_return"] == pytest.approx(0.4)
    assert metrics["eval/train_pair/7/normal_minus_dropped_return"] == pytest.approx(0.6)
    assert metrics["eval/train_pair/9/normal_minus_dropped_return"] == pytest.approx(0.2)


def test_live_mirror_collects_compact_eval_but_not_raw_generations(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluations" / "update-10"
    evaluation.mkdir(parents=True)
    for name in ("manifest.json", "rows.jsonl", "summary.json", "raw.jsonl"):
        (evaluation / name).write_text(name)
    files = compact_files(tmp_path)
    assert "evaluations/update-10/summary.json" in files
    assert "evaluations/update-10/rows.jsonl" in files
    assert "evaluations/update-10/raw.jsonl" not in files
