from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.audit_paired_credit_toy import audit as audit_paired_credit
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
    _load_manifest,
    _ordinary_cases,
    _selection_design_counts,
    _validate_frozen_confirmation,
)
from scripts.run_staged_pulses import (
    _candidate_models,
    _target_swap_scope_args,
    _validate_pair7_summary,
    _validate_step_zero_control_config,
    _validate_training_pair_summary,
    _wait_retained_checkpoints,
)
from scripts.run_v10_clean_holdout import (
    _clean_handoff_worlds,
    _clean_ordinary_cases,
    _expected_rows,
    _load_lock,
    _validate_bindings,
)
from scripts.run_v10_holdout_mirror import _snapshot_rows, _write_raw_shard
from swarm_ctf_eval.evaluation_contract import (
    initializer_improvement,
    pulse_config,
    required_independent_units,
    staged_evaluation_budget,
    verify_served_adapters,
)
from swarm_ctf_eval.progress_eval_v5 import summarize_rl_specific_progress_eval
from swarm_ctf_eval.semantic_holdout import summarize_semantic_holdout


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
    assert TIER_PLANS["selection"].critical_conditions[-1] == "target_swapped"
    assert TIER_PLANS["frozen"].critical_conditions[-1] == "target_swapped"
    assert TIER_PLANS["online"].sides == ("BLUE", "RED")
    distributed = _distributed_roster(
        ("http://actor-1/v1", "http://actor-2/v1", "http://actor-3/v1"),
        ["same-adapter"] * 4,
        "local",
    )
    assert len({id(model) for model in distributed}) == 4
    assert [model.name for model in distributed] == ["same-adapter"] * 4


def test_step_zero_pulse_measures_actual_initializer_not_sft_harness() -> None:
    assert _candidate_models(0, "sft-opponent") == [f"blue-{i}" for i in range(4)]
    assert _candidate_models(10, "sft-opponent") == [
        "blue-0",
        "blue-1",
        "blue-2",
        "blue-3",
    ]
    with pytest.raises(ValueError, match="cannot be negative"):
        _candidate_models(-1, "sft-opponent")


def test_pulse_identity_checks_sft_revision_and_every_served_weight(tmp_path: Path) -> None:
    hashes = {}
    registry = {"data": []}
    for alias in [f"blue-{i}" for i in range(4)] + ["sft-opponent", "history"]:
        root = tmp_path / alias
        root.mkdir()
        data = alias.encode()
        (root / "adapter_model.safetensors").write_bytes(data)
        hashes[alias] = hashlib.sha256(data).hexdigest()
        registry["data"].append({"id": alias, "root": str(root)})
    snapshots = {
        family: SimpleNamespace(revision=revision, model_name=alias, adapter_sha256=hashes.get(alias))
        for family, revision, alias in (("base", "base-rev", "base"), ("sft", "sft-rev", "sft-opponent"), ("historical", "old-rev", "history"))
    }
    ready = {"step": 0, "policy_revision": "v13-rev", "policy_adapter_sha256": {f"blue-{i}": hashes[f"blue-{i}"] for i in range(4)}}
    config = pulse_config(base_urls=["http://localhost:8001"], ready=ready, snapshots=snapshots, baseline_revision="sft-rev")
    assert config["purpose"] == "actual_initializer"
    assert config["candidate"]["revision"] == "v13-rev"
    assert config["candidate"]["models"] == [f"blue-{i}" for i in range(4)]
    assert config["baseline"]["revision"] == "sft-rev"
    assert len(verify_served_adapters(config, {"http://localhost:8001/v1": registry})["http://localhost:8001/v1"]) == 6
    with pytest.raises(ValueError, match="SFT baseline revision"):
        pulse_config(base_urls=[], ready=ready, snapshots=snapshots, baseline_revision="v13-rev")
    with pytest.raises(ValueError, match="every configured server"):
        verify_served_adapters(config, {})
    (tmp_path / "blue-3/adapter_model.safetensors").write_bytes(b"wrong-policy")
    with pytest.raises(ValueError, match="served adapter hash mismatch"):
        verify_served_adapters(config, {"http://localhost:8001/v1": registry})


def test_initializer_comparison_uses_paired_bundles_and_rejects_missing_cells() -> None:
    initial = [
        {"suite": "handoff_critical", "case_id": f"case-{i}", "independent_id": f"bundle-{i}",
         "opponent_id": "sft", "opponent_revision": "sft-rev", "side": side,
         "condition": condition, "policy_variant": "candidate_rl", "terminal_return": 0.2}
        for i in range(6) for side in ("BLUE", "RED") for condition in ("normal", "dropped")
    ]
    current = [{**row, "terminal_return": row["terminal_return"] + (0.1 if row["condition"] == "normal" else 0)} for row in initial]
    result = initializer_improvement(current, initial)
    endpoint = result["return_changes"]["handoff_critical/normal"]
    assert endpoint["independent_units"] == 6
    assert endpoint["mean_difference"] == pytest.approx(0.1)
    assert result["communication_effect_changes"]["handoff_critical/normal_minus_dropped"]["mean_difference"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="exactly matched"):
        initializer_improvement(current[:-1], initial)
    with pytest.raises(ValueError, match="duplicate"):
        initializer_improvement(current + [current[0]], initial)


def test_budget_rejects_observed_a6000_schedule_and_derives_shared_timeouts() -> None:
    report = staged_evaluation_budget(updates=40, interval=10, games_per_minute=1.3,
                                      update_seconds=60, available_seconds=9 * 3600)
    assert report["fresh_games"] == 672
    assert report["fits"] is False
    assert report["checkpoint_barrier_timeout_seconds"] > 7200
    assert report["pulse_wait_timeout_seconds"] >= report["checkpoint_barrier_timeout_seconds"]
    feasible = staged_evaluation_budget(updates=40, interval=10, games_per_minute=4,
                                        update_seconds=60, available_seconds=9 * 3600)
    assert feasible["fits"] is True
    with pytest.raises(ValueError, match="finite"):
        staged_evaluation_budget(updates=40, interval=10, games_per_minute=float("nan"), update_seconds=60, available_seconds=100)
    assert required_independent_units(bundle_sd=0.1, worthwhile_effect=0.02) == 197


def test_enumerable_credit_audit_distinguishes_shared_rng_from_independent_baseline() -> None:
    for centering in ("none", "replica_mean"):
        independent = audit_paired_credit(coupled=False, centering=centering)
        shared = audit_paired_credit(coupled=True, centering=centering)
        assert independent["expected_implemented_score_update"] == pytest.approx(0.21)
        assert independent["bias"] == pytest.approx(0.0)
        assert shared["expected_implemented_score_update"] == pytest.approx(0.09)
        assert shared["bias"] == pytest.approx(-0.12)


def test_semantic_probe_is_separate_from_frozen_stage_pulse() -> None:
    assert TIER_PLANS["pulse"].critical_conditions == ("normal", "dropped")
    assert TIER_PLANS["semantic_pulse"].critical_conditions == ("normal", "dropped", "target_swapped")
    assert TIER_PLANS["semantic_pulse"].handoff_pairs == 6


def test_terminal_wrapper_preserves_failure_even_with_no_training_progress(tmp_path: Path) -> None:
    child = tmp_path / "fail.py"
    child.write_text("raise SystemExit(3)\n")
    script = Path(__file__).parents[1] / "scripts/supervise_staged_role.py"
    result = subprocess.run([sys.executable, str(script), "--run-dir", str(tmp_path), "--role", "controller", "--", str(child)], check=False)
    assert result.returncode == 3
    record = json.loads((tmp_path / "ABORTED.json").read_text())
    assert record["status"] == "operational_abort"
    assert record["role"] == "controller"
    assert not (tmp_path / "REJECTED.json").exists()


def test_budget_cli_refuses_infeasible_profile_before_launch(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("# synthetic fixture, not a measured production configuration\n")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "version": "staged-operational-profile-v1", "inference_config_sha256": digest,
        "trainer_config_sha256": digest, "topology": "0/1,2,3", "gpu_model": "fixture",
        "game_concurrency": 1, "evidence": ["synthetic regression fixture only"],
        "games_per_minute": 1.3, "update_seconds": 60, "remaining_setup_seconds": 0,
        "checkpoint_seconds": 600, "safety_factor": 1.25,
    }))
    output = tmp_path / "admission.json"
    script = Path(__file__).parents[1] / "scripts/preflight_staged_budget.py"
    result = subprocess.run([sys.executable, str(script), "--profile", str(profile),
                             "--expected-updates", "40", "--interval", "10", "--available-seconds", "32400",
                             "--inference-config", str(config), "--trainer-config", str(config),
                             "--topology", "0/1,2,3", "--gpu-model", "fixture", "--output", str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "schedule exceeds" in result.stderr
    assert json.loads(output.read_text())["fits"] is False


def test_step_zero_control_requires_identical_four_model_rosters() -> None:
    config = {
        "candidate": {"models": ["sft-opponent"] * 4},
        "baseline": {"models": ["sft-opponent"] * 4},
    }
    _validate_step_zero_control_config(config)
    config["candidate"]["models"][3] = "blue-3"
    with pytest.raises(ValueError, match="rosters must be identical"):
        _validate_step_zero_control_config(config)


def test_staged_pulse_matches_receiver_isolated_training_scope() -> None:
    assert _target_swap_scope_args("paired_receiver_target_swap") == ["--receiver-isolated-target-swap"]
    assert _target_swap_scope_args("paired_target_swap") == []


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


def test_selection_design_counts_support_v11_and_v12_schemas() -> None:
    data_root = Path(__file__).parents[1] / "data"
    v11 = json.loads((data_root / "rl_v11" / "progress_eval_design.json").read_text())
    v12 = json.loads((data_root / "rl_v12" / "progress_eval_design.json").read_text())

    assert _selection_design_counts(v11) == (12, 24)
    assert _selection_design_counts(v12) == (36, 36)


def test_v12_selection_expands_all_36_ordinary_cases() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "rl_v12"
    ordinary = _load_manifest(data_dir / "ordinary_hard_development.json")
    rows = _ordinary_cases("selection", ordinary, total_cases=36)

    assert len(rows) == 36
    assert sum(row[3] == "ordinary_legacy" for row in rows) == 18
    assert sum(row[3] == "ordinary_hard" for row in rows) == 18


def test_pulse_and_online_do_not_require_selection_metadata() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "rl_v12"
    ordinary = _load_manifest(data_dir / "ordinary_hard_development.json")

    assert len(_ordinary_cases("pulse", ordinary)) == 12
    assert len(_ordinary_cases("online", ordinary)) == 8


def test_v11_eval_route_preserves_global_development_ids_and_all_frozen_pairs() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "rl_v11"
    design = json.loads((data_dir / "progress_eval_design.json").read_text())
    development = _load_manifest(data_dir / "handoff_development.json")
    frozen = _load_manifest(data_dir / "handoff_frozen_ood.json")

    selection = _handoff_worlds(
        "selection",
        development,
        pair_count=len(design["development_selection"]["handoff_pair_indices"]),
    )
    final = _handoff_worlds(
        "frozen",
        frozen,
        pair_count=design["frozen_final"]["handoff_pairs"],
    )

    assert {row[1] for row in selection} == {
        f"handoff-bundle-{index:03d}" for index in range(96, 108)
    }
    assert len({row[1] for row in final}) == 36


def test_v10_clean_holdout_excludes_every_previously_opened_unit() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "rl_v4"
    lock = _load_lock(data_dir / "v10_clean_holdout_lock.json")
    handoff = _load_manifest(data_dir / "handoff_frozen_ood.json")
    ordinary = _load_manifest(data_dir / "ordinary_hard_frozen_ood.json")
    handoff_rows = _clean_handoff_worlds(lock, handoff)
    ordinary_rows = _clean_ordinary_cases(lock, ordinary)

    assert len({row[1] for row in handoff_rows}) == 22
    assert not {
        "handoff-bundle-004",
        "handoff-bundle-017",
    } & {row[1] for row in handoff_rows}
    assert len({row[1] for row in ordinary_rows if row[3] == "ordinary_hard"}) == 22
    assert not {
        "ordinary-hard-003",
        "ordinary-hard-018",
    } & {row[0] for row in ordinary_rows}
    assert "legacy-seed-3000003" not in {row[1] for row in ordinary_rows}
    assert _expected_rows(ordinary_rows, handoff_rows, 3) == 4260

    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "v10_clean_holdout_4b.json").read_text(encoding="utf-8")
    )
    _validate_bindings(
        lock=lock,
        config=config,
        data_dir=data_dir,
        hard_manifest=ordinary,
        handoff_manifest=handoff,
        ordinary=ordinary_rows,
        handoffs=handoff_rows,
    )
    config["opponents"][0]["revision"] = "mutated"
    with pytest.raises(ValueError, match="opponent revisions"):
        _validate_bindings(
            lock=lock,
            config=config,
            data_dir=data_dir,
            hard_manifest=ordinary,
            handoff_manifest=handoff,
            ordinary=ordinary_rows,
            handoffs=handoff_rows,
        )


def test_v10_mirror_skips_orphan_raw_records_and_snapshots_a_stable_prefix(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps({"evaluation_id": name}) + "\n" for name in ("game-a", "game-b")),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "snapshot.jsonl"
    assert _snapshot_rows(rows_path, snapshot_path, 2) == ["game-a", "game-b"]
    assert len(snapshot_path.read_text(encoding="utf-8").splitlines()) == 2

    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps({"evaluation_id": name, "raw": {"name": name}}) + "\n"
            for name in ("orphan", "game-a", "game-a", "game-b")
        ),
        encoding="utf-8",
    )
    shard_path = tmp_path / "raw.jsonl.gz"
    end = _write_raw_shard(
        raw_path=raw_path,
        output_path=shard_path,
        start_offset=0,
        expected_evaluation_ids=["game-a", "game-b"],
    )
    assert end == raw_path.stat().st_size
    with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
        mirrored = [json.loads(line)["evaluation_id"] for line in handle]
    assert mirrored == ["game-a", "game-b"]


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
                            suite == "handoff_critical" and variant == "candidate_rl" and condition == "normal"
                        ),
                        "sender_target_fact": (suite == "handoff_critical" and condition == "normal"),
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
    assert summary["communication_mechanism"]["candidate_sender_target_fact_rate"]["mean_difference"] == pytest.approx(
        1.0
    )
    assert summary["communication_mechanism"]["rl_specific_capture_lift"]["mean_difference"] == pytest.approx(1.0)
    assert summary["candidate_protocol"]["broadcast_protocol_rate"] == pytest.approx(1.0)
    assert summary["candidate_protocol_denominators"]["broadcast_protocol_rate"] == {
        "defined_rows": 6,
        "undefined_rows": 1,
    }
    metrics = summarize_evaluation(summary)
    assert metrics["eval/rl_specific_communication_lift"] == pytest.approx(0.3)
    assert metrics["eval/overall_gameplay_rl_minus_sft"] == pytest.approx(0.3)


def test_semantic_holdout_uses_itt_and_separates_sft_and_decoy_effects() -> None:
    rows = []
    for unit in ("bundle-a", "bundle-b"):
        for suite, variant, normal, swapped in (
            ("handoff_critical", "candidate_rl", 0.6, 0.1),
            ("handoff_critical", "sft_init", 0.3, 0.2),
            ("handoff_decoy", "candidate_rl", 0.2, 0.1),
        ):
            for condition, value, target_action in (
                ("normal", normal, True),
                ("target_swapped", swapped, False),
            ):
                rows.append(
                    {
                        "independent_id": unit,
                        "suite": suite,
                        "policy_variant": variant,
                        "opponent_id": "sft",
                        "opponent_revision": "sft-revision",
                        "side": "BLUE",
                        "condition": condition,
                        "terminal_return": value,
                        "receiver_target_action": target_action,
                        "target_swap_eligible": condition == "target_swapped",
                        "sender_target_fact": True,
                    }
                )
    summary = summarize_semantic_holdout(rows)
    assert summary["candidate_critical_normal_minus_target_swapped"]["mean_difference"] == pytest.approx(0.5)
    assert summary["rl_specific_semantic_lift"]["mean_difference"] == pytest.approx(0.4)
    assert summary["critical_minus_decoy_semantic_specificity"]["mean_difference"] == pytest.approx(0.4)
    assert summary["receiver_target_action_gap"]["mean_difference"] == pytest.approx(1.0)
    assert summary["claim_checks"]["rl_specific_semantic_interval_positive"]
    assert summary["claim_checks"]["critical_specificity_interval_positive"]


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
            for world in ("left", "right"):
                rows.append(
                    {
                        "kind": kind,
                        "condition": condition,
                        "world": world,
                        "repeat": 0,
                        "terminal_return": value,
                        "receiver_target_action": condition == "normal",
                        "sender_target_fact": condition == "normal",
                        "broadcast_valid": 1.0,
                        "broadcast_grounded": 1.0,
                        "action_valid": 1.0,
                        "target_swap_eligible": condition == "target_swapped",
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
                    terminal_return = lift if kind == "critical" and condition == "normal" else 0.0
                    rows.append(
                        {
                            "pair_index": pair_index,
                            "kind": kind,
                            "condition": condition,
                            "world": world,
                            "repeat": 0,
                            "terminal_return": terminal_return,
                            "receiver_target_action": condition == "normal",
                            "sender_target_fact": condition == "normal",
                            "broadcast_valid": 1.0,
                            "broadcast_grounded": 1.0,
                            "action_valid": 1.0,
                            "target_swap_eligible": condition == "target_swapped",
                        }
                    )
    summary = summarize_pair7(rows, (7, 9))
    assert summary["version"] == "multipair-semantic-communication-eval-v4"
    assert summary["critical"]["normal_minus_dropped_return"] == pytest.approx(0.4)
    assert summary["by_pair"]["7"]["critical"]["normal_minus_dropped_return"] == pytest.approx(0.6)
    assert summary["by_pair"]["9"]["critical"]["normal_minus_dropped_return"] == pytest.approx(0.2)
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


def test_semantic_summary_excludes_ineligible_target_swap_units() -> None:
    rows = []
    for kind in ("critical", "decoy"):
        for world, eligible in (("left", True), ("right", False)):
            for condition in ("normal", "dropped", "sender_shuffled", "target_swapped"):
                rows.append(
                    {
                        "pair_index": 7,
                        "kind": kind,
                        "condition": condition,
                        "world": world,
                        "repeat": 0,
                        "terminal_return": (1.0 if condition == "normal" else 0.0),
                        "receiver_target_action": condition == "normal",
                        "sender_target_fact": condition == "normal",
                        "broadcast_valid": 1.0,
                        "broadcast_grounded": 1.0,
                        "action_valid": 1.0,
                        "target_swap_eligible": (eligible if condition == "target_swapped" else None),
                    }
                )
    summary = summarize_pair7(rows)
    assert summary["critical"]["target_swap_eligible_units"] == 1
    assert summary["critical"]["target_swap_total_units"] == 2
    assert summary["critical"]["target_swap_eligibility_rate"] == 0.5
    assert summary["critical"]["normal_minus_target_swapped_return"] == 1.0
    assert summary["specificity"]["critical_minus_decoy_target_swapped_lift"] == 0.0


def test_live_mirror_collects_compact_eval_but_not_raw_generations(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluations" / "update-10"
    evaluation.mkdir(parents=True)
    for name in ("manifest.json", "rows.jsonl", "summary.json", "raw.jsonl"):
        (evaluation / name).write_text(name)
    files = compact_files(tmp_path)
    assert "evaluations/update-10/summary.json" in files
    assert "evaluations/update-10/rows.jsonl" in files
    assert "evaluations/update-10/raw.jsonl" not in files
