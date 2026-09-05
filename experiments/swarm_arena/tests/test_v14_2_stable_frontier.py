from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from experiments.swarm_arena.scripts.assess_v14_2_stable_frontier_screen import (
    assess,
)
from experiments.swarm_arena.scripts.build_v14_2_stable_frontier import build
from experiments.swarm_arena.scripts.finalize_v14_2_stable_frontier import (
    canonical_sha256,
    finalize,
)
from experiments.swarm_arena.swarm_ctf_eval.adaptive_curriculum import (
    select_ordinary_stage_cases,
)
from experiments.swarm_arena.swarm_ctf_eval.rl_production import (
    AdaptiveCurriculumConfig,
    OpponentSnapshot,
    OrdinaryCase,
    ScenarioAssignment,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict, dict, dict, dict, dict[str, str]]:
    root = _root()
    return (
        json.loads(
            (root / "results" / "rl_v14_zero_update_rejection" / "ASSESSMENT.json")
            .read_text()
        ),
        json.loads(
            (
                root
                / "results"
                / "rl_v14_1_zero_update_rejection"
                / "PARTIAL_SCREEN_ASSESSMENT.json"
            ).read_text()
        ),
        json.loads((root / "data" / "rl_v14_1" / "ordinary_case_pool.json").read_text()),
        json.loads((root / "data" / "rl_v14_1" / "curriculum.json").read_text()),
        {
            "v14_assessment": "a" * 64,
            "v14_1_partial": "b" * 64,
            "v14_1_pool": "c" * 64,
            "v14_1_curriculum": "d" * 64,
        },
    )


def _artifacts() -> dict[str, dict]:
    assessment, partial, pool, curriculum, hashes = _inputs()
    return build(
        assessment,
        partial,
        pool,
        curriculum,
        source_hashes=hashes,
    )


def _diagnostic(case: dict, *, variable: bool) -> list[dict]:
    focused = case["focused_agent"]
    returns = [0.0, 0.2, 0.0, 0.2] if variable else [0.0] * 4
    advantages = [-0.1, 0.1, -0.1, 0.1] if variable else [0.0] * 4
    return [
        {
            "groups": [
                {
                    "scenario": {
                        "source": "ordinary",
                        "seed": case["seed"],
                        "size": case["size"],
                        "scheduled_horizon": case["horizon"],
                        "focused_agent": focused,
                        "opponent": {"family": case["opponent_family"]},
                    },
                    "replicas": [
                        {
                            "return": value,
                            "advantages": {focused: advantage},
                            "focused_action": {
                                "type": "PROBE",
                                "target": f"V{index % 2}" if variable else "V0",
                            },
                        }
                        for index, (value, advantage) in enumerate(
                            zip(returns, advantages, strict=True)
                        )
                    ],
                }
            ]
        }
    ]


def _passing_assessment(manifest: dict) -> dict:
    variable_ids = set()
    by_policy: dict[str, list[dict]] = {}
    for case in manifest["cases"]:
        if case["blocking"]:
            by_policy.setdefault(case["focused_agent"], []).append(case)
    for cases in by_policy.values():
        families = set()
        for case in cases:
            if case["opponent_family"] not in families:
                variable_ids.add(case["case_id"])
                families.add(case["opponent_family"])
            if len(families) == 2:
                break
    diagnostics = {
        case["case_id"]: _diagnostic(
            case,
            variable=case["case_id"] in variable_ids,
        )
        for case in manifest["cases"]
    }
    return assess(manifest, diagnostics)


def test_builder_records_instability_and_creates_bounded_disjoint_pilot() -> None:
    assessment, partial, _pool, curriculum, _hashes = _inputs()
    artifacts = _artifacts()
    diagnosis = artifacts["diagnosis.json"]
    manifest = artifacts["pilot_screen_manifest.json"]

    assert diagnosis["observed_frontier_retest_count"] == 7
    assert diagnosis["observed_frontier_retained_count"] == 4
    assert math.isclose(diagnosis["observed_frontier_retention_rate"], 4 / 7)
    assert diagnosis["credit_estimator_decision"]["change"] is False
    assert manifest["case_count"] == 24
    assert manifest["games"] == 96
    assert set(manifest["policy_case_counts"].values()) == {6}
    assert manifest["role_case_counts"] == {
        "current_probe": 8,
        "frontier_candidate": 16,
    }
    assert all(
        row["provenance"] == "cross_opponent_transfer_from_observed_frontier"
        and row["blocking"] is False
        for row in manifest["cases"]
        if row["admission_role"] == "current_probe"
    )
    assert artifacts["curriculum.json"]["credit_assignment"] == curriculum[
        "credit_assignment"
    ]
    assert artifacts["curriculum.json"]["stage_gate_policy"] == curriculum[
        "stage_gate_policy"
    ]
    assert partial["optimizer_updates"] == 0
    assert assessment["protocol_admission_rate"] == 1.0


def test_builder_is_byte_deterministic() -> None:
    first = _artifacts()
    second = _artifacts()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_assessor_passes_with_two_variable_families_and_flat_current_probes() -> None:
    manifest = _artifacts()["pilot_screen_manifest.json"]
    result = _passing_assessment(manifest)

    assert result["status"] == "passed"
    assert result["failed_gates"] == []
    assert result["pilot_trajectories_used_for_optimization"] is False
    assert all(row["variable_groups"] == 0 for row in result["current_probe_metrics"].values())
    assert all(row["variable_groups"] == 2 for row in result["policy_metrics"].values())
    assert all(len(row["variable_families"]) == 2 for row in result["policy_metrics"].values())


def test_assessor_fails_policy_with_only_one_variable_family() -> None:
    manifest = _artifacts()["pilot_screen_manifest.json"]
    passing = _passing_assessment(manifest)
    variable_ids = {
        row["case_id"]
        for row in passing["cases"]
        if row["focused_agent"] != "blue-2" and row["return_range"] > 0
    }
    blue2 = [
        row for row in manifest["cases"] if row["focused_agent"] == "blue-2" and row["blocking"]
    ]
    variable_ids.add(blue2[0]["case_id"])
    diagnostics = {
        case["case_id"]: _diagnostic(case, variable=case["case_id"] in variable_ids)
        for case in manifest["cases"]
    }
    result = assess(manifest, diagnostics)

    assert result["status"] == "failed"
    assert "blue-2/variable_groups" in result["failed_gates"]
    assert "blue-2/variable_families" in result["failed_gates"]


def test_finalizer_binds_only_pilot_cases_and_discards_pilot_trajectories() -> None:
    artifacts = _artifacts()
    manifest = artifacts["pilot_screen_manifest.json"]
    pilot = _passing_assessment(manifest)
    _assessment, _partial, source_pool, _curriculum, _hashes = _inputs()
    finalized = finalize(
        manifest,
        pilot,
        source_pool,
        artifacts["curriculum.json"],
        source_hashes={
            "pilot_manifest": "1" * 64,
            "pilot_assessment": "2" * 64,
            "source_pool": "3" * 64,
            "curriculum_template": "4" * 64,
        },
    )
    pool = finalized["ordinary_case_pool.json"]
    curriculum = finalized["curriculum.json"]
    audit = finalized["finalization_audit.json"]

    assert pool["case_count"] == 24
    assert len(pool["cell_counts"]) == 16
    assert pool["classification_counts"] == {"frontier": 8, "stalled": 16}
    assert len(pool["pilot_case_metrics"]) == 24
    assert set(pool["cases"][0]) == {
        "case_id",
        "focused_agent",
        "horizon",
        "initial_classification",
        "opponent_family",
        "provenance",
        "seed",
        "size",
        "source_case_id",
    }
    for row in pool["cases"]:
        OrdinaryCase(**row).validate()
    assert curriculum["ordinary_case_pool"] == pool
    assert curriculum["ordinary_frontier_stability_repair"][
        "pilot_assessment_sha256"
    ] == pilot["sha256"]
    assert audit["optimizer_updates_authorized"] == 40
    assert audit["pilot_trajectories_used_for_optimization"] is False


def test_finalized_pool_drives_every_policy_opponent_cell() -> None:
    artifacts = _artifacts()
    manifest = artifacts["pilot_screen_manifest.json"]
    pilot = _passing_assessment(manifest)
    _assessment, _partial, source_pool, _curriculum, _hashes = _inputs()
    finalized = finalize(
        manifest,
        pilot,
        source_pool,
        artifacts["curriculum.json"],
        source_hashes={"synthetic": "5" * 64},
    )
    curriculum = finalized["curriculum.json"]
    config_values = dict(curriculum["adaptive_curriculum"])
    config_values["candidate_cases"] = tuple(config_values["candidate_cases"])
    config = AdaptiveCurriculumConfig(**config_values)
    pool = tuple(OrdinaryCase(**row) for row in curriculum["ordinary_case_pool"]["cases"])
    families = ("base", "sft", "historical", "current")
    schedule = tuple(
        ScenarioAssignment(
            ordinal=ordinal,
            kind="ordinary",
            pair_index=None,
            ordinary_seed=1000 + ordinal,
            stage="pilot-bound-stage",
            ordinary_size=16,
            ordinary_horizon=8,
        )
        for ordinal in range(16)
    )
    opponents = tuple(
        OpponentSnapshot(
            opponent_id=f"{family}-{ordinal}",
            family=family,
            model_name=f"model-{family}-{ordinal}",
            revision="6" * 64,
            adapter_sha256=None if family == "base" else "7" * 64,
            update_index=0,
        )
        for ordinal in range(16)
        for family in (families[ordinal // 4],)
    )
    selected, selection = select_ordinary_stage_cases(
        schedule,
        stage_name="pilot-bound-stage",
        opponent_schedule=opponents,
        pool=pool,
        analysis={"sha256": "8" * 64, "ordinary_cases": {}},
        config=config,
        selection_namespace="test-v14.2",
    )

    assert len(selected) == 16
    assert selection["frozen_or_development_data_used"] is False
    for ordinal, case in selected.items():
        assert case.focused_agent == f"blue-{ordinal % 4}"
        assert case.opponent_family == families[ordinal // 4]


def test_finalizer_rejects_failed_pilot() -> None:
    artifacts = _artifacts()
    manifest = artifacts["pilot_screen_manifest.json"]
    pilot = _passing_assessment(manifest)
    failed = copy.deepcopy(pilot)
    failed["status"] = "failed"
    failed["sha256"] = canonical_sha256(
        {key: value for key, value in failed.items() if key != "sha256"}
    )
    _assessment, _partial, source_pool, _curriculum, _hashes = _inputs()

    try:
        finalize(
            manifest,
            failed,
            source_pool,
            artifacts["curriculum.json"],
            source_hashes={},
        )
    except ValueError as error:
        assert "passed pilot" in str(error)
    else:
        raise AssertionError("failed V14.2 pilot unexpectedly authorized training")
