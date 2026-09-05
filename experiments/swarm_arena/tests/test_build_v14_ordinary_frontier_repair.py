from __future__ import annotations

import json
from pathlib import Path

from experiments.swarm_arena.scripts.build_v14_ordinary_frontier_repair import (
    build,
)


def _inputs() -> tuple[dict, dict, dict, dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    assessment_path = root / "results" / "rl_v14_zero_update_rejection" / "ASSESSMENT.json"
    screen_path = root / "data" / "rl_v13" / "ordinary_signal_screen_manifest.json"
    curriculum_path = root / "data" / "rl_v14" / "curriculum.json"
    return (
        json.loads(assessment_path.read_text()),
        json.loads(screen_path.read_text()),
        json.loads(curriculum_path.read_text()),
        {
            "assessment": "a" * 64,
            "original_screen": "b" * 64,
            "curriculum": "c" * 64,
        },
    )


def test_frontier_repair_is_balanced_bounded_and_keeps_gates() -> None:
    assessment, screen, curriculum, hashes = _inputs()
    artifacts = build(
        assessment,
        screen,
        curriculum,
        source_hashes=hashes,
    )
    audit = artifacts["audit.json"]
    repaired_screen = artifacts["ordinary_frontier_screen_manifest.json"]
    pool = artifacts["ordinary_case_pool.json"]

    assert audit["status"] == "cpu_repair_passed_gpu_screen_pending"
    assert audit["thresholds_unchanged"] is True
    assert audit["frozen_or_development_data_used"] is False
    assert audit["optimizer_updates_used"] == 0
    assert repaired_screen["case_count"] == 32
    assert repaired_screen["games"] == 128
    assert set(repaired_screen["policy_case_counts"].values()) == {8}
    assert set(audit["screen_cases_per_policy_family"].values()) == {2}
    assert repaired_screen["thresholds"] == screen["thresholds"]
    assert len({row["case_id"] for row in pool["cases"]}) == pool["case_count"]
    pool_by_id = {row["case_id"]: row for row in pool["cases"]}
    for row in repaired_screen["cases"]:
        source = pool_by_id[row["pool_case_id"]]
        assert row["focused_agent"] == source["focused_agent"]
        assert row["opponent_family"] == source["opponent_family"]
        assert row["seed"] == source["seed"]
        assert row["size"] == source["size"]
        assert row["horizon"] == source["horizon"]
    adaptive_scope = artifacts["curriculum.json"]["adaptive_scope"]
    assert "ordinary case identities" in adaptive_scope["changes"]
    assert "ordinary retention schedule" not in adaptive_scope["does_not_change"]


def test_blue1_blue2_current_repair_uses_only_unseen_frontier_candidates() -> None:
    assessment, screen, curriculum, hashes = _inputs()
    artifacts = build(
        assessment,
        screen,
        curriculum,
        source_hashes=hashes,
    )
    rows = artifacts["ordinary_frontier_screen_manifest.json"]["cases"]
    current = [
        row
        for row in rows
        if row["focused_agent"] in {"blue-1", "blue-2"}
        and row["opponent_family"] == "current"
    ]

    assert len(current) == 4
    assert all(row["provenance"] != "complete_v14_screen" for row in current)
    assert {row["provenance"] for row in current} == {
        "cross_opponent_transfer_from_observed_frontier",
        "deterministic_unseen_neighbor_of_observed_frontier",
    }


def test_frontier_repair_is_byte_deterministic() -> None:
    assessment, screen, curriculum, hashes = _inputs()
    first = build(assessment, screen, curriculum, source_hashes=hashes)
    second = build(assessment, screen, curriculum, source_hashes=hashes)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
