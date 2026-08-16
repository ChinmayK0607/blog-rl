from __future__ import annotations

from collections import Counter

import pytest
from swarm_ctf_eval.rl_production import (
    CurriculumStage,
    CurriculumMix,
    OpponentPool,
    OpponentSnapshot,
    exact_curriculum_schedule,
    exact_staged_curriculum_schedule,
    scenario_sampling_namespace,
)


def _opponent(opponent_id: str, family: str, marker: str) -> OpponentSnapshot:
    return OpponentSnapshot(
        opponent_id=opponent_id,
        family=family,  # type: ignore[arg-type]
        model_name=opponent_id,
        revision=f"revision-{opponent_id}",
        adapter_sha256=None if family == "base" else marker * 64,
        update_index=0,
    )


def test_exact_curriculum_schedule_preserves_mix_and_critical_decoy_pairs() -> None:
    schedule = exact_curriculum_schedule(
        CurriculumMix(50, 25, 25),
        total_groups=24,
        pair_offset=11,
        ordinary_seed_base=8_000_000,
        shuffle_seed=17,
    )

    assert Counter(row.kind for row in schedule) == {
        "ordinary": 12,
        "critical": 6,
        "decoy": 6,
    }
    assert {row.pair_index for row in schedule if row.kind == "critical"} == {
        row.pair_index for row in schedule if row.kind == "decoy"
    }
    assert len(
        {row.ordinary_seed for row in schedule if row.kind == "ordinary"}
    ) == 12
    assert schedule == exact_curriculum_schedule(
        CurriculumMix(50, 25, 25),
        total_groups=24,
        pair_offset=11,
        ordinary_seed_base=8_000_000,
        shuffle_seed=17,
    )


def test_stage_two_mix_requires_complete_twenty_group_blocks() -> None:
    with pytest.raises(ValueError, match="multiple of curriculum block size 20"):
        exact_curriculum_schedule(
            CurriculumMix(70, 15, 15),
            total_groups=4,
            pair_offset=0,
            ordinary_seed_base=1,
            shuffle_seed=2,
        )


def test_production_ordinary_scenario_never_uses_legacy_pair_fallback() -> None:
    ordinary = exact_curriculum_schedule(
        CurriculumMix(50, 25, 25),
        total_groups=4,
        pair_offset=0,
        ordinary_seed_base=8_000_000,
        shuffle_seed=20_260_815,
    )
    ordinary_assignment = next(row for row in ordinary if row.kind == "ordinary")

    assert (
        scenario_sampling_namespace(
            ordinary_assignment,
            run_id="canary",
            step=0,
            fallback_pair_index=999,
        )
        is None
    )


def test_sampling_namespace_matches_critical_decoy_pair_and_legacy_fallback() -> None:
    schedule = exact_curriculum_schedule(
        CurriculumMix(50, 25, 25),
        total_groups=4,
        pair_offset=7,
        ordinary_seed_base=8_000_000,
        shuffle_seed=20_260_815,
    )
    paired = [row for row in schedule if row.kind in {"critical", "decoy"}]

    assert {
        scenario_sampling_namespace(row, run_id="canary", step=2) for row in paired
    } == {"canary:step-2:pair-7"}
    assert scenario_sampling_namespace(
        None,
        run_id="legacy",
        step=3,
        fallback_pair_index=4,
    ) == "legacy:step-3:pair-4"


def test_opponent_pool_rotates_all_model_families_exactly() -> None:
    pool = OpponentPool(
        snapshots=(
            _opponent("base", "base", "a"),
            _opponent("sft", "sft", "b"),
            _opponent("historical", "historical", "c"),
            _opponent("current", "current", "d"),
        ),
        rotation_seed=9,
    )
    schedule = pool.schedule(12)

    assert Counter(row.family for row in schedule) == {
        "base": 3,
        "sft": 3,
        "historical": 3,
        "current": 3,
    }
    assert len(pool.sha256) == 64


def test_staged_schedule_is_exact_paired_and_deterministic_per_update() -> None:
    tactical = CurriculumMix(ordinary=2, critical=1, decoy=1)
    communication = CurriculumMix(ordinary=0, critical=2, decoy=2)
    stages = (
        CurriculumStage(
            name="warmup",
            updates=2,
            update_pattern=(tactical,),
            ordinary_sizes=(12, 13),
            ordinary_horizons=(4, 5),
        ),
        CurriculumStage(
            name="handoff",
            updates=2,
            update_pattern=(tactical, communication),
            ordinary_sizes=(16,),
            ordinary_horizons=(8,),
        ),
    )
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=9,
        ordinary_seed_base=90_000,
        shuffle_seed=17,
    )

    assert len(schedule) == 16
    for update in range(4):
        block = schedule[update * 4 : (update + 1) * 4]
        assert {row.pair_index for row in block if row.kind == "critical"} == {
            row.pair_index for row in block if row.kind == "decoy"
        }
    assert Counter(row.kind for row in schedule) == {
        "ordinary": 6,
        "critical": 5,
        "decoy": 5,
    }
    assert {row.stage for row in schedule[:8]} == {"warmup"}
    assert {row.stage for row in schedule[8:]} == {"handoff"}
    assert {
        (row.ordinary_size, row.ordinary_horizon)
        for row in schedule[8:]
        if row.kind == "ordinary"
    } == {(16, 8)}
    assert schedule == exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=9,
        ordinary_seed_base=90_000,
        shuffle_seed=17,
    )
