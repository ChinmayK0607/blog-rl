from __future__ import annotations

from collections import Counter

import pytest
from swarm_ctf_eval.rl_production import (
    CurriculumMix,
    OpponentPool,
    OpponentSnapshot,
    exact_curriculum_schedule,
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
