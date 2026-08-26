from __future__ import annotations

import json
import tomllib
from collections import Counter
from pathlib import Path

import pytest
from swarm_ctf_eval.rl_production import (
    CurriculumMix,
    CurriculumStage,
    OpponentPool,
    OpponentSnapshot,
    exact_curriculum_schedule,
    exact_staged_curriculum_schedule,
    load_production_plan,
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
    assert len({row.ordinary_seed for row in schedule if row.kind == "ordinary"}) == 12
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

    assert {scenario_sampling_namespace(row, run_id="canary", step=2) for row in paired} == {"canary:step-2:pair-7"}
    assert (
        scenario_sampling_namespace(
            None,
            run_id="legacy",
            step=3,
            fallback_pair_index=4,
        )
        == "legacy:step-3:pair-4"
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


def test_rollout_runtime_changes_plan_identity_without_changing_legacy_default(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "results"
        / "rl_v4_1_7b_lr_ablation"
        / "variant_a"
        / "plan_original_mix.json"
    )
    raw = json.loads(source.read_text())
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(raw))
    legacy, _ = load_production_plan(legacy_path)

    explicit_default_path = tmp_path / "explicit-default.json"
    explicit_default_path.write_text(
        json.dumps(
            {
                **raw,
                "rollout_runtime": {
                    "shared_return_replicas": 4,
                    "action_prompt_profile": "full",
                    "paired_contrast_centering": "replica_mean",
                },
            }
        )
    )
    explicit_default, _ = load_production_plan(explicit_default_path)
    assert explicit_default.sha256 == legacy.sha256

    compact_path = tmp_path / "compact.json"
    compact_path.write_text(
        json.dumps(
            {
                **raw,
                "rollout_runtime": {
                    "shared_return_replicas": 8,
                    "action_prompt_profile": "focused_handoff_compact",
                },
            }
        )
    )
    compact, _ = load_production_plan(compact_path)
    assert compact.shared_return_replicas == 8
    assert compact.action_prompt_profile == "focused_handoff_compact"
    assert compact.sha256 != legacy.sha256


def test_decoy_counterfactual_challenge_requires_a_decoy_curriculum(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "configs" / "rl_v11_4b_base_plan.json").read_text())
    curriculum = json.loads(
        (root / "data" / "rl_v11" / "staged_curriculum_v11_4b_diverse_receiver_180.json").read_text()
    )
    curriculum["stages"][0]["update_pattern"] = [
        {"ordinary": 2, "critical": 1, "decoy": 1}
    ]
    raw["version"] = "arena-rl-v4-staged-production-plan-v1"
    raw["curriculum_stages"] = curriculum["stages"]
    raw["rollout_runtime"] = {
        "shared_return_replicas": 4,
        "action_prompt_profile": "full",
        "shared_return_baseline": "paired_receiver_target_swap",
        "decoy_shared_return_baseline": "paired_receiver_target_swap_challenge",
    }
    path = tmp_path / "challenge.json"
    path.write_text(json.dumps(raw))
    plan, _ = load_production_plan(path)
    assert plan.decoy_shared_return_baseline == "paired_receiver_target_swap_challenge"

    curriculum["stages"][0]["update_pattern"] = [
        {"ordinary": 1, "critical": 3, "decoy": 0}
    ]
    raw["curriculum_stages"] = curriculum["stages"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="requires scheduled decoy groups"):
        load_production_plan(path)


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
            handoff_focus_roles=("receiver", "sender"),
        ),
        CurriculumStage(
            name="handoff",
            updates=2,
            update_pattern=(tactical, communication),
            ordinary_sizes=(16,),
            ordinary_horizons=(8,),
            handoff_focus_roles=("sender", "receiver"),
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
        roles_by_pair = {
            row.pair_index: row.handoff_focus_role
            for row in block
            if row.kind == "critical"
        }
        assert roles_by_pair == {
            row.pair_index: row.handoff_focus_role
            for row in block
            if row.kind == "decoy"
        }
    assert Counter(row.kind for row in schedule) == {
        "ordinary": 6,
        "critical": 5,
        "decoy": 5,
    }
    assert {row.stage for row in schedule[:8]} == {"warmup"}
    assert {row.stage for row in schedule[8:]} == {"handoff"}
    assert {(row.ordinary_size, row.ordinary_horizon) for row in schedule[8:] if row.kind == "ordinary"} == {(16, 8)}
    assert Counter(
        row.handoff_focus_role for row in schedule if row.handoff_focus_role is not None
    ) == {"sender": 6, "receiver": 4}
    assert schedule == exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=9,
        ordinary_seed_base=90_000,
        shuffle_seed=17,
    )


def test_staged_schedule_can_reserve_decoys_for_evaluation_only() -> None:
    stage = CurriculumStage(
        name="critical-only-learnability",
        updates=2,
        update_pattern=(CurriculumMix(ordinary=0, critical=4, decoy=0),),
        ordinary_sizes=(12,),
        ordinary_horizons=(4,),
        handoff_focus_roles=("receiver",),
        handoff_cases=(
            (7, "left_exposed"),
            (7, "right_exposed"),
            (10, "left_exposed"),
            (10, "right_exposed"),
        ),
        handoff_remaining_turns=2,
    )
    schedule = exact_staged_curriculum_schedule(
        (stage,),
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=90_000,
        shuffle_seed=17,
    )

    assert Counter(row.kind for row in schedule) == {"critical": 8}
    for update in range(2):
        block = schedule[update * 4 : (update + 1) * 4]
        assert {(row.pair_index, row.handoff_world) for row in block} == set(
            stage.handoff_cases
        )


def test_partial_decoy_mix_uses_only_matched_critical_cases() -> None:
    stage = CurriculumStage(
        name="partial-decoy",
        updates=1,
        update_pattern=(CurriculumMix(ordinary=0, critical=3, decoy=1),),
        ordinary_sizes=(12,),
        ordinary_horizons=(4,),
        handoff_focus_roles=("receiver",),
        handoff_cases=((7, "left_exposed"), (7, "right_exposed"), (10, "left_exposed")),
    )
    schedule = exact_staged_curriculum_schedule(
        (stage,),
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=90_000,
        shuffle_seed=17,
    )
    critical = {(row.pair_index, row.handoff_world) for row in schedule if row.kind == "critical"}
    decoy = {(row.pair_index, row.handoff_world) for row in schedule if row.kind == "decoy"}
    assert len(critical) == 3
    assert len(decoy) == 1
    assert decoy < critical


def test_v11_curriculum_scales_diversity_without_role_or_world_imbalance() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "rl_v11"
    curriculum = json.loads(
        (data_dir / "staged_curriculum_v11_4b_diverse_receiver_180.json").read_text()
    )
    manifest = json.loads((data_dir / "handoff_train.json").read_text())
    development = json.loads((data_dir / "handoff_development.json").read_text())
    index = json.loads((data_dir / "index.json").read_text())
    stages = tuple(
        CurriculumStage(
            name=stage["name"],
            updates=stage["updates"],
            update_pattern=tuple(CurriculumMix(**mix) for mix in stage["update_pattern"]),
            ordinary_sizes=tuple(stage["ordinary_sizes"]),
            ordinary_horizons=tuple(stage["ordinary_horizons"]),
            handoff_focus_roles=tuple(stage["handoff_focus_roles"]),
            handoff_cases=tuple(
                (case["pair_index"], case["world"]) for case in stage["handoff_cases"]
            ),
            handoff_remaining_turns=stage["handoff_remaining_turns"],
        )
        for stage in curriculum["stages"]
    )
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=18_000_107,
        shuffle_seed=20_260_824,
    )
    critical = [row for row in schedule if row.kind == "critical"]

    assert len(schedule) == 720
    assert Counter(row.kind for row in schedule) == {"critical": 540, "ordinary": 180}
    assert Counter(row.handoff_world for row in critical) == {
        "left_exposed": 270,
        "right_exposed": 270,
    }
    assert Counter(
        manifest["pairs"][row.pair_index]["critical"]["receiver"]
        for row in critical
    ) == {f"blue-{index}": 135 for index in range(4)}
    assert max(row.pair_index for row in critical) == 95
    assert min(curriculum["online_eval_pair_indices"]) == 96
    assert development["pair_count"] == 24
    assert development["source_pair_start"] == 96
    assert development["pairs"] == manifest["pairs"][96:120]
    assert index["handoff"]["development"]["sha256"] == development["sha256"]
    assert (data_dir / "curriculum.json").read_bytes() == (
        data_dir / "staged_curriculum_v11_4b_diverse_receiver_180.json"
    ).read_bytes()


def test_v12_curriculum_balances_counterfactual_challenges_and_preserves_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "rl_v12"
    curriculum = json.loads((data_dir / "curriculum.json").read_text())
    manifest = json.loads((data_dir / "handoff_train.json").read_text())
    stages = tuple(
        CurriculumStage(
            name=stage["name"],
            updates=stage["updates"],
            update_pattern=tuple(CurriculumMix(**mix) for mix in stage["update_pattern"]),
            ordinary_sizes=tuple(stage["ordinary_sizes"]),
            ordinary_horizons=tuple(stage["ordinary_horizons"]),
            handoff_focus_roles=("receiver",),
            handoff_cases=tuple(
                (case["pair_index"], case["world"]) for case in stage["handoff_cases"]
            ),
            handoff_remaining_turns=stage["handoff_remaining_turns"],
        )
        for stage in curriculum["stages"]
    )
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=curriculum["schedule"]["pair_offset"],
        ordinary_seed_base=curriculum["schedule"]["ordinary_seed_base"],
        shuffle_seed=curriculum["schedule"]["shuffle_seed"],
    )
    assert Counter(row.kind for row in schedule) == {
        "ordinary": 260,
        "critical": 220,
        "decoy": 160,
    }
    for kind, expected in (("critical", 55), ("decoy", 40)):
        assert Counter(
            manifest["pairs"][row.pair_index][kind]["receiver"]
            for row in schedule
            if row.kind == kind
        ) == {f"blue-{index}": expected for index in range(4)}
    assert curriculum["runtime"]["decoy_shared_return_baseline"] == (
        "paired_receiver_target_swap_challenge"
    )
    assert curriculum["runtime"]["paired_contrast_centering"] == "none"
    production, _ = load_production_plan(root / "configs" / "rl_v12_4b_base_plan.json")
    assert production.paired_contrast_centering == "none"
    assert (data_dir / "handoff_frozen_ood.json").read_bytes() == (
        root / "data" / "rl_v11" / "handoff_frozen_ood.json"
    ).read_bytes()
    assert (data_dir / "ordinary_hard_frozen_ood.json").read_bytes() == (
        root / "data" / "rl_v11" / "ordinary_hard_frozen_ood.json"
    ).read_bytes()


def test_joint_curriculum_balances_sender_receiver_focus_and_retains_ordinary_play() -> None:
    root = Path(__file__).resolve().parents[1]
    curriculum = json.loads(
        (root / "data" / "rl_v4" / "staged_curriculum_v3_joint_80.json").read_text()
    )
    stages = tuple(
        CurriculumStage(
            name=stage["name"],
            updates=stage["updates"],
            update_pattern=tuple(CurriculumMix(**mix) for mix in stage["update_pattern"]),
            ordinary_sizes=tuple(stage["ordinary_sizes"]),
            ordinary_horizons=tuple(stage["ordinary_horizons"]),
            handoff_focus_roles=tuple(stage["handoff_focus_roles"]),
        )
        for stage in curriculum["stages"]
    )
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=8_000_000,
        shuffle_seed=17,
    )

    assert len(schedule) == 320
    assert Counter(row.kind for row in schedule) == {
        "ordinary": 60,
        "critical": 130,
        "decoy": 130,
    }
    assert Counter(
        row.handoff_focus_role for row in schedule if row.kind == "critical"
    ) == {"sender": 65, "receiver": 65}
    assert all(row.handoff_focus_role is None for row in schedule if row.kind == "ordinary")
    for update in range(80):
        block = schedule[update * 4 : (update + 1) * 4]
        assert {row.pair_index for row in block if row.kind == "critical"} == {
            row.pair_index for row in block if row.kind == "decoy"
        }


def test_receiver_terminal_curriculum_binds_screened_worlds_and_horizons() -> None:
    root = Path(__file__).resolve().parents[1]
    curriculum = json.loads(
        (root / "data" / "rl_v4" / "staged_curriculum_v4_receiver_terminal_40.json").read_text()
    )
    analysis = json.loads(
        (root / "results" / "rl_v4_passk_screen_1_7b" / "analysis.json").read_text()
    )
    selected_cases = {
        (row["pair_index"], row["world"])
        for row in analysis["selection"]["bands"]["primary_receiver_band"]
    }
    assert all(
        {(case["pair_index"], case["world"]) for case in stage["handoff_cases"]}
        == selected_cases
        for stage in curriculum["stages"]
    )
    stages = tuple(
        CurriculumStage(
            name=stage["name"],
            updates=stage["updates"],
            update_pattern=tuple(CurriculumMix(**mix) for mix in stage["update_pattern"]),
            ordinary_sizes=tuple(stage["ordinary_sizes"]),
            ordinary_horizons=tuple(stage["ordinary_horizons"]),
            handoff_focus_roles=tuple(stage["handoff_focus_roles"]),
            handoff_cases=tuple(
                (case["pair_index"], case["world"]) for case in stage["handoff_cases"]
            ),
            handoff_remaining_turns=stage.get("handoff_remaining_turns"),
        )
        for stage in curriculum["stages"]
    )
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=8_000_000,
        shuffle_seed=17,
    )

    assert len(schedule) == 160
    assert Counter(row.kind for row in schedule) == {
        "ordinary": 40,
        "critical": 60,
        "decoy": 60,
    }
    handoffs = [row for row in schedule if row.kind != "ordinary"]
    assert {row.handoff_focus_role for row in handoffs} == {"receiver"}
    assert all(row.handoff_world is not None for row in handoffs)
    assert {
        row.handoff_remaining_turns for row in schedule[:80] if row.kind != "ordinary"
    } == {2}
    assert {
        row.handoff_remaining_turns for row in schedule[80:] if row.kind != "ordinary"
    } == {None}
    for update in range(40):
        block = schedule[update * 4 : (update + 1) * 4]
        critical = {(row.pair_index, row.handoff_world) for row in block if row.kind == "critical"}
        decoy = {(row.pair_index, row.handoff_world) for row in block if row.kind == "decoy"}
        assert critical == decoy


def test_communication_overfit_curriculum_requires_the_message_to_select_the_world() -> None:
    root = Path(__file__).resolve().parents[1]
    curriculum = json.loads(
        (
            root
            / "data"
            / "rl_v4"
            / "staged_curriculum_v5_communication_overfit_60.json"
        ).read_text()
    )
    handoffs = json.loads((root / "data" / "rl_v4" / "handoff_train.json").read_text())
    stage = curriculum["stages"][0]
    selected_cases = {(case["pair_index"], case["world"]) for case in stage["handoff_cases"]}

    assert curriculum["total_updates"] == 60
    assert selected_cases == {(7, "left_exposed"), (7, "right_exposed")}
    pair = handoffs["pairs"][7]
    assert pair["critical"]["receiver"] == "blue-1"
    assert pair["critical"]["sender"] == "blue-2"
    assert pair["matched_pair_audit"]["critical_receiver_worlds_indistinguishable_without_message"]
    assert pair["matched_pair_audit"]["receiver_action_sets_match_across_worlds"]
    assert pair["matched_pair_audit"]["message_does_not_change_receiver_legal_actions"]

    stages = (
        CurriculumStage(
            name=stage["name"],
            updates=stage["updates"],
            update_pattern=tuple(CurriculumMix(**mix) for mix in stage["update_pattern"]),
            ordinary_sizes=tuple(stage["ordinary_sizes"]),
            ordinary_horizons=tuple(stage["ordinary_horizons"]),
            handoff_focus_roles=tuple(stage["handoff_focus_roles"]),
            handoff_cases=tuple(
                (case["pair_index"], case["world"]) for case in stage["handoff_cases"]
            ),
            handoff_remaining_turns=stage["handoff_remaining_turns"],
        ),
    )
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=8_000_000,
        shuffle_seed=17,
    )

    assert len(schedule) == 240
    assert Counter(row.kind for row in schedule) == {"critical": 120, "decoy": 120}
    for update in range(60):
        block = schedule[update * 4 : (update + 1) * 4]
        for kind in ("critical", "decoy"):
            assert {
                (row.pair_index, row.handoff_world)
                for row in block
                if row.kind == kind
            } == selected_cases
        assert {row.handoff_focus_role for row in block} == {"receiver"}
        assert {row.handoff_remaining_turns for row in block} == {2}


def test_compact_multipair_curriculum_concentrates_only_verified_critical_signal() -> None:
    root = Path(__file__).resolve().parents[1]
    curriculum = json.loads(
        (
            root
            / "data"
            / "rl_v4"
            / "staged_curriculum_v6_compact_multipair_40.json"
        ).read_text()
    )
    handoffs = json.loads((root / "data" / "rl_v4" / "handoff_train.json").read_text())
    stage = curriculum["stages"][0]
    selected_cases = {
        (case["pair_index"], case["world"]) for case in stage["handoff_cases"]
    }

    assert selected_cases == {
        (7, "left_exposed"),
        (7, "right_exposed"),
        (9, "left_exposed"),
        (9, "right_exposed"),
    }
    assert curriculum["runtime"] == {
        "shared_return_replicas": 8,
        "action_prompt_profile": "focused_handoff_compact",
        "online_evaluation_mode": "multipair",
    }
    for pair_index in (7, 9):
        pair = handoffs["pairs"][pair_index]
        audit = pair["matched_pair_audit"]
        assert audit["critical_receiver_worlds_indistinguishable_without_message"]
        assert audit["receiver_action_sets_match_across_worlds"]
        assert audit["message_does_not_change_receiver_legal_actions"]
        assert all(
            certificate["advantage"] > 0
            for certificate in pair["critical"]["certificates"]
        )

    schedule = exact_staged_curriculum_schedule(
        (
            CurriculumStage(
                name=stage["name"],
                updates=stage["updates"],
                update_pattern=tuple(
                    CurriculumMix(**mix) for mix in stage["update_pattern"]
                ),
                ordinary_sizes=tuple(stage["ordinary_sizes"]),
                ordinary_horizons=tuple(stage["ordinary_horizons"]),
                handoff_focus_roles=tuple(stage["handoff_focus_roles"]),
                handoff_cases=tuple(selected_cases),
                handoff_remaining_turns=stage["handoff_remaining_turns"],
            ),
        ),
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=8_000_000,
        shuffle_seed=17,
    )
    assert len(schedule) == 160
    assert Counter(row.kind for row in schedule) == {"critical": 160}
    assert all(row.handoff_focus_role == "receiver" for row in schedule)


def test_staged_run_keeps_training_short_and_preserves_ten_step_checkpoints() -> None:
    root = Path(__file__).resolve().parents[1]
    curriculum = json.loads((root / "data" / "rl_v4" / "staged_curriculum_v1.json").read_text())
    with (root / "configs" / "rl_v4_1_7b_staged.toml").open("rb") as handle:
        trainer = tomllib.load(handle)

    assert curriculum["total_updates"] == 120
    assert {horizon for stage in curriculum["stages"] for horizon in stage["ordinary_horizons"]} == {4, 5}
    assert max(size for stage in curriculum["stages"] for size in stage["ordinary_sizes"]) == 13
    assert trainer["ckpt"]["interval"] == 10
    assert trainer["ckpt"]["keep_interval"] == 10
    assert trainer["wandb"]["offline"]
    assert trainer["atomic_multi_run_updates"]
    parity = trainer["rollout_parity_gate"]
    assert parity == {
        "max_mean_logprob_error": 0.05,
        "probability_tail_threshold": 0.05,
        "max_mean_mismatch_kl": 0.002,
    }
    admission = json.loads((root / "configs" / "async_admission_minimal_v1.json").read_text())
    assert admission == {
        "max_policy_lag": 1,
        "max_mean_abs_log_ratio": 0.05,
        "max_mean_mismatch_kl": 0.002,
        "max_p99_abs_log_ratio": None,
        "max_symmetric_importance_ratio": None,
        "max_p99_probability_error": None,
        "probability_tail_threshold": 0.05,
        "max_probability_tail_fraction": None,
    }
