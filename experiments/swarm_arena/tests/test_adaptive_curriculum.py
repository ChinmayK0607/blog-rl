from __future__ import annotations

import json

from experiments.swarm_arena.swarm_ctf_eval.adaptive_curriculum import (
    adapt_stage_assignments,
    handoff_case_key,
    select_ordinary_stage_cases,
    select_handoff_cases,
    summarize_training_progress,
)
from experiments.swarm_arena.swarm_ctf_eval.rl_production import (
    AdaptiveCurriculumConfig,
    OpponentSnapshot,
    OrdinaryCase,
    ScenarioAssignment,
)


def _replica(effect_field: str, effect: float, *, target: str = "V1") -> dict[str, object]:
    return {
        "return": effect,
        "semantic_effect": effect if effect_field == "semantic_effect" else None,
        "challenge_effect": effect if effect_field == "challenge_effect" else None,
        "advantages": {"blue-0": effect},
        "focused_action": {"type": "CAPTURE", "target": target},
    }


def test_training_frontier_analysis_classifies_mastered_stalled_and_frontier() -> None:
    config = AdaptiveCurriculumConfig()
    progress = [
        {
            "step": 0,
            "groups": [
                {
                    "scenario": {
                        "kind": "critical",
                        "pair_index": 1,
                        "world": "left_exposed",
                        "receiver": "blue-0",
                        "focused_agent": "blue-0",
                        "active_target": "V1",
                    },
                    "replicas": [_replica("semantic_effect", 0.2) for _ in range(4)],
                },
                {
                    "scenario": {
                        "kind": "decoy",
                        "pair_index": 2,
                        "world": "right_exposed",
                        "receiver": "blue-0",
                        "focused_agent": "blue-0",
                    },
                    "replicas": [_replica("challenge_effect", 0.0) for _ in range(4)],
                },
                {
                    "scenario": {
                        "source": "ordinary",
                        "size": 16,
                        "scheduled_horizon": 8,
                        "focused_agent": "blue-0",
                    },
                    "replicas": [
                        _replica("ordinary", value) for value in (0.2, -0.2, 0.1, -0.1)
                    ],
                },
            ],
        }
    ]

    analysis = summarize_training_progress(progress, config=config)

    assert analysis["handoff_cases"]["critical:1:left_exposed"]["classification"] == "mastered"
    assert analysis["handoff_cases"]["decoy:2:right_exposed"]["classification"] == "stalled"
    assert analysis["ordinary_buckets"]["ordinary:16:8"]["classification"] == "frontier"
    assert analysis["scope"] == "training_rollouts_only_no_development_or_frozen_data"


def test_adaptive_selector_retains_small_mastered_and_stalled_anchors() -> None:
    config = AdaptiveCurriculumConfig()
    pool = [(index, "left_exposed") for index in range(10)]
    stats = {}
    for index, case in enumerate(pool):
        category = "mastered" if index == 0 else "stalled" if index == 1 else "frontier"
        stats[handoff_case_key("critical", *case)] = {"classification": category}
    analysis = {"handoff_cases": stats}

    selected = select_handoff_cases(
        kind="critical",
        receiver_sequence=["blue-0"] * 10,
        pool_by_receiver={"blue-0": pool},
        analysis=analysis,
        config=config,
        selection_namespace="test",
    )
    categories = [
        stats[handoff_case_key("critical", *case)]["classification"] for case in selected
    ]

    assert categories.count("frontier") == 8
    assert categories.count("mastered") == 1
    assert categories.count("stalled") == 1


def test_adaptive_stage_preserves_decoy_matching_and_ordinary_assignment() -> None:
    config = AdaptiveCurriculumConfig()
    schedule = (
        ScenarioAssignment(0, "ordinary", None, 100, stage="next", ordinary_size=16, ordinary_horizon=8),
        ScenarioAssignment(1, "critical", 1, None, stage="next", handoff_world="left_exposed"),
        ScenarioAssignment(2, "decoy", 1, None, stage="next", handoff_world="left_exposed"),
        ScenarioAssignment(3, "critical", 2, None, stage="next", handoff_world="right_exposed"),
    )
    receiver_by_case = {
        (1, "left_exposed"): "blue-0",
        (2, "right_exposed"): "blue-0",
    }
    analysis = {
        "sha256": "a" * 64,
        "handoff_cases": {
            "critical:1:left_exposed": {"classification": "stalled"},
            "critical:2:right_exposed": {"classification": "frontier"},
            "decoy:1:left_exposed": {"classification": "stalled"},
            "decoy:2:right_exposed": {"classification": "frontier"},
        },
    }

    adapted, selection = adapt_stage_assignments(
        schedule,
        stage_name="next",
        analysis=analysis,
        receiver_by_case=receiver_by_case,
        config=config,
    )

    critical_cases = {
        (row.pair_index, row.handoff_world) for row in adapted if row.kind == "critical"
    }
    decoy_cases = {
        (row.pair_index, row.handoff_world) for row in adapted if row.kind == "decoy"
    }
    assert decoy_cases <= critical_cases
    assert adapted[0] == schedule[0]
    assert selection["ordinary_schedule_changed"] is False
    assert selection["frozen_or_development_data_used"] is False
    assert json.loads(json.dumps(selection)) == selection


def test_ordinary_case_analysis_uses_complete_training_groups_only() -> None:
    config = AdaptiveCurriculumConfig()
    progress = [
        {
            "step": 0,
            "groups": [
                {
                    "scenario": {
                        "source": "ordinary",
                        "ordinary_case_id": "case-variable",
                        "seed": 10,
                        "size": 16,
                        "scheduled_horizon": 8,
                        "focused_agent": "blue-0",
                        "opponent": {"family": "base"},
                    },
                    "replicas": [
                        _replica("ordinary", value)
                        for value in (0.2, -0.2, 0.1, -0.1)
                    ],
                },
                {
                    "scenario": {
                        "source": "ordinary",
                        "ordinary_case_id": "case-mastered",
                        "seed": 11,
                        "size": 18,
                        "scheduled_horizon": 10,
                        "focused_agent": "blue-0",
                        "opponent": {"family": "sft"},
                    },
                    "replicas": [_replica("ordinary", 0.2) for _ in range(4)],
                },
            ],
        }
    ]

    analysis = summarize_training_progress(progress, config=config)

    assert analysis["ordinary_cases"]["case-variable"]["classification"] == "frontier"
    assert analysis["ordinary_cases"]["case-mastered"]["classification"] == "mastered"
    assert analysis["scope"] == "training_rollouts_only_no_development_or_frozen_data"


def test_ordinary_selector_preserves_policy_and_opponent_binding() -> None:
    config = AdaptiveCurriculumConfig()
    families = ("base", "sft", "historical", "current")
    schedule = tuple(
        ScenarioAssignment(
            ordinal,
            "ordinary",
            None,
            1000 + ordinal,
            stage="next",
            ordinary_size=16,
            ordinary_horizon=8,
        )
        for ordinal in range(16)
    )
    opponents = tuple(
        OpponentSnapshot(
            opponent_id=f"opponent-{ordinal}",
            family=families[ordinal // 4],  # type: ignore[arg-type]
            model_name=f"model-{ordinal}",
            revision=f"revision-{ordinal}",
            adapter_sha256=None if ordinal < 4 else f"{ordinal:064x}",
            update_index=0,
        )
        for ordinal in range(16)
    )
    pool = tuple(
        OrdinaryCase(
            case_id=f"case-{policy}-{family}",
            focused_agent=f"blue-{policy}",  # type: ignore[arg-type]
            opponent_family=family,  # type: ignore[arg-type]
            seed=2000 + policy * 10 + family_index,
            size=16,
            horizon=8,
            initial_classification="unseen",
            provenance="test",
            source_case_id="source",
        )
        for policy in range(4)
        for family_index, family in enumerate(families)
    )

    selected, selection = select_ordinary_stage_cases(
        schedule,
        stage_name="next",
        opponent_schedule=opponents,
        pool=pool,
        analysis={"sha256": "a" * 64, "ordinary_cases": {}},
        config=config,
        selection_namespace="test",
    )

    assert len(selected) == 16
    for ordinal, case in selected.items():
        assert case.focused_agent == f"blue-{ordinal % 4}"
        assert case.opponent_family == opponents[ordinal].family
    assert selection["frozen_or_development_data_used"] is False
    assert json.loads(json.dumps(selection)) == selection
