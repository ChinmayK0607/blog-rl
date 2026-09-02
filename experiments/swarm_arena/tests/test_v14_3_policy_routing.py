from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from experiments.swarm_arena.scripts.finalize_v14_3_policy_routing import (
    canonical_sha256,
    finalize,
)
from experiments.swarm_arena.scripts.freeze_v14_3_cpu_bundle import (
    build_bundle,
    load_hashed,
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

ROOT = Path(__file__).parents[1]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict[str, object], ...]:
    return (
        _json(ROOT / "data" / "rl_v14_2" / "pilot_screen_manifest.json"),
        _json(ROOT / "results" / "rl_v14_2_zero_update_rejection" / "ASSESSMENT.json"),
        _json(ROOT / "data" / "rl_v14_1" / "ordinary_case_pool.json"),
        _json(ROOT / "data" / "rl_v14_2" / "curriculum.json"),
    )


def test_failed_pilot_becomes_policy_routes_without_reusing_trajectories() -> None:
    manifest, assessment, source_pool, curriculum = _inputs()
    artifacts = finalize(
        manifest,
        assessment,
        source_pool,
        curriculum,
        source_hashes={"test": "a" * 64},
    )
    pool = artifacts["ordinary_case_pool.json"]
    routed = artifacts["curriculum.json"]
    audit = artifacts["finalization_audit.json"]

    assert audit["policy_modes"] == {
        "blue-0": "expand",
        "blue-1": "consolidate",
        "blue-2": "consolidate",
        "blue-3": "discover",
    }
    assert pool["case_count"] == 128
    assert len(pool["cell_counts"]) == 16
    assert pool["classification_counts"] == {
        "frontier": 8,
        "mastered": 6,
        "stalled": 10,
        "unseen": 104,
    }
    assert routed["ordinary_policy_routing"]["fresh_rollouts_only"] is True
    assert audit["pilot_trajectories_used_for_optimization"] is False
    assert audit["stage_gates_changed"] is False


def test_policy_router_consolidates_expands_and_discovers_independently() -> None:
    modes = (
        "blue-0:expand",
        "blue-1:consolidate",
        "blue-2:consolidate",
        "blue-3:discover",
    )
    config = AdaptiveCurriculumConfig(
        mastered_anchor_fraction=0.0,
        stalled_anchor_fraction=0.0,
        policy_modes=modes,
        expand_frontier_fraction=0.5,
        discovery_frontier_fraction=0.25,
    )
    schedule = tuple(
        ScenarioAssignment(
            ordinal=ordinal,
            kind="ordinary",
            pair_index=None,
            ordinary_seed=1000 + ordinal,
            stage="next",
            ordinary_size=16,
            ordinary_horizon=8,
        )
        for ordinal in range(16)
    )
    opponents = tuple(
        OpponentSnapshot(
            opponent_id=f"base-{ordinal}",
            family="base",
            model_name="base",
            revision="b" * 64,
            adapter_sha256=None,
            update_index=0,
        )
        for ordinal in range(16)
    )
    pool = []
    for policy in range(4):
        categories = (
            ("frontier", "unseen") if policy < 3 else ("unseen", "stalled")
        )
        for category in categories:
            for index in range(4):
                pool.append(
                    OrdinaryCase(
                        case_id=f"blue-{policy}-{category}-{index}",
                        focused_agent=f"blue-{policy}",
                        opponent_family="base",
                        seed=2000 + policy * 100 + index,
                        size=16,
                        horizon=8,
                        initial_classification=category,
                        provenance="test",
                        source_case_id="test-source",
                    )
                )

    selected, selection = select_ordinary_stage_cases(
        schedule,
        stage_name="next",
        opponent_schedule=opponents,
        pool=tuple(pool),
        analysis={"sha256": "c" * 64, "ordinary_cases": {}},
        config=config,
        selection_namespace="policy-routing-test",
    )
    categories = Counter(
        (case.focused_agent, case.initial_classification)
        for case in selected.values()
    )

    assert categories[("blue-0", "frontier")] == 2
    assert categories[("blue-0", "unseen")] == 2
    assert categories[("blue-1", "frontier")] == 4
    assert categories[("blue-2", "frontier")] == 4
    assert categories[("blue-3", "unseen")] == 4
    assert selection["policy_modes"] == dict(value.split(":") for value in modes)


def test_policy_routing_still_rejects_protocol_failure() -> None:
    manifest, assessment, source_pool, curriculum = _inputs()
    invalid = copy.deepcopy(assessment)
    invalid["protocol_admission_rate"] = 0.99
    invalid["sha256"] = canonical_sha256(
        {key: value for key, value in invalid.items() if key != "sha256"}
    )

    try:
        finalize(
            manifest,
            invalid,
            source_pool,
            curriculum,
            source_hashes={},
        )
    except ValueError as error:
        assert "protocol admission" in str(error)
    else:
        raise AssertionError("policy routing bypassed protocol admission")


def test_cpu_bundle_reproduces_exactly() -> None:
    data = ROOT / "data"
    results = ROOT / "results" / "rl_v14_2_zero_update_rejection"
    artifacts = {
        "assessment": results / "ASSESSMENT.json",
        "audit": data / "rl_v14_3" / "finalization_audit.json",
        "curriculum": data / "rl_v14_3" / "curriculum.json",
        "ordinary_pool": data / "rl_v14_3" / "ordinary_case_pool.json",
        "stage_gates": data / "rl_v14" / "stage_gates.json",
    }
    code_paths = tuple(
        Path("experiments/swarm_arena") / path
        for path in (
            "scripts/finalize_v14_3_policy_routing.py",
            "scripts/freeze_v14_3_cpu_bundle.py",
            "swarm_ctf_eval/adaptive_curriculum.py",
            "swarm_ctf_eval/rl_production.py",
            "scripts/run_live_rl.py",
            "scripts/build_staged_rl_plan.py",
            "scripts/preflight_staged_rl.py",
            "scripts/run_staged_pulses.py",
            "scripts/launch_staged_rl.sh",
        )
    )
    actual = build_bundle(
        base_bundle=load_hashed(data / "rl_v14_2" / "cpu_bundle.json"),
        assessment=load_hashed(artifacts["assessment"]),
        audit=load_hashed(artifacts["audit"]),
        curriculum=load_hashed(artifacts["curriculum"]),
        ordinary_pool=load_hashed(artifacts["ordinary_pool"]),
        stage_gates=load_hashed(artifacts["stage_gates"]),
        artifact_paths=artifacts,
        code_paths=code_paths,
    )

    assert actual == load_hashed(data / "rl_v14_3" / "cpu_bundle.json")
