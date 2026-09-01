from __future__ import annotations

import json
from pathlib import Path

from experiments.swarm_arena.scripts.freeze_v14_1_cpu_bundle import build_bundle


def _hashed(path: Path) -> dict:
    return json.loads(path.read_text())


def test_v14_1_bundle_is_frozen_and_zero_update() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    artifacts = {
        "audit": data / "rl_v14_1" / "audit.json",
        "curriculum": data / "rl_v14_1" / "curriculum.json",
        "ordinary_case_pool": data / "rl_v14_1" / "ordinary_case_pool.json",
        "ordinary_frontier_screen": (
            data / "rl_v14_1" / "ordinary_frontier_screen_manifest.json"
        ),
        "stage_gates": data / "rl_v14" / "stage_gates.json",
        "trainer_config": root / "configs" / "rl_v14_4b_grounded_40.toml",
        "base_plan": root / "configs" / "rl_v12_4b_base_plan.json",
        "admission_limits": root / "configs" / "async_admission_minimal_v1.json",
        "handoff_manifest": data / "rl_v4" / "handoff_train.json",
    }
    bundle = build_bundle(
        base_bundle=_hashed(data / "rl_v14" / "cpu_bundle.json"),
        audit=_hashed(artifacts["audit"]),
        curriculum=_hashed(artifacts["curriculum"]),
        pool=_hashed(artifacts["ordinary_case_pool"]),
        repair_screen=_hashed(artifacts["ordinary_frontier_screen"]),
        original_screen=_hashed(data / "rl_v13" / "ordinary_signal_screen_manifest.json"),
        stage_gates=_hashed(artifacts["stage_gates"]),
        artifact_paths=artifacts,
        code_paths=(root / "scripts" / "build_v14_ordinary_frontier_repair.py",),
    )

    assert bundle["status"] == "cpu_frozen_source_publication_and_gpu_screen_pending"
    assert bundle["optimizer_updates_authorized"] == 0
    assert bundle["repair"]["screen_games"] == 128
    assert bundle["repair"]["thresholds_unchanged"] is True
    assert bundle["frozen_data_opened"] is False
