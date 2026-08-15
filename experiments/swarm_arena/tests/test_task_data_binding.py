from pathlib import Path

import pytest
from swarm_ctf_eval.rl_v3 import RL_TASK_VERSION
from swarm_ctf_eval.task_data_binding import (
    RL_TASK_V4_VERSION,
    resolve_task_data_binding,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def test_binds_v3_without_changing_replay_identity() -> None:
    binding = resolve_task_data_binding(DATA_ROOT / "rl_v3", "v3")

    assert binding.task_version == RL_TASK_VERSION
    assert binding.curriculum_manifest("train") == "train.json"
    assert len(binding.train_sha256) == 64
    assert len(binding.development_sha256) == 64
    assert len(binding.final_sha256) == 64


def test_binds_all_v4_task_and_evaluation_inputs() -> None:
    binding = resolve_task_data_binding(DATA_ROOT / "rl_v4", "v4")

    assert binding.task_version == RL_TASK_V4_VERSION
    assert binding.curriculum_manifest("train") == "handoff_train.json"
    assert binding.curriculum_manifest("development") == "handoff_development.json"
    assert binding.curriculum_manifest("frozen_ood") == "handoff_frozen_ood.json"
    assert len({binding.train_sha256, binding.development_sha256, binding.final_sha256}) == 3


def test_rejects_unknown_task_generation_or_split() -> None:
    with pytest.raises(ValueError, match="unsupported task data version"):
        resolve_task_data_binding(DATA_ROOT / "rl_v4", "v5")

    binding = resolve_task_data_binding(DATA_ROOT / "rl_v4", "v4")
    with pytest.raises(ValueError, match="unknown curriculum split"):
        binding.curriculum_manifest("not-a-split")
