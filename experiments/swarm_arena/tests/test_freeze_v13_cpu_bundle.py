from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.swarm_arena.scripts.freeze_v13_cpu_bundle import freeze


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_binds_complete_progress_and_only_gpu_screen_blocker(tmp_path: Path) -> None:
    adapter_hashes = {f"blue-{index}": str(index) * 64 for index in range(4)}
    progress = [
        {
            "step": step,
            "policy_revision": "r" * 64,
            "policy_adapter_sha256": adapter_hashes,
        }
        for step in range(160)
    ]
    progress_path = _write(tmp_path / "progress.json", progress)
    initializer_path = _write(
        tmp_path / "initializer.json",
        {
            "ready": {
                "step": 160,
                "policy_revision": "r" * 64,
                "policy_adapter_sha256": adapter_hashes,
            }
        },
    )
    selection_path = _write(
        tmp_path / "selection.json",
        {
            "admission": {"status": "training_only_complete"},
            "source": {"progress_sha256": _sha(progress_path)},
        },
    )
    curriculum_path = _write(
        tmp_path / "curriculum.json",
        {
            "total_updates": 80,
            "groups_per_update": 4,
            "challenge_role_quotas": {f"blue-{index}": 20 for index in range(4)},
        },
    )
    audit_path = _write(
        tmp_path / "audit.json",
        {
            "status": "cpu_schedule_passed_gpu_gates_pending",
            "remaining_blockers": ["run training-only ordinary pass@4 signal screen"],
            "schedule_sha256": "s" * 64,
        },
    )
    screen_path = _write(
        tmp_path / "screen.json",
        {
            "curriculum_file_sha256": _sha(curriculum_path),
            "curriculum_audit_file_sha256": _sha(audit_path),
            "initializer_manifest_file_sha256": _sha(initializer_path),
            "games": 256,
        },
    )
    result = freeze(
        progress_path=progress_path,
        initializer_path=initializer_path,
        selection_path=selection_path,
        curriculum_path=curriculum_path,
        audit_path=audit_path,
        screen_path=screen_path,
    )
    assert result["status"] == "cpu_complete_gpu_signal_screen_pending"
    assert result["next_gpu_action"]["optimizer_updates"] == 0
    assert result["frozen_data_opened"] is False
