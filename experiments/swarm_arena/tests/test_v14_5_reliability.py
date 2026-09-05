from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.swarm_arena.scripts.freeze_v14_5_reliable_execution import (
    build_bundle,
    load_hashed,
)
from experiments.swarm_arena.swarm_ctf_eval.staged_runtime import (
    optimizer_application_summary,
    orchestrator_lora,
    parity_quarantined_logical_updates,
)
from prime_rl.trainer.rl.parity import (
    load_rollout_parity_quarantine_counts,
    rollout_parity_failures,
    rollout_parity_quarantine_disposition,
)


def _write_progress(root: Path) -> None:
    progress = [
        {
            "step": index,
            "optimizer_step_applied": index != 1,
            "parity_quarantined": index == 1,
        }
        for index in range(10)
    ]
    (root / "live_rl_progress.json").write_text(
        json.dumps(progress) + "\n",
        encoding="utf-8",
    )


def test_progress_distinguishes_logical_and_optimizer_updates(tmp_path: Path) -> None:
    _write_progress(tmp_path)
    assert optimizer_application_summary(tmp_path, step=10, interval=10) == {
        "logical_updates": 10,
        "optimizer_steps_applied": 9,
        "parity_quarantined_updates": 1,
        "replacement_batches_sampled": 0,
    }


def test_preflight_uses_current_orchestrator_schema() -> None:
    lora = {"name": "blue-2", "rank": 32, "alpha": 64.0}
    assert orchestrator_lora(
        {"student": {"model": {"name": "model", "lora": lora}}},
        "blue-2",
    ) == lora
    with pytest.raises(ValueError, match="student model metadata"):
        orchestrator_lora(
            {"model": {"name": "model", "lora": lora}},
            "blue-2",
        )


def test_controller_reads_append_only_quarantine_record(tmp_path: Path) -> None:
    audit = tmp_path / "audit" / "rollout_parity_quarantine.jsonl"
    audit.parent.mkdir()
    audit.write_text(
        json.dumps(
            {
                "version": "prime-rl-parity-quarantine-v1",
                "logical_update": 2,
                "action": "quarantine_logical_update",
                "optimizer_step_applied": False,
                "replacement_batch_sampled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert parity_quarantined_logical_updates(tmp_path) == {2}


def test_controller_rejects_favorable_replacement_claim(tmp_path: Path) -> None:
    audit = tmp_path / "audit" / "rollout_parity_quarantine.jsonl"
    audit.parent.mkdir()
    audit.write_text(
        json.dumps(
            {
                "version": "prime-rl-parity-quarantine-v1",
                "logical_update": 2,
                "action": "quarantine_logical_update",
                "optimizer_step_applied": False,
                "replacement_batch_sampled": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="replacement sampling"):
        parity_quarantined_logical_updates(tmp_path)


def test_rollout_parity_failures_reports_only_enabled_excesses() -> None:
    metrics = {
        "mean_logprob_error": 0.1,
        "p99_logprob_error": 3.0,
        "max_probability_error": 0.4,
        "p99_probability_error": 0.2,
        "probability_tail_fraction": 0.1,
        "mean_mismatch_kl": 0.0046,
        "max_mismatch_kl": 1.0,
    }
    gate = SimpleNamespace(
        max_mean_logprob_error=0.25,
        max_p99_logprob_error=None,
        max_probability_error=None,
        max_p99_probability_error=None,
        max_probability_tail_fraction=None,
        max_mean_mismatch_kl=0.002,
        max_mismatch_kl=None,
    )

    assert rollout_parity_failures(metrics, gate) == {
        "mean_mismatch_kl": (0.0046, 0.002)
    }


def test_one_failed_batch_per_ten_update_window_is_quarantined() -> None:
    assert rollout_parity_quarantine_disposition(
        logical_update=2,
        prior_window_count=0,
        window_size=10,
        window_limit=1,
    ) == (0, 1, True)
    assert rollout_parity_quarantine_disposition(
        logical_update=9,
        prior_window_count=1,
        window_size=10,
        window_limit=1,
    ) == (0, 2, False)
    assert rollout_parity_quarantine_disposition(
        logical_update=11,
        prior_window_count=0,
        window_size=10,
        window_limit=1,
    ) == (1, 1, True)


@pytest.mark.parametrize(
    ("logical_update", "prior_count", "window_size", "window_limit"),
    ((0, 0, 10, 1), (1, -1, 10, 1), (1, 0, 0, 1), (1, 0, 10, 11)),
)
def test_invalid_parity_quarantine_state_fails_closed(
    logical_update: int,
    prior_count: int,
    window_size: int,
    window_limit: int,
) -> None:
    with pytest.raises(ValueError):
        rollout_parity_quarantine_disposition(
            logical_update=logical_update,
            prior_window_count=prior_count,
            window_size=window_size,
            window_limit=window_limit,
        )


def test_parity_quarantine_limit_survives_trainer_restart(tmp_path: Path) -> None:
    ledger = tmp_path / "rollout_parity_quarantine.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "version": "prime-rl-parity-quarantine-v1",
                "logical_update": 2,
                "window_index": 0,
                "window_size": 10,
                "window_quarantine_count": 1,
                "action": "quarantine_logical_update",
                "optimizer_step_applied": False,
                "replacement_batch_sampled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_rollout_parity_quarantine_counts(ledger, window_size=10) == {0: 1}


def test_rejected_parity_window_cannot_resume(tmp_path: Path) -> None:
    ledger = tmp_path / "rollout_parity_quarantine.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "version": "prime-rl-parity-quarantine-v1",
                "logical_update": 7,
                "action": "abort_quarantine_limit_exceeded",
                "optimizer_step_applied": False,
                "replacement_batch_sampled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cannot resume"):
        load_rollout_parity_quarantine_counts(ledger, window_size=10)


def test_v14_5_cpu_bundle_reproduces_exactly() -> None:
    arena_root = Path(__file__).parents[1]
    repo_root = arena_root.parents[1]
    code_paths = tuple(
        repo_root / path
        for path in (
            "packages/prime-rl-configs/src/prime_rl/configs/trainer.py",
            "src/prime_rl/trainer/rl/loss.py",
            "src/prime_rl/trainer/rl/parity.py",
            "src/prime_rl/trainer/rl/train.py",
            "experiments/swarm_arena/swarm_ctf_eval/staged_runtime.py",
            "experiments/swarm_arena/scripts/prepare_live_rl_run.py",
            "experiments/swarm_arena/scripts/preflight_staged_rl.py",
            "experiments/swarm_arena/scripts/capture_runtime_parity_probe.py",
            "experiments/swarm_arena/scripts/bind_runtime_certificate.py",
            "experiments/swarm_arena/scripts/launch_inference_pool.sh",
            "experiments/swarm_arena/scripts/launch_staged_rl.sh",
            "experiments/swarm_arena/scripts/run_live_rl.py",
            "experiments/swarm_arena/scripts/run_staged_pulses.py",
            "experiments/swarm_arena/scripts/run_live_artifact_mirror.py",
            "experiments/swarm_arena/scripts/summarize_runtime_profile.py",
            "experiments/swarm_arena/swarm_ctf_eval/runtime_topology.py",
            "experiments/swarm_arena/scripts/freeze_v14_5_reliable_execution.py",
        )
    )
    actual = build_bundle(
        parent=load_hashed(arena_root / "data" / "rl_v14_4" / "cpu_bundle.json"),
        prior_trainer_path=(
            arena_root / "configs" / "rl_v14_4_4b_policy_routed_40.toml"
        ),
        trainer_path=(
            arena_root / "configs" / "rl_v14_5_4b_policy_routed_40.toml"
        ),
        plan_path=arena_root / "V14_5_EXECUTION_PLAN.md",
        code_paths=code_paths,
    )
    assert actual == load_hashed(arena_root / "data" / "rl_v14_5" / "cpu_bundle.json")
