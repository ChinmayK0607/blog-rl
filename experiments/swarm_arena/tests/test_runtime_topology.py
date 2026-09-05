from __future__ import annotations

import pytest

from experiments.swarm_arena.scripts.summarize_runtime_profile import (
    summarize_profile,
)
from experiments.swarm_arena.swarm_ctf_eval.runtime_topology import (
    runtime_topology,
)


def test_six_inference_two_trainer_topology_uses_all_eight_gpus() -> None:
    topology = runtime_topology(
        [0, 1],
        [2, 3, 4, 5, 6, 7],
        [8001, 8002, 8003, 8004, 8005, 8006],
        visible_gpu_count=8,
    )
    assert topology.base_urls == tuple(
        f"http://127.0.0.1:{port}" for port in range(8001, 8007)
    )


@pytest.mark.parametrize(
    ("trainers", "inference", "ports", "message"),
    (
        ([0, 1], [1, 2], [8001, 8002], "overlap"),
        ([0], [1, 2], [8001], "one rollout port"),
        ([0], [2, 3], [8001, 8002], "assign every visible GPU"),
    ),
)
def test_invalid_runtime_topology_fails_closed(
    trainers: list[int],
    inference: list[int],
    ports: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        runtime_topology(
            trainers,
            inference,
            ports,
            visible_gpu_count=4,
        )


def _timed_rows(*, rollout: float, trainer: float) -> list[dict]:
    return [
        {
            "timing": {
                "rollout_generation_seconds": rollout,
                "batch_prepare_seconds": 1.0,
                "trainer_update_seconds": trainer,
                "total_seconds": rollout + trainer + 1.0,
            }
        }
        for _ in range(3)
    ]


def test_runtime_profile_favors_inference_when_rollouts_dominate() -> None:
    result = summarize_profile(
        _timed_rows(rollout=80.0, trainer=10.0),
        trainer_gpus=2,
        inference_gpus=6,
        minimum_updates=3,
    )
    assert result["recommendation"] == "favor_inference"
    assert result["scope"] == "operational_timings_only_no_reward_or_gate_inputs"


def test_runtime_profile_favors_trainer_when_update_wait_dominates() -> None:
    result = summarize_profile(
        _timed_rows(rollout=30.0, trainer=50.0),
        trainer_gpus=2,
        inference_gpus=6,
        minimum_updates=3,
    )
    assert result["recommendation"] == "favor_trainer"


def test_runtime_profile_requires_bounded_evidence_window() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        summarize_profile(
            _timed_rows(rollout=10.0, trainer=10.0)[:2],
            trainer_gpus=2,
            inference_gpus=6,
            minimum_updates=3,
        )


def test_runtime_profile_uses_exact_first_window_and_rejects_nonfinite() -> None:
    rows = _timed_rows(rollout=80.0, trainer=10.0)
    rows.append(_timed_rows(rollout=1.0, trainer=1000.0)[0])
    result = summarize_profile(
        rows,
        trainer_gpus=2,
        inference_gpus=6,
        minimum_updates=3,
    )
    assert result["timed_updates"] == 3
    assert result["recommendation"] == "favor_inference"
    rows[0]["timing"]["trainer_update_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        summarize_profile(
            rows,
            trainer_gpus=2,
            inference_gpus=6,
            minimum_updates=3,
        )
