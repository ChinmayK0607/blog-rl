from __future__ import annotations

import pytest
from swarm_ctf_eval.checkpoint_monitor import (
    CHECKPOINT_MONITOR_VERSION,
    CheckpointMonitorPlan,
    MonitorTask,
    due_checkpoint_steps,
)


def _tasks(eval_tier: str = "online") -> tuple[MonitorTask, ...]:
    return (
        MonitorTask("export", ("export", "{step}")),
        MonitorTask("online_eval", ("evaluate", "--tier", eval_tier)),
        MonitorTask("regression_v1", ("regression-v1",)),
        MonitorTask("regression_v2", ("regression-v2",)),
        MonitorTask("policy_kl", ("policy-kl",)),
        MonitorTask("collapse", ("collapse",)),
        MonitorTask("publish", ("publish",)),
    )


def test_monitor_selects_only_due_completed_policy_updates(tmp_path) -> None:
    plan = CheckpointMonitorPlan(
        CHECKPOINT_MONITOR_VERSION,
        every_updates=5,
        workdir=tmp_path,
        output_root=tmp_path / "results",
        tasks=_tasks(),
    )

    plan.validate()
    assert due_checkpoint_steps(
        [{"step": step} for step in range(12)],
        plan.every_updates,
    ) == (5, 10)


def test_monitor_refuses_to_open_frozen_evaluation_during_training(tmp_path) -> None:
    plan = CheckpointMonitorPlan(
        CHECKPOINT_MONITOR_VERSION,
        every_updates=5,
        workdir=tmp_path,
        output_root=tmp_path / "results",
        tasks=_tasks("frozen"),
    )

    with pytest.raises(ValueError, match="cannot open selection or frozen tiers"):
        plan.validate()
