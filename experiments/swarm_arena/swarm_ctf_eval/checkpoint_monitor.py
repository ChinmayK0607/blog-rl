from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MonitorTaskKind = Literal[
    "export",
    "online_eval",
    "regression_v1",
    "regression_v2",
    "policy_kl",
    "collapse",
    "publish",
]

CHECKPOINT_MONITOR_VERSION = "arena-rl-v4-checkpoint-monitor-v1"
REQUIRED_TASKS = frozenset(
    {
        "export",
        "online_eval",
        "regression_v1",
        "regression_v2",
        "policy_kl",
        "collapse",
        "publish",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class MonitorTask:
    kind: MonitorTaskKind
    argv: tuple[str, ...]

    def validate(self) -> None:
        if self.kind not in REQUIRED_TASKS or not self.argv or any(not value for value in self.argv):
            raise ValueError("checkpoint monitor task is malformed")
        if self.kind == "online_eval":
            try:
                tier = self.argv[self.argv.index("--tier") + 1]
            except (ValueError, IndexError) as error:
                raise ValueError("online evaluation task must explicitly use --tier online") from error
            if tier != "online":
                raise ValueError("checkpoint monitoring cannot open selection or frozen tiers")


@dataclass(frozen=True)
class CheckpointMonitorPlan:
    version: str
    every_updates: int
    workdir: Path
    output_root: Path
    tasks: tuple[MonitorTask, ...]

    def validate(self) -> None:
        if self.version != CHECKPOINT_MONITOR_VERSION or self.every_updates < 1:
            raise ValueError("unsupported checkpoint-monitor plan")
        for task in self.tasks:
            task.validate()
        kinds = [task.kind for task in self.tasks]
        if len(kinds) != len(set(kinds)) or set(kinds) != REQUIRED_TASKS:
            raise ValueError("checkpoint monitor requires each safety/evaluation task exactly once")
        if not self.workdir.is_dir():
            raise ValueError(f"checkpoint monitor workdir does not exist: {self.workdir}")

    @property
    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                **asdict(self),
                "workdir": str(self.workdir),
                "output_root": str(self.output_root),
            }
        )


def load_checkpoint_monitor_plan(path: Path) -> CheckpointMonitorPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    plan = CheckpointMonitorPlan(
        version=str(raw["version"]),
        every_updates=int(raw["every_updates"]),
        workdir=(path.parent / raw["workdir"]).resolve(),
        output_root=(path.parent / raw["output_root"]).resolve(),
        tasks=tuple(
            MonitorTask(kind=row["kind"], argv=tuple(str(value) for value in row["argv"]))
            for row in raw["tasks"]
        ),
    )
    plan.validate()
    return plan


def due_checkpoint_steps(progress: list[dict[str, object]], every_updates: int) -> tuple[int, ...]:
    steps = sorted({int(row["step"]) + 1 for row in progress})
    return tuple(step for step in steps if step % every_updates == 0)


def run_checkpoint_tasks(
    plan: CheckpointMonitorPlan,
    *,
    checkpoint_step: int,
    state_path: Path,
) -> dict[str, object]:
    plan.validate()
    if checkpoint_step < 1 or checkpoint_step % plan.every_updates:
        raise ValueError("checkpoint step is not due under the monitor plan")
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"version": CHECKPOINT_MONITOR_VERSION, "plan_sha256": plan.sha256, "steps": {}}
    )
    if state["plan_sha256"] != plan.sha256:
        raise ValueError("checkpoint monitor state belongs to a different immutable plan")
    step_key = str(checkpoint_step)
    step_state = state["steps"].setdefault(step_key, {"tasks": {}})
    checkpoint_dir = plan.output_root / f"checkpoint_{checkpoint_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        "step": str(checkpoint_step),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(checkpoint_dir),
    }
    for task in plan.tasks:
        if step_state["tasks"].get(task.kind, {}).get("returncode") == 0:
            continue
        argv = tuple(value.format_map(replacements) for value in task.argv)
        log_path = checkpoint_dir / f"{task.kind}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(  # noqa: S603
                argv,
                cwd=plan.workdir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        step_state["tasks"][task.kind] = {
            "argv": argv,
            "log": str(log_path),
            "returncode": completed.returncode,
        }
        _atomic_json(state_path, state)
        if completed.returncode != 0:
            raise RuntimeError(f"checkpoint monitor task failed: {task.kind}")
    step_state["complete"] = True
    _atomic_json(state_path, state)
    return state
