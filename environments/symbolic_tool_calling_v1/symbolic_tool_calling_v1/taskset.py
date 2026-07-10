from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field

from symbolic_tool_calling_v1.generator import generate_task
from symbolic_tool_calling_v1.models import BenchmarkState, BenchmarkTask
from symbolic_tool_calling_v1.prompts import SYSTEM_PROMPT, TASK_PROMPT
from symbolic_tool_calling_v1.servers.tools import SymbolicTools
from symbolic_tool_calling_v1.state import SymbolicState
from symbolic_tool_calling_v1.validation import validate_task


class SymbolicCondition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_bucket: Literal["short", "medium", "long", "xlong", "xxlong"]
    distractor_ratio: float = Field(default=0.5, ge=0, le=1)
    recovery_cost: int = Field(default=2, ge=1)
    verbosity_setting: Literal["low", "high"] = "low"
    imbalance_setting: Literal["low", "high"] = "low"


class SymbolicToolCallingConfig(vf.TasksetConfig):
    num_tasks: int = Field(default=100, ge=1)
    seed: int = 17
    horizon_bucket: Literal["short", "medium", "long", "xlong", "xxlong"] = "medium"
    branching_factor: int = Field(default=2, ge=1, le=2)
    distractor_ratio: float = Field(default=0.5, ge=0, le=1)
    recovery_cost: int = Field(default=2, ge=1)
    verbosity_setting: Literal["low", "high"] = "low"
    imbalance_setting: Literal["low", "high"] = "low"
    conditions: tuple[SymbolicCondition, ...] = ()
    task_file: Path | None = None
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class SymbolicTask(vf.Task):
    spec: BenchmarkTask


class SymbolicToolCallingTaskset(vf.Taskset[SymbolicTask, SymbolicToolCallingConfig, SymbolicState]):
    def load_tasks(self) -> list[SymbolicTask]:
        if self.config.task_file is not None:
            specs = [
                BenchmarkTask.model_validate_json(line)
                for line in self.config.task_file.read_text().splitlines()
                if line
            ]
            if not specs:
                raise ValueError(f"curated task file is empty: {self.config.task_file}")
            if len({spec.task_id for spec in specs}) != len(specs):
                raise ValueError("curated task file contains duplicate task ids")
            for spec in specs:
                validate_task(spec)
            return [
                SymbolicTask(idx=idx, name=spec.task_id, prompt=TASK_PROMPT, system_prompt=SYSTEM_PROMPT, spec=spec)
                for idx, spec in enumerate(specs)
            ]
        tasks = []
        for idx in range(self.config.num_tasks):
            condition = (
                self.config.conditions[idx % len(self.config.conditions)]
                if self.config.conditions
                else SymbolicCondition(
                    horizon_bucket=self.config.horizon_bucket,
                    distractor_ratio=self.config.distractor_ratio,
                    recovery_cost=self.config.recovery_cost,
                    verbosity_setting=self.config.verbosity_setting,
                    imbalance_setting=self.config.imbalance_setting,
                )
            )
            spec = generate_task(
                self.config.seed + idx,
                horizon_bucket=condition.horizon_bucket,
                branching_factor=self.config.branching_factor,
                distractor_ratio=condition.distractor_ratio,
                recovery_cost=condition.recovery_cost,
                verbosity_setting=condition.verbosity_setting,
                imbalance_setting=condition.imbalance_setting,
            )
            validate_task(spec)
            tasks.append(
                SymbolicTask(
                    idx=idx,
                    name=spec.task_id,
                    prompt=TASK_PROMPT,
                    system_prompt=SYSTEM_PROMPT,
                    spec=spec,
                )
            )
        return tasks

    async def setup(self, task: SymbolicTask, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.task = task.spec
        trace.state.environment = BenchmarkState(current_room=task.spec.hidden_graph_spec.start_room)

    def tools(self, task: SymbolicTask) -> list[vf.Toolset]:
        return [SymbolicTools(self.config.tools)]

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        return bool(trace.state.environment and trace.state.environment.terminal)

    @vf.reward
    async def success(self, trace: vf.Trace) -> float:
        return float(bool(trace.state.environment and trace.state.environment.success))

    @vf.metric
    async def environment_stats(self, trace: vf.Trace) -> dict[str, float]:
        state = trace.state.environment
        if state is None:
            return {"environment_turns": 0.0, "invalid_actions": 0.0}
        return {"environment_turns": float(state.turns), "invalid_actions": float(state.invalid_actions)}

    async def finalize(self, task: SymbolicTask, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if trace.state.environment is not None:
            trace.info["final_environment_state"] = trace.state.environment.model_dump(mode="json")
            trace.info["task_id"] = task.spec.task_id
