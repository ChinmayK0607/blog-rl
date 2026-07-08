from pathlib import Path
from typing import Literal

from pydantic import Field

from symbolic_tool_calling_v1.artifacts import stable_hash, token_count, write_artifact
from symbolic_tool_calling_v1.engine import apply_action, initial_state
from symbolic_tool_calling_v1.generator import generate_task
from symbolic_tool_calling_v1.models import BenchmarkTask, FrozenModel
from symbolic_tool_calling_v1.policies import policy_by_id
from symbolic_tool_calling_v1.prompts import TASK_PROMPT
from symbolic_tool_calling_v1.schemas import RolloutRecord, RolloutStep, SamplingConfig
from symbolic_tool_calling_v1.validation import validate_task

TASK_DATASET_VERSION = "1.0.0"
ROLLOUT_DATASET_VERSION = "1.0.0"
PROMPT = TASK_PROMPT


class TaskDatasetConfig(FrozenModel):
    dataset_version: Literal["1.0.0"] = TASK_DATASET_VERSION
    seed: int = 17
    tasks_per_condition: int = Field(default=20, ge=1)
    horizon_buckets: tuple[Literal["short", "medium", "long"], ...] = ("short", "medium", "long")
    imbalance_settings: tuple[Literal["low", "high"], ...] = ("low", "high")
    branching_factor: int = Field(default=2, ge=1, le=2)
    distractor_ratio: float = Field(default=0.5, ge=0, le=1)
    recovery_cost: int = Field(default=2, ge=1)


class RolloutCollectionConfig(FrozenModel):
    dataset_version: Literal["1.0.0"] = ROLLOUT_DATASET_VERSION
    group_size: int = Field(default=4, ge=1)
    policy_ids: tuple[str, ...] = (
        "scripted-optimal-v1",
        "scripted-exploratory-v1",
        "scripted-exploratory-v1",
        "scripted-broken-code-v1",
    )
    policy_checkpoint: str = "deterministic-scripted-v1"
    sampling: SamplingConfig = SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        max_turns=128,
        max_tokens=32768,
    )


def build_tasks(config: TaskDatasetConfig) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    offset = 0
    for horizon in config.horizon_buckets:
        for imbalance in config.imbalance_settings:
            for index in range(config.tasks_per_condition):
                task = generate_task(
                    config.seed + offset + index,
                    horizon_bucket=horizon,
                    branching_factor=config.branching_factor,
                    distractor_ratio=config.distractor_ratio,
                    recovery_cost=config.recovery_cost,
                    verbosity_setting="high" if imbalance == "high" else "low",
                    imbalance_setting=imbalance,
                )
                validate_task(task)
                tasks.append(task)
            offset += config.tasks_per_condition
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task generation produced duplicate task IDs")
    return tasks


def collect_rollout(task: BenchmarkTask, config: RolloutCollectionConfig, rollout_index: int) -> RolloutRecord:
    policy_id = config.policy_ids[rollout_index % len(config.policy_ids)]
    rollout_seed = task.seed * 10_000 + rollout_index
    actions = policy_by_id(policy_id)(task, rollout_seed)
    state = initial_state(task)
    steps: list[RolloutStep] = []
    for turn_index, action in enumerate(actions[: config.sampling.max_turns]):
        model_output = action.model_dump_json()
        next_state, response = apply_action(task, state, action)
        steps.append(
            RolloutStep(
                turn_index=turn_index,
                model_output=model_output,
                tool_call=action,
                tool_response=response,
                environment_state=next_state.model_copy(deep=True),
                input_token_count=token_count(PROMPT) if turn_index == 0 else 0,
                output_token_count=token_count(model_output),
                response_token_count=token_count(response),
            )
        )
        state = next_state
        if state.terminal:
            break
    stop_reason = (
        "submitted" if state.submitted else "max_turns" if len(steps) == config.sampling.max_turns else "policy_stopped"
    )
    identity = {
        "task_id": task.task_id,
        "policy_id": policy_id,
        "rollout_seed": rollout_seed,
        "actions": [action.model_dump(mode="json") for action in actions],
    }
    return RolloutRecord(
        rollout_id=stable_hash(identity, prefix="rollout-")[:25],
        task_id=task.task_id,
        prompt_id=f"prompt-{task.task_id}",
        policy_id=policy_id,
        policy_checkpoint=config.policy_checkpoint,
        sampling_config_hash=stable_hash(config.sampling.model_dump(mode="json")),
        environment_version=task.generator_version,
        rollout_seed=rollout_seed,
        full_prompt_text=PROMPT,
        initial_environment_state=initial_state(task),
        steps=tuple(steps),
        success=state.success,
        terminal_reward=float(state.success),
        total_turns=len(steps),
        total_tokens=sum(step.token_count for step in steps),
        stop_reason=stop_reason,
    )


def collect_rollouts(tasks: list[BenchmarkTask], config: RolloutCollectionConfig) -> list[RolloutRecord]:
    return [collect_rollout(task, config, index) for task in tasks for index in range(config.group_size)]


def write_task_dataset(output_dir: Path, config: TaskDatasetConfig, repo: Path):
    tasks = build_tasks(config)
    summary = {
        "num_tasks": len(tasks),
        "by_horizon": {
            bucket: sum(task.horizon_bucket == bucket for task in tasks) for bucket in config.horizon_buckets
        },
        "by_imbalance": {
            setting: sum(task.imbalance_setting == setting for task in tasks) for setting in config.imbalance_settings
        },
        "optimal_plan_length": {
            "min": min(task.optimal_plan_length for task in tasks),
            "max": max(task.optimal_plan_length for task in tasks),
        },
    }
    return write_artifact(
        output_dir,
        artifact_id=f"tasks-{stable_hash(config.model_dump(mode='json'))[:12]}",
        artifact_type="tasks",
        artifact_version=config.dataset_version,
        config=config,
        records=tasks,
        records_filename="tasks.jsonl",
        summary=summary,
        repo=repo,
    )


def write_rollout_dataset(
    output_dir: Path,
    tasks: list[BenchmarkTask],
    config: RolloutCollectionConfig,
    repo: Path,
):
    rollouts = collect_rollouts(tasks, config)
    summary = {
        "num_rollouts": len(rollouts),
        "success_rate": sum(rollout.success for rollout in rollouts) / len(rollouts),
        "turns": {
            "min": min(rollout.total_turns for rollout in rollouts),
            "max": max(rollout.total_turns for rollout in rollouts),
            "mean": sum(rollout.total_turns for rollout in rollouts) / len(rollouts),
        },
        "prompt_group_sizes": {
            prompt_id: sum(rollout.prompt_id == prompt_id for rollout in rollouts)
            for prompt_id in sorted({rollout.prompt_id for rollout in rollouts})
        },
    }
    return write_artifact(
        output_dir,
        artifact_id=f"rollouts-{stable_hash(config.model_dump(mode='json'))[:12]}",
        artifact_type="rollouts",
        artifact_version=config.dataset_version,
        config=config,
        records=rollouts,
        records_filename="rollouts.jsonl",
        summary=summary,
        repo=repo,
    )
