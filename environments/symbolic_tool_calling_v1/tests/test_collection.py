import json

import pytest
from symbolic_tool_calling_v1.artifacts import read_jsonl, verify_artifact
from symbolic_tool_calling_v1.collection import (
    RolloutCollectionConfig,
    TaskDatasetConfig,
    build_tasks,
    collect_rollouts,
    write_task_dataset,
)
from symbolic_tool_calling_v1.models import BenchmarkTask
from symbolic_tool_calling_v1.policies import exploratory_policy, optimal_policy, policy_by_id


def test_task_matrix_is_deterministic_unique_and_balanced():
    config = TaskDatasetConfig(tasks_per_condition=3)
    first = build_tasks(config)
    second = build_tasks(config)
    assert [task.model_dump() for task in first] == [task.model_dump() for task in second]
    assert len(first) == 18
    assert len({task.task_id for task in first}) == 18
    assert {bucket: sum(task.horizon_bucket == bucket for task in first) for bucket in ("short", "medium", "long")} == {
        "short": 6,
        "medium": 6,
        "long": 6,
    }


def test_scripted_policies_create_success_and_length_heterogeneity():
    task = build_tasks(
        TaskDatasetConfig(tasks_per_condition=1, horizon_buckets=("long",), imbalance_settings=("high",))
    )[0]
    optimal = optimal_policy(task, 1)
    exploratory = exploratory_policy(task, 1)
    assert len(optimal) == task.optimal_plan_length
    assert len(exploratory) > len(optimal)
    rollouts = collect_rollouts([task], RolloutCollectionConfig())
    assert [rollout.success for rollout in rollouts] == [True, True, True, False]
    assert len({rollout.total_turns for rollout in rollouts}) > 1
    assert len({rollout.rollout_seed for rollout in rollouts}) == 4
    assert {rollout.prompt_id for rollout in rollouts} == {f"prompt-{task.task_id}"}


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        policy_by_id("missing")


def test_artifact_writer_is_immutable_and_checksummed(tmp_path, repo_root):
    output = tmp_path / "tasks-v1"
    manifest = write_task_dataset(output, TaskDatasetConfig(tasks_per_condition=1), repo_root)
    assert verify_artifact(output) == manifest
    tasks = read_jsonl(output / "tasks.jsonl", BenchmarkTask)
    assert len(tasks) == 6
    summary = json.loads((output / "summary.json").read_text())
    assert summary["num_tasks"] == 6
    with pytest.raises(FileExistsError):
        write_task_dataset(output, TaskDatasetConfig(tasks_per_condition=1), repo_root)

    with (output / "tasks.jsonl").open("a") as file:
        file.write("corruption\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifact(output)
