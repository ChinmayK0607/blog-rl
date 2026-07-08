from symbolic_tool_calling_v1.prompts import SYSTEM_PROMPT, TASK_PROMPT
from symbolic_tool_calling_v1.taskset import (
    SymbolicCondition,
    SymbolicToolCallingConfig,
    SymbolicToolCallingTaskset,
)


def test_taskset_applies_benchmark_specific_prompts():
    taskset = SymbolicToolCallingTaskset(SymbolicToolCallingConfig(id="symbolic-tool-calling-v1", num_tasks=1))
    task = taskset.load_tasks()[0]
    assert task.prompt == TASK_PROMPT
    assert task.system_prompt == SYSTEM_PROMPT
    assert "Never guess" in task.system_prompt
    assert "symbolic_submit" in task.system_prompt


def test_taskset_cycles_through_all_declared_conditions():
    config = SymbolicToolCallingConfig(
        id="symbolic-tool-calling-v1",
        num_tasks=6,
        conditions=(
            SymbolicCondition(horizon_bucket="short", distractor_ratio=0.0, imbalance_setting="low"),
            SymbolicCondition(horizon_bucket="medium", distractor_ratio=0.5, imbalance_setting="high"),
        ),
    )
    tasks = SymbolicToolCallingTaskset(config).load_tasks()
    assert [task.spec.horizon_bucket for task in tasks] == ["short", "medium"] * 3
    assert [task.spec.imbalance_setting for task in tasks] == ["low", "high"] * 3
