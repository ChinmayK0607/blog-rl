import pytest
from symbolic_tool_calling_v1.pass_at_k import analyze_pass_at_k


def record(task_id: str, seed: int, success: int, *, errors=None):
    return {
        "task": {
            "spec": {
                "task_id": task_id,
                "seed": seed,
                "horizon_bucket": "long",
                "imbalance_setting": "high",
                "optimal_plan_length": 13,
            }
        },
        "rewards": {"success": success},
        "errors": errors or [],
    }


def test_classifies_all_pass_all_fail_and_mixed_groups():
    records = []
    for task_id, successes in (("fail", 0), ("mixed", 2), ("pass", 4)):
        records.extend(record(task_id, 1, int(index < successes)) for index in range(4))
    groups, summary = analyze_pass_at_k(records, 4, max_selected=1)
    assert {group["task_id"]: group["bucket"] for group in groups} == {
        "fail": "all_fail",
        "mixed": "mixed",
        "pass": "all_pass",
    }
    assert summary["bucket_counts"] == {"all_fail": 1, "mixed": 1, "all_pass": 1}
    assert summary["mixed_task_ids"] == ["mixed"]
    assert summary["selected_task_ids"] == ["mixed"]
    assert summary["num_selected"] == 1


def test_rejects_incomplete_or_errored_groups():
    with pytest.raises(ValueError, match="expected 4"):
        analyze_pass_at_k([record("x", 1, 0)], 4)
    records = [record("x", 1, 0) for _ in range(4)]
    records[0]["errors"] = [{"message": "boom"}]
    with pytest.raises(ValueError, match="errored rollouts"):
        analyze_pass_at_k(records, 4)
