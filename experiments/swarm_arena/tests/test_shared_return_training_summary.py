from __future__ import annotations

import json

from scripts.summarize_shared_return_training import summarize


def _step(step: int) -> dict:
    groups = []
    for group_index, kind in enumerate(("critical", "decoy", "critical", "decoy")):
        pair_index = group_index // 2
        returns = [0.2, 0.0, -0.1, 0.1] if kind == "critical" else [0.1, 0.0, -0.1, 0.0]
        mean = sum(returns) / len(returns)
        groups.append(
            {
                "scenario": {
                    "kind": kind,
                    "pair_index": pair_index,
                    "sampling_namespace": f"step-{step}-pair-{pair_index}",
                },
                "replicas": [
                    {"return": value, "advantage": value - mean} for value in returns
                ],
            }
        )
    return {
        "step": step,
        "groups": groups,
        "policy_adapter_sha256": {
            f"blue-{index}": f"step-{step}-policy-{index}" for index in range(4)
        },
    }


def test_shared_return_summary_checks_signal_and_policy_isolation(tmp_path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps([_step(0), _step(1)]), encoding="utf-8")
    result = summarize(progress)
    assert result["completed_steps"] == 2
    assert result["steps"][0]["critical_minus_decoy_mean_return"] == 0.05
    assert result["steps"][1]["policies_changed_since_previous_step"] == 4
    assert all(result["mechanical_checks"].values())


def test_shared_return_summary_accepts_production_50_25_25_mix(tmp_path) -> None:
    row = _step(0)
    for group in row["groups"][:2]:
        group["scenario"] = {"source": "ordinary"}
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps([row]), encoding="utf-8")

    result = summarize(progress)

    step = result["steps"][0]
    assert (step["ordinary_groups"], step["critical_groups"], step["decoy_groups"]) == (
        2,
        1,
        1,
    )
    assert result["mechanical_checks"]["recognized_four_group_mixture"]
