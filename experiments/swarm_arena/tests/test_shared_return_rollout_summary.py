from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_shared_return_rollouts import summarize


def _write(path: Path, groups: list[dict]) -> Path:
    path.write_text(json.dumps([{"step": 0, "groups": groups}]), encoding="utf-8")
    return path


def _group(kind: str | None, returns: list[float], *, pair: int = 0) -> dict:
    scenario = {"source": "ordinary"}
    if kind is not None:
        scenario = {
            "source": "curriculum",
            "kind": kind,
            "pair_index": pair,
            "sampling_namespace": f"pair-{pair}",
        }
    mean = sum(returns) / len(returns)
    return {
        "scenario": scenario,
        "replicas": [
            {"return": value, "advantage": value - mean} for value in returns
        ],
    }


def test_summarizes_reward_density_and_matched_handoffs(tmp_path: Path) -> None:
    handoff = _write(
        tmp_path / "handoff.json",
        [
            _group("critical", [1.0, 0.0, 0.0, 0.0]),
            _group("decoy", [0.5, 0.0, 0.0, 0.0]),
        ],
    )
    ordinary = _write(
        tmp_path / "ordinary.json",
        [_group(None, [0.4, 0.2, 0.2, 0.6])],
    )

    summary = summarize([handoff, ordinary])

    assert summary["overall"]["groups"] == 3
    assert summary["overall"]["replicas"] == 12
    assert summary["overall"]["nonzero_advantage_rate"] == 1.0
    assert summary["families"]["ordinary"]["return_variance_group_rate"] == 1.0
    assert summary["paired_handoff"] == {
        "complete_pairs": 1,
        "replica_differences": 4,
        "critical_minus_decoy_mean_return": 0.125,
    }
