from __future__ import annotations

import hashlib

import pytest

from experiments.swarm_arena.scripts.apply_v13_ordinary_seed_repair import (
    apply_repair,
    digest,
)


def _inputs() -> tuple[dict, dict]:
    cases = []
    selected = []
    for index in range(8):
        policy = f"blue-{index // 4}"
        family = ("base", "sft", "historical", "current")[index % 4]
        case_id = f"case-{index}"
        cases.append(
            {"case_id": case_id, "focused_agent": policy, "opponent_family": family, "seed": index, "size": 14, "horizon": 6}
        )
        selected.append(
            {
                "accepted": True,
                "case_id": case_id,
                "policy": policy,
                "family": family,
                "seed": 100 + index,
                "size": 18,
                "horizon": 10,
                "return_range": 0.1,
                "action_diversity": 2,
                "advantages": [-0.1, 0.1, -0.1, 0.1],
                "diagnostic_sha256": str(index) * 64,
            }
        )
    manifest_body = {"cases": cases, "version": "screen"}
    search_body = {"training_only": True, "optimizer_updates": 0, "selected": selected, "history": selected}
    return ({**manifest_body, "sha256": digest(manifest_body)}, {**search_body, "sha256": digest(search_body)})


def test_apply_repair_is_hash_bound_and_changes_only_selected_geometry() -> None:
    manifest, search = _inputs()
    result = apply_repair(manifest, search, search_sha256="a" * 64)
    assert [row["seed"] for row in result["cases"]] == list(range(100, 108))
    assert result["seed_repair"]["optimizer_updates"] == 0
    assert result["seed_repair"]["frozen_data_opened"] is False
    assert result["sha256"] == digest({key: value for key, value in result.items() if key != "sha256"})


def test_apply_repair_rejects_nonzero_optimizer_search() -> None:
    manifest, search = _inputs()
    search["optimizer_updates"] = 1
    with pytest.raises(ValueError, match="zero optimizer"):
        apply_repair(manifest, search, search_sha256=hashlib.sha256(b"x").hexdigest())
