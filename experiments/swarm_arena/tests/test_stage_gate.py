from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from swarm_ctf_eval.stage_gate import evaluate_stage_gate, load_stage_gates


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _gates() -> dict:
    body = {
        "version": "arena-rl-stage-gates-v1",
        "checkpoints": {
            "10": {
                "stage": "mechanism",
                "on_fail": "stop_before_next_optimizer_update",
                "requirements": [
                    {
                        "name": "semantic_return",
                        "path": ["communication_effects", "normal_minus_dropped", "mean_difference"],
                        "minimum": 0.02,
                    },
                    {
                        "name": "protocol",
                        "path": ["candidate_protocol", "action_protocol_rate"],
                        "equals": 1.0,
                    },
                ],
            }
        },
    }
    return {**body, "sha256": _digest(body)}


def test_stage_gate_passes_and_binds_summary() -> None:
    result = evaluate_stage_gate(
        _gates(),
        {
            "communication_effects": {"normal_minus_dropped": {"mean_difference": 0.03}},
            "candidate_protocol": {"action_protocol_rate": 1.0},
        },
        step=10,
        summary_sha256="a" * 64,
    )
    assert result["status"] == "passed"
    assert result["summary_sha256"] == "a" * 64
    assert len(result["sha256"]) == 64


def test_stage_gate_fails_without_weakening_other_requirements() -> None:
    result = evaluate_stage_gate(
        _gates(),
        {
            "communication_effects": {"normal_minus_dropped": {"mean_difference": 0.01}},
            "candidate_protocol": {"action_protocol_rate": 1.0},
        },
        step=10,
        summary_sha256="b" * 64,
    )
    assert result["status"] == "failed"
    assert [row["passed"] for row in result["requirements"]] == [False, True]


def test_stage_gate_loader_rejects_mutation(tmp_path: Path) -> None:
    gates = _gates()
    gates["checkpoints"]["10"]["requirements"][0]["minimum"] = -1.0
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(gates), encoding="utf-8")
    with pytest.raises(ValueError, match="body hash mismatch"):
        load_stage_gates(path)
