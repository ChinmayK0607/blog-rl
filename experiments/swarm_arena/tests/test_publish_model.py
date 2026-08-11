from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "publish_model.py"
SPEC = importlib.util.spec_from_file_location("publish_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_model_card_is_explicit_about_scope_and_claim_gate() -> None:
    selection = {"selected_step": 240, "num_candidates": 8, "num_eligible": 4}
    test = {
        "selection_score": 0.99,
        "act": {"exact": 1.0},
        "broadcast": {"exact": 0.98, "supported": 1.0},
    }
    arena = {
        "message_strict_rate": 1.0,
        "generated_minus_dropped_reward": 0.5,
        "manifest_sha256": "frozen",
        "conditions": {
            "generated": {
                "strict_action_rate": 1.0,
                "mean_oracle_regret": 1.0,
                "mean_environment_reward": 2.0,
            }
        },
    }
    comparison = {"claim_gates": {"coordination_improvement_supported": False}}
    card = MODULE.build_model_card(
        "CK0607/model",
        selection,
        test,
        arena,
        comparison,
        source_commit="deadbeef",
        training_run_url="https://wandb.example/run",
    )
    assert "not** a multi-agent-RL-trained model" in card
    assert "claim gate: false" in card
    assert "deadbeef" in card
