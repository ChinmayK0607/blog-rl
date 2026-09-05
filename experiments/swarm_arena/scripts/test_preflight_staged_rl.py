from types import SimpleNamespace

import pytest
from preflight_staged_rl import _validate_shared_return_launcher
from swarm_ctf_eval.staged_runtime import orchestrator_lora


def _v12_plan() -> SimpleNamespace:
    return SimpleNamespace(
        shared_return_replicas=4,
        trainable_phases=("ACT",),
        trainable_turn_offsets=(0,),
        shared_return_baseline="paired_receiver_target_swap",
        decoy_shared_return_baseline="paired_receiver_target_swap_challenge",
        action_prompt_profile="full",
        paired_contrast_centering="none",
    )


def test_v12_launcher_requires_focused_agent_credit() -> None:
    with pytest.raises(
        ValueError,
        match="paired message intervention credit requires focused-agent ACT credit",
    ):
        _validate_shared_return_launcher(_v12_plan(), "shared_team")


def test_v12_launcher_accepts_focused_agent_credit() -> None:
    _validate_shared_return_launcher(_v12_plan(), "focused_agent")


def test_preflight_reads_current_student_model_lora_schema() -> None:
    lora = {"name": "blue-2", "rank": 32, "alpha": 64.0}
    assert orchestrator_lora(
        {"student": {"model": {"name": "model", "lora": lora}}},
        "blue-2",
    ) == lora


def test_preflight_rejects_obsolete_top_level_model_schema() -> None:
    with pytest.raises(ValueError, match="student model metadata"):
        orchestrator_lora(
            {"model": {"name": "model", "lora": {"name": "blue-0"}}},
            "blue-0",
        )
