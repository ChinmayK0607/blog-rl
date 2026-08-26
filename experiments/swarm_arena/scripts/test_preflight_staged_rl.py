from types import SimpleNamespace

import pytest
from preflight_staged_rl import _validate_shared_return_launcher


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
