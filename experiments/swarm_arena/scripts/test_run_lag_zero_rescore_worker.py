from run_lag_zero_rescore_worker import _rescore_decision_id


def _row(branch: str, replaced_agent: str | None) -> dict[str, object]:
    return {
        "game_id": "game-1:replica-0:swap",
        "branch": branch,
        "replaced_agent": replaced_agent,
        "agent_id": "blue-1",
        "policy_id": "blue-policy-1",
        "policy_revision": "revision-1",
        "team": "BLUE",
        "turn": 2,
        "phase": "ACT",
        "trajectory_index": 1,
        "prompt_ids": [1],
        "completion_ids": [2],
        "rollout_logprobs": [-0.1],
        "constraint_sha256": "a" * 64,
        "sampling_key": "sample-1",
        "context_sha256": "b" * 64,
        "request_sha256": "c" * 64,
        "output_sha256": "d" * 64,
        "allowed_token_ids": [[2]],
        "transport_attempts": 1,
        "serving_allowed_logprobs": [[[2, 0.0]]],
    }


def test_rescore_decision_id_preserves_message_swap_identity() -> None:
    assert _rescore_decision_id(_row("message_swap", "blue-0")) == (
        "game-1:replica-0:swap:swap-message-blue-0:blue-1:2:ACT"
    )


def test_rescore_decision_id_preserves_actual_identity() -> None:
    assert _rescore_decision_id(_row("actual", None)) == (
        "game-1:replica-0:swap:actual:blue-1:2:ACT"
    )
