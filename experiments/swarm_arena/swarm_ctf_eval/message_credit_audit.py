from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .arena import Action, legal_actions, state_to_dict
from .arena_protocol import Broadcast
from .rl_v3 import ArenaRLEnv
from .safety_supervisor import Approval, BranchReplay, MessageCreditGroupEvidence


def _actions(actions: tuple[Action, ...]) -> list[dict[str, Any]]:
    return [action.to_dict() for action in actions]


def _broadcasts(rows: tuple[tuple[str, Broadcast], ...]) -> dict[str, dict[str, Any]]:
    return {agent_id: broadcast.to_dict() for agent_id, broadcast in rows}


def _branch_record(
    evidence: MessageCreditGroupEvidence,
    branch: BranchReplay,
    *,
    target: str | None,
) -> dict[str, Any]:
    env = ArenaRLEnv(
        size=len(evidence.initial_state.nodes),
        config=evidence.episode_config,
    )
    env.reset_from_state(evidence.initial_state)
    turns = []
    for replay_turn in branch.turns:
        state = env._require_state()
        legal = {
            agent_id: _actions(legal_actions(state, agent_id))
            for agent_id in sorted(state.agents)
        }
        target_before = (
            state_to_dict(state)["nodes"].get(target) if target is not None else None
        )
        broadcasts = dict(replay_turn.broadcasts)
        delivered = dict(replay_turn.delivered_broadcasts)
        env.broadcast_phase(broadcasts, delivered_broadcasts=delivered)
        chosen = dict(replay_turn.actions)
        transition = env.advance(chosen)
        target_after = (
            state_to_dict(env._require_state())["nodes"].get(target)
            if target is not None
            else None
        )
        events = list(transition.info.get("events", []))
        turns.append(
            {
                "turn": replay_turn.turn,
                "pre_state_sha256": replay_turn.pre_state_sha256,
                "post_state_sha256": replay_turn.post_state_sha256,
                "broadcasts": _broadcasts(replay_turn.broadcasts),
                "delivered_broadcasts": _broadcasts(replay_turn.delivered_broadcasts),
                "legal_actions": legal,
                "actions": {
                    agent_id: action.to_dict()
                    for agent_id, action in replay_turn.actions
                },
                "events": events,
                "target_before": target_before,
                "target_after": target_after,
                "target_capture_events": [
                    event
                    for event in events
                    if event.get("kind") == "CAPTURE"
                    and event.get("target") == target
                ],
            }
        )
    return {
        "branch": "actual" if branch.replaced_agent is None else "message_drop",
        "dropped_sender": branch.replaced_agent,
        "terminal_return": branch.terminal_return,
        "turns": turns,
    }


def _target_fact(
    evidence: MessageCreditGroupEvidence,
    sender: str | None,
    target: str | None,
) -> dict[str, Any]:
    if sender is None or target is None:
        return {"present": None, "facts": []}
    first_turn = next(
        row for row in evidence.actual.turns if row.turn == evidence.intervention_turn
    )
    message = dict(first_turn.broadcasts)[sender]
    facts = [fact.to_dict() for fact in message.facts if fact.node == target]
    return {"present": bool(facts), "facts": facts}


def _receiver_effect(
    branches: list[dict[str, Any]],
    *,
    sender: str | None,
    receiver: str | None,
    target: str | None,
) -> dict[str, Any]:
    if sender is None or receiver is None or target is None:
        return {}
    actual = next(row for row in branches if row["branch"] == "actual")
    sender_drop = next(
        row
        for row in branches
        if row["branch"] == "message_drop" and row["dropped_sender"] == sender
    )

    def receiver_actions(branch: dict[str, Any]) -> list[dict[str, Any]]:
        return [turn["actions"][receiver] for turn in branch["turns"]]

    def target_captures(branch: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            event
            for turn in branch["turns"]
            for event in turn["target_capture_events"]
            if event.get("actor") == receiver
        ]

    actual_actions = receiver_actions(actual)
    dropped_actions = receiver_actions(sender_drop)
    actual_target_actions = [row for row in actual_actions if row.get("target") == target]
    dropped_target_actions = [row for row in dropped_actions if row.get("target") == target]
    actual_captures = target_captures(actual)
    dropped_captures = target_captures(sender_drop)
    return {
        "actual_actions": actual_actions,
        "sender_drop_actions": dropped_actions,
        "action_sequence_changed": actual_actions != dropped_actions,
        "actual_target_actions": actual_target_actions,
        "sender_drop_target_actions": dropped_target_actions,
        "target_action_sequence_changed": actual_target_actions
        != dropped_target_actions,
        "actual_target_capture_events": actual_captures,
        "sender_drop_target_capture_events": dropped_captures,
        "target_capture_changed": actual_captures != dropped_captures,
    }


def message_credit_audit_record(
    evidence: MessageCreditGroupEvidence,
    approval: Approval,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, replay-derived record for causal message-credit audits."""
    if approval.game_id != evidence.game_id:
        raise ValueError("approval and message-credit evidence identify different games")
    sender = scenario.get("sender")
    receiver = scenario.get("receiver")
    target = scenario.get("target")
    if sender is not None and not isinstance(sender, str):
        raise TypeError("scenario sender must be a string")
    if receiver is not None and not isinstance(receiver, str):
        raise TypeError("scenario receiver must be a string")
    if target is not None and not isinstance(target, str):
        raise TypeError("scenario target must be a string")

    branches = [
        _branch_record(evidence, evidence.actual, target=target),
        *(
            _branch_record(evidence, branch, target=target)
            for branch in evidence.drops
        ),
    ]
    blue_agents = sorted(
        agent_id
        for agent_id, agent in evidence.initial_state.agents.items()
        if agent.team == "BLUE"
    )
    credits = {
        envelope.agent_id: {
            "policy_id": envelope.policy_id,
            "advantage": envelope.advantage,
            "decision_ids": list(envelope.decision_ids),
            "completion_tokens": envelope.completion_tokens,
        }
        for envelope in approval.envelopes
    }
    decisions = [
        {
            "decision_id": row.decision_id,
            "branch": row.branch,
            "dropped_sender": row.replaced_agent,
            "agent_id": row.agent_id,
            "team": row.team,
            "policy_id": row.policy_id,
            "policy_revision": row.policy_revision,
            "turn": row.turn,
            "phase": row.phase,
            "trajectory_index": row.trajectory_index,
            "sampling_key": row.sampling_key,
            "context_sha256": row.context_sha256,
            "constraint_sha256": row.constraint_sha256,
            "request_sha256": row.request_sha256,
            "output_sha256": row.output_sha256,
        }
        for row in evidence.decisions
    ]
    return {
        "schema_version": "message-credit-audit-v1",
        "game_id": evidence.game_id,
        "run_lock_sha256": evidence.run_lock_sha256,
        "evidence_sha256": approval.evidence_sha256,
        "approval_signature": approval.signature,
        "initial_state_sha256": evidence.initial_state_sha256,
        "episode_config": asdict(evidence.episode_config),
        "intervention_turn": evidence.intervention_turn,
        "scenario": scenario,
        "roles": {
            "sender": sender,
            "receiver": receiver,
            "off_role": [
                agent_id
                for agent_id in blue_agents
                if agent_id not in {sender, receiver}
            ],
        },
        "target_fact": _target_fact(evidence, sender, target),
        "actual_return": evidence.actual.terminal_return,
        "credits": credits,
        "branches": branches,
        "receiver_effect": _receiver_effect(
            branches,
            sender=sender,
            receiver=receiver,
            target=target,
        ),
        "decisions": decisions,
    }
