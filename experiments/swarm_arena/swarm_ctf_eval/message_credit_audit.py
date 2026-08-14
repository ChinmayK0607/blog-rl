from __future__ import annotations

from dataclasses import asdict
from statistics import fmean
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


def _actual_sender_message(record: dict[str, Any]) -> dict[str, Any]:
    sender = record["roles"]["sender"]
    intervention_turn = record["intervention_turn"]
    actual = next(row for row in record["branches"] if row["branch"] == "actual")
    turn = next(row for row in actual["turns"] if row["turn"] == intervention_turn)
    return turn["broadcasts"][sender]


def _successful_target_capture(record: dict[str, Any], branch: dict[str, Any]) -> bool:
    target = record["scenario"]["target"]
    return any(
        event.get("target") == target and event.get("success") is True
        for turn in branch["turns"]
        for event in turn["target_capture_events"]
    )


def summarize_message_credit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen Stage B gates to critical/decoy evidence records."""
    paired: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("schema_version") != "message-credit-audit-v1":
            raise ValueError("unexpected message-credit evidence schema")
        scenario = record["scenario"]
        pair_index = int(scenario["pair_index"])
        kind = str(scenario["kind"])
        if kind not in {"critical", "decoy"}:
            raise ValueError(f"unexpected scenario kind: {kind}")
        bucket = paired.setdefault(pair_index, {})
        if kind in bucket:
            raise ValueError(f"duplicate {kind} evidence for pair {pair_index}")
        bucket[kind] = record
    incomplete = [index for index, pair in paired.items() if set(pair) != {"critical", "decoy"}]
    if incomplete:
        raise ValueError(f"incomplete message-credit pairs: {incomplete}")

    rows = []
    for pair_index, pair in sorted(paired.items()):
        critical = pair["critical"]
        decoy = pair["decoy"]
        if critical["roles"] != decoy["roles"]:
            raise ValueError(f"role mismatch in pair {pair_index}")
        sender = critical["roles"]["sender"]
        receiver = critical["roles"]["receiver"]
        off_role = critical["roles"]["off_role"]
        critical_message = _actual_sender_message(critical)
        decoy_message = _actual_sender_message(decoy)
        effects = {
            agent_id: critical["credits"][agent_id]["advantage"]
            - decoy["credits"][agent_id]["advantage"]
            for agent_id in critical["credits"]
        }
        critical_actual = next(
            row for row in critical["branches"] if row["branch"] == "actual"
        )
        critical_drop = next(
            row
            for row in critical["branches"]
            if row["branch"] == "message_drop" and row["dropped_sender"] == sender
        )
        decoy_actual = next(
            row for row in decoy["branches"] if row["branch"] == "actual"
        )
        decoy_drop = next(
            row
            for row in decoy["branches"]
            if row["branch"] == "message_drop" and row["dropped_sender"] == sender
        )
        rows.append(
            {
                "pair_index": pair_index,
                "sender": sender,
                "receiver": receiver,
                "off_role": off_role,
                "sender_message_identical": critical_message == decoy_message,
                "sender_message_sha256": canonical_sha256(critical_message),
                "target_fact_present": critical["target_fact"]["present"] is True,
                "critical_actual_return": critical["actual_return"],
                "decoy_actual_return": decoy["actual_return"],
                "critical_credits": {
                    agent_id: row["advantage"]
                    for agent_id, row in critical["credits"].items()
                },
                "decoy_credits": {
                    agent_id: row["advantage"]
                    for agent_id, row in decoy["credits"].items()
                },
                "paired_effects": effects,
                "intended_sender_effect": effects[sender],
                "receiver_effect": effects[receiver],
                "off_role_effects": {agent_id: effects[agent_id] for agent_id in off_role},
                "critical_receiver_target_effect": bool(
                    critical["receiver_effect"]["target_action_sequence_changed"]
                    or critical["receiver_effect"]["target_capture_changed"]
                ),
                "decoy_receiver_target_effect": bool(
                    decoy["receiver_effect"]["target_action_sequence_changed"]
                    or decoy["receiver_effect"]["target_capture_changed"]
                ),
                "critical_actual_target_capture": _successful_target_capture(
                    critical, critical_actual
                ),
                "critical_sender_drop_target_capture": _successful_target_capture(
                    critical, critical_drop
                ),
                "decoy_actual_target_capture": _successful_target_capture(
                    decoy, decoy_actual
                ),
                "decoy_sender_drop_target_capture": _successful_target_capture(
                    decoy, decoy_drop
                ),
            }
        )

    intended = [row["intended_sender_effect"] for row in rows]
    off_role = [
        value for row in rows for value in row["off_role_effects"].values()
    ]
    tolerance = 1e-12
    mean_abs_intended = fmean(abs(value) for value in intended)
    mean_abs_off_role = fmean(abs(value) for value in off_role)
    localization_ratio = (
        None
        if mean_abs_off_role <= tolerance
        else mean_abs_intended / mean_abs_off_role
    )
    localization_gate = (
        mean_abs_intended > tolerance
        if localization_ratio is None
        else localization_ratio >= 2.0
    )
    aggregate = {
        "pair_count": len(rows),
        "sender_messages_identical": sum(row["sender_message_identical"] for row in rows),
        "target_fact_present": sum(row["target_fact_present"] for row in rows),
        "intended_sender_mean_effect": fmean(intended),
        "intended_sender_positive": sum(value > tolerance for value in intended),
        "intended_sender_negative": sum(value < -tolerance for value in intended),
        "intended_sender_zero": sum(abs(value) <= tolerance for value in intended),
        "intended_sender_mean_absolute_effect": mean_abs_intended,
        "off_role_mean_absolute_effect": mean_abs_off_role,
        "localization_ratio": localization_ratio,
        "pairs_with_nonzero_off_role_effect": sum(
            any(abs(value) > tolerance for value in row["off_role_effects"].values())
            for row in rows
        ),
        "critical_receiver_target_effects": sum(
            row["critical_receiver_target_effect"] for row in rows
        ),
        "decoy_receiver_target_effects": sum(
            row["decoy_receiver_target_effect"] for row in rows
        ),
        "critical_mean_actual_return": fmean(
            row["critical_actual_return"] for row in rows
        ),
        "decoy_mean_actual_return": fmean(row["decoy_actual_return"] for row in rows),
        "critical_actual_target_capture_rate": fmean(
            row["critical_actual_target_capture"] for row in rows
        ),
        "critical_sender_drop_target_capture_rate": fmean(
            row["critical_sender_drop_target_capture"] for row in rows
        ),
        "decoy_actual_target_capture_rate": fmean(
            row["decoy_actual_target_capture"] for row in rows
        ),
        "decoy_sender_drop_target_capture_rate": fmean(
            row["decoy_sender_drop_target_capture"] for row in rows
        ),
    }
    gates = {
        "identical_sender_message_12_of_12": aggregate["sender_messages_identical"]
        == 12,
        "target_fact_at_least_8_of_12": aggregate["target_fact_present"] >= 8,
        "positive_mean_intended_effect": aggregate["intended_sender_mean_effect"] > 0,
        "sign_count": aggregate["intended_sender_positive"] >= 8
        and aggregate["intended_sender_negative"] <= 2,
        "localization_at_least_2x": localization_gate,
        "off_role_nonzero_at_most_4_of_12": aggregate[
            "pairs_with_nonzero_off_role_effect"
        ]
        <= 4,
        "receiver_target_effect_more_common_in_critical": aggregate[
            "critical_receiver_target_effects"
        ]
        > aggregate["decoy_receiver_target_effects"],
    }
    hard_rejection = not all(
        gates[name]
        for name in (
            "positive_mean_intended_effect",
            "localization_at_least_2x",
            "off_role_nonzero_at_most_4_of_12",
        )
    )
    verdict = (
        "rejected"
        if hard_rejection
        else "promising"
        if all(gates.values())
        else "inconclusive_capability_limited"
    )
    return {
        "schema_version": "message-credit-stage-b-summary-v1",
        "estimand": "critical message-drop credit minus matched-decoy message-drop credit",
        "independent_unit": "certified critical/decoy pair",
        "aggregate": aggregate,
        "gates": gates,
        "verdict": verdict,
        "pairs": rows,
    }
