from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from scripts.summarize_message_credit_audit import load_verified_payloads
from swarm_ctf_eval.arena import Action, state_to_dict
from swarm_ctf_eval.arena_protocol import Broadcast
from swarm_ctf_eval.broadcast_priority import (
    PROMPT_VARIANTS,
    apply_prompt_variant,
    summarize_priority_rows,
)
from swarm_ctf_eval.episode import EMPTY_BROADCAST, EpisodeConfig
from swarm_ctf_eval.message_credit_audit import (
    message_credit_audit_record,
    summarize_message_credit_records,
)
from swarm_ctf_eval.multi_policy_contract import AgentPolicy
from swarm_ctf_eval.prime_rl_bridge import RolloutDecision
from swarm_ctf_eval.rl_v3 import ArenaRLEnv
from swarm_ctf_eval.safety_supervisor import (
    BranchReplay,
    MessageCreditGroupEvidence,
    ReplayTurn,
    RunLock,
    append_hash_chained_record,
    approve_message_credit_group,
    canonical_sha256,
)


def test_message_credit_summary_combines_independently_verified_shards(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    append_hash_chained_record(first, {"pair": 0, "kind": "critical"})
    append_hash_chained_record(first, {"pair": 0, "kind": "decoy"})
    append_hash_chained_record(second, {"pair": 1, "kind": "critical"})
    append_hash_chained_record(second, {"pair": 1, "kind": "decoy"})

    assert load_verified_payloads([first, second]) == [
        {"pair": 0, "kind": "critical"},
        {"pair": 0, "kind": "decoy"},
        {"pair": 1, "kind": "critical"},
        {"pair": 1, "kind": "decoy"},
    ]


def test_message_drop_credit_is_sender_local_and_broadcast_only() -> None:
    config = EpisodeConfig(
        horizon=2,
        communication_cost=0.0,
        invalid_broadcast_cost=0.0,
        invalid_action_cost=0.0,
    )
    source = ArenaRLEnv(seed=73, size=12, config=config)
    source.reset(73)
    initial = source._require_state().clone()
    agent_ids = tuple(sorted(initial.agents))
    broadcasts = {
        agent_id: Broadcast((), Action("WAIT"), 0)
        if agent_id == "blue-0"
        else EMPTY_BROADCAST
        for agent_id in agent_ids
    }

    def replay(dropped_sender: str | None):
        env = ArenaRLEnv(size=12, config=config)
        env.reset_from_state(initial)
        contexts = {}
        outputs = {}
        turns = []
        final = None
        for turn in range(2):
            pre_state = canonical_sha256(state_to_dict(env._require_state()))
            contexts.update(
                {
                    (agent_id, turn, "BROADCAST"): canonical_sha256(observation)
                    for agent_id, observation in env.observations().items()
                }
            )
            delivered = dict(broadcasts)
            if dropped_sender is not None and turn == 0:
                delivered[dropped_sender] = EMPTY_BROADCAST
            phase = env.broadcast_phase(broadcasts, delivered_broadcasts=delivered)
            outputs.update(
                {
                    (agent_id, turn, "BROADCAST"): canonical_sha256(message.to_dict())
                    for agent_id, message in phase.accepted.items()
                }
            )
            contexts.update(
                {
                    (agent_id, turn, "ACT"): canonical_sha256(observation)
                    for agent_id, observation in env.action_observations().items()
                }
            )
            actions = {agent_id: Action("WAIT") for agent_id in agent_ids}
            outputs.update(
                {
                    (agent_id, turn, "ACT"): canonical_sha256(action.to_dict())
                    for agent_id, action in actions.items()
                }
            )
            final = env.advance(actions)
            turns.append(
                ReplayTurn(
                    turn,
                    tuple(sorted(broadcasts.items())),
                    tuple(sorted(phase.delivered.items())),
                    tuple(sorted(actions.items())),
                    pre_state,
                    canonical_sha256(state_to_dict(env._require_state())),
                )
            )
        assert final is not None
        return BranchReplay(dropped_sender, tuple(turns), final.rewards["BLUE"]), contexts, outputs

    bindings = tuple(
        AgentPolicy(
            f"{team.lower()}-{index}",
            team,
            f"blue-policy-{index}" if team == "BLUE" else "red-opponent",
            team == "BLUE",
        )
        for team in ("BLUE", "RED")
        for index in range(4)
    )
    actual_replay, actual_contexts, actual_outputs = replay(None)
    drop_data = {f"blue-{index}": replay(f"blue-{index}") for index in range(4)}
    constraint = "a" * 64

    def decisions(
        branch: str,
        dropped_sender: str | None,
        contexts: dict[tuple[str, int, str], str],
        outputs: dict[tuple[str, int, str], str],
        offset: int,
    ) -> tuple[RolloutDecision, ...]:
        rows = []
        for turn in range(2):
            for phase_index, phase in enumerate(("BROADCAST", "ACT")):
                for agent_index, agent_id in enumerate(agent_ids):
                    team = "BLUE" if agent_id.startswith("blue") else "RED"
                    policy_id = (
                        f"blue-policy-{agent_id[-1]}" if team == "BLUE" else "red-opponent"
                    )
                    revision = "trainable-r0" if team == "BLUE" else "opponent-r0"
                    rows.append(
                        RolloutDecision(
                            "message-game-1",
                            branch,
                            dropped_sender,
                            agent_id,
                            policy_id,
                            revision,
                            team,
                            turn,
                            phase,
                            offset + turn * 16 + phase_index * 8 + agent_index,
                            (1,),
                            (2,),
                            (-0.1,),
                            constraint,
                            f"shared:{agent_id}:{turn}:{phase}",
                            contexts[(agent_id, turn, phase)],
                            canonical_sha256(
                                {
                                    "sampling_key": f"shared:{agent_id}:{turn}:{phase}",
                                    "context_sha256": contexts[(agent_id, turn, phase)],
                                    "policy_id": policy_id,
                                    "revision": revision,
                                }
                            ),
                            outputs[(agent_id, turn, phase)],
                        )
                    )
        return tuple(rows)

    actual_decisions = decisions("actual", None, actual_contexts, actual_outputs, 0)
    drop_decisions = tuple(
        row
        for index in range(4)
        for row in decisions(
            "message_drop",
            f"blue-{index}",
            drop_data[f"blue-{index}"][1],
            drop_data[f"blue-{index}"][2],
            32 + index * 32,
        )
    )
    lock = RunLock(
        "message-credit-smoke",
        "commit-1234567",
        "arena-rl-v3-terminal-control",
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "base-r0",
        tuple((f"blue-policy-{index}", "trainable-r0") for index in range(4)),
        (("red-opponent", "opponent-r0"),),
        None,
        "sft-opponent",
        "opponent-r0",
        (constraint,),
        None,
        "message_drop",
    )
    evidence = MessageCreditGroupEvidence(
        lock.sha256,
        "message-game-1",
        initial,
        canonical_sha256(state_to_dict(initial)),
        config,
        0,
        actual_replay,
        tuple(drop_data[f"blue-{index}"][0] for index in range(4)),
        actual_decisions + drop_decisions,
        {
            row.decision_id: row.rollout_logprobs
            for row in actual_decisions
            if row.team == "BLUE" and row.phase == "BROADCAST" and row.turn == 0
        },
    )
    approval = approve_message_credit_group(
        lock,
        evidence,
        bindings,
        "BLUE",
        b"message-credit-test-signing-key-32b",
    )
    assert len(approval.envelopes) == 4
    assert all(envelope.advantage == 0.0 for envelope in approval.envelopes)
    broadcast_ids = {
        row.decision_id
        for row in actual_decisions
        if row.team == "BLUE" and row.phase == "BROADCAST" and row.turn == 0
    }
    assert {
        decision_id
        for envelope in approval.envelopes
        for decision_id in envelope.decision_ids
    } == broadcast_ids
    assert all(":ACT" not in decision_id for decision_id in broadcast_ids)

    target = sorted(initial.nodes)[0]
    record = message_credit_audit_record(
        evidence,
        approval,
        {
            "source": "curriculum",
            "pair_index": 0,
            "kind": "critical",
            "sender": "blue-0",
            "receiver": "blue-1",
            "target": target,
        },
    )
    assert record["initial_state_sha256"] == evidence.initial_state_sha256
    assert len(record["branches"]) == 5
    assert len(record["decisions"]) == len(evidence.decisions)
    assert record["roles"]["off_role"] == ["blue-2", "blue-3"]
    assert record["receiver_effect"]["actual_actions"]
    assert all(
        turn["legal_actions"] and turn["pre_state_sha256"]
        for branch in record["branches"]
        for turn in branch["turns"]
    )

    paired_records = []
    for pair_index in range(12):
        critical = deepcopy(record)
        critical["scenario"]["pair_index"] = pair_index
        critical["scenario"]["kind"] = "critical"
        critical["target_fact"]["present"] = True
        critical["credits"]["blue-0"]["advantage"] = 0.2
        critical["receiver_effect"]["target_action_sequence_changed"] = True
        decoy = deepcopy(record)
        decoy["scenario"]["pair_index"] = pair_index
        decoy["scenario"]["kind"] = "decoy"
        decoy["target_fact"]["present"] = True
        paired_records.extend((critical, decoy))
    summary = summarize_message_credit_records(paired_records)
    assert summary["verdict"] == "promising"
    assert summary["aggregate"]["intended_sender_positive"] == 12
    assert summary["aggregate"]["localization_ratio"] is None


    first_drop_index = len(actual_decisions)
    mismatched_request = replace(
        evidence.decisions[first_drop_index],
        request_sha256="f" * 64,
    )
    request_tamper = replace(
        evidence,
        decisions=(
            *evidence.decisions[:first_drop_index],
            mismatched_request,
            *evidence.decisions[first_drop_index + 1 :],
        ),
    )
    try:
        approve_message_credit_group(
            lock,
            request_tamper,
            bindings,
            "BLUE",
            b"message-credit-test-signing-key-32b",
        )
    except ValueError as error:
        assert "different inference requests" in str(error)
    else:
        raise AssertionError("a mismatched common-random request must fail closed")

    wrong_drop = replace(
        evidence.drops[0],
        turns=(
            replace(
                evidence.drops[0].turns[0],
                delivered_broadcasts=evidence.actual.turns[0].delivered_broadcasts,
            ),
            *evidence.drops[0].turns[1:],
        ),
    )
    tampered = replace(evidence, drops=(wrong_drop, *evidence.drops[1:]))
    try:
        approve_message_credit_group(
            lock,
            tampered,
            bindings,
            "BLUE",
            b"message-credit-test-signing-key-32b",
        )
    except ValueError as error:
        assert "delivery intervention" in str(error)
    else:
        raise AssertionError("a missing sender-message drop must fail closed")


def test_broadcast_priority_variants_are_generic_and_summarized() -> None:
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "private observation"},
    ]
    for variant, suffix in PROMPT_VARIANTS.items():
        updated = apply_prompt_variant(messages, variant)
        assert updated[0]["content"] == "base" + suffix
        assert messages[0]["content"] == "base"
        assert "target" not in suffix.lower()
    rows = [
        {
            "variant": "current",
            "pair_index": pair,
            "repetition": repetition,
            "protocol_valid": True,
            "target_fact_present": pair % 2 == 0,
            "fact_count": 2,
        }
        for pair in range(12)
        for repetition in range(2)
    ]
    summary = summarize_priority_rows(rows)
    assert summary["current"]["target_fact_rate"] == 0.5
    assert summary["current"]["pairs_target_fact_majority"] == 6
