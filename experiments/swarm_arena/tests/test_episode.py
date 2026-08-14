from __future__ import annotations

import asyncio
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from swarm_ctf_eval.arena import WAIT, Action, legal_actions, state_to_dict
from swarm_ctf_eval.arena_protocol import Broadcast
from swarm_ctf_eval.collapse_audit import audit_training_collapse
from swarm_ctf_eval.communication_curriculum import (
    generate_manifest,
    generate_pair,
    informed_state,
    reconstruct_manifest_scenario,
    swap_team_labels,
)
from swarm_ctf_eval.crossplay_eval import (
    FROZEN_CROSSPLAY_CASES,
    development_cases,
    evaluate_crossplay,
    parse_conditions,
    prepare_manifest,
    summarize_side_swapped,
)
from swarm_ctf_eval.episode import EMPTY_BROADCAST, ArenaEpisodeEnv, EpisodeConfig, message_units
from swarm_ctf_eval.episode_model_eval import evaluate_episode
from swarm_ctf_eval.episode_protocol import (
    EPISODE_PROMPT_VERSION,
    episode_action_prompt,
    episode_broadcast_prompt,
)
from swarm_ctf_eval.episode_splits import EPISODE_EVAL_CASES
from swarm_ctf_eval.final_eval_runner import FinalEvalIdentity, evaluate_final_case
from swarm_ctf_eval.final_eval_v3 import summarize_final_eval
from swarm_ctf_eval.live_rl_rollout import (
    ChoiceCompletion,
    PolicyEndpoint,
    VLLMChoiceGenerator,
    _verify_serving_constraint_rows,
    build_live_credit_group,
    protocol_constraint_sha256,
)
from swarm_ctf_eval.multi_policy_contract import (
    AgentPolicy,
    AgentTokenSpan,
    attach_credits_to_spans,
    replacement_credits,
    validate_policy_roster,
    validate_token_spans,
)
from swarm_ctf_eval.prime_multi_run_router import (
    PolicyRunRoute,
    merge_routed_batch_groups,
    route_approved_samples,
)
from swarm_ctf_eval.prime_rl_bridge import (
    RolloutDecision,
    build_training_envelopes,
    verify_trainer_logprob_parity,
)
from swarm_ctf_eval.rl_v3 import ArenaRLEnv, terminal_control_delta
from swarm_ctf_eval.safety_supervisor import (
    BranchReplay,
    CreditGroupEvidence,
    ReplayTurn,
    RunLock,
    append_hash_chained_record,
    approve_credit_group,
    canonical_sha256,
    verify_approval_signature,
    verify_hash_chain,
)
from swarm_ctf_eval.structured_protocol import (
    action_json_schema,
    broadcast_json_schema,
    completion_allowed_token_ids,
    protocol_choices,
)


def waits(env: ArenaEpisodeEnv) -> dict[str, Action]:
    assert env.state is not None
    return {agent_id: WAIT for agent_id in env.state.agents}


def test_episode_has_terminal_only_zero_sum_reward() -> None:
    env = ArenaEpisodeEnv(config=EpisodeConfig(horizon=2))
    env.reset(11)
    env.broadcast_phase({})
    first = env.advance(waits(env))
    assert first.rewards == {"BLUE": 0.0, "RED": 0.0}
    env.broadcast_phase({})
    final = env.advance(waits(env))
    assert final.truncated
    assert final.rewards["BLUE"] + final.rewards["RED"] == 0.0


def test_red_gain_has_symmetric_positive_terminal_reward() -> None:
    env = ArenaEpisodeEnv(config=EpisodeConfig(horizon=2))
    env.reset(16)
    assert env.state is not None
    neutral = next(node for node in env.state.nodes.values() if node.owner == "NEUTRAL")
    neutral.owner = "RED"
    rewards = env._terminal_rewards()
    assert rewards["RED"] > 0
    assert rewards["BLUE"] == -rewards["RED"]


def test_messages_are_private_budgeted_and_delivered_only_to_teammates() -> None:
    env = ArenaEpisodeEnv(config=EpisodeConfig(horizon=2, message_budget_per_agent=4))
    env.reset(12)
    assert env.state is not None
    fact = next(iter(env.state.knowledge["blue-0"].values()))
    env.state.knowledge["blue-1"].pop(fact.node, None)
    red_memory = dict(env.state.knowledge["red-0"])
    message = Broadcast((fact,), WAIT, 0)
    phase = env.broadcast_phase({"blue-0": message})
    assert phase.message_units["blue-0"] == message_units(message) == 3
    assert phase.remaining_budget["blue-0"] == 1
    assert len(phase.inboxes["blue-1"]) == 1
    assert phase.shared_fact_updates["blue-1"] == 1
    assert env.state.knowledge["blue-1"][fact.node] == fact
    assert phase.inboxes["red-0"] == ()
    assert env.state.knowledge["red-0"] == red_memory


def test_unsupported_and_over_budget_messages_are_dropped() -> None:
    env = ArenaEpisodeEnv(config=EpisodeConfig(horizon=2, message_budget_per_agent=1))
    env.reset(13)
    assert env.state is not None
    known = next(iter(env.state.knowledge["blue-0"].values()))
    too_expensive = Broadcast((known,), WAIT, 0)
    phase = env.broadcast_phase({"blue-0": too_expensive})
    assert phase.accepted["blue-0"] == EMPTY_BROADCAST
    assert phase.errors["blue-0"] == ("message_budget_exceeded",)
    assert phase.remaining_budget["blue-0"] == 1


def test_resource_regeneration_keeps_long_horizon_active() -> None:
    env = ArenaEpisodeEnv(config=EpisodeConfig(horizon=3, resource_regen_period=1))
    env.reset(14)
    assert env.state is not None
    for agent in env.state.agents.values():
        agent.resource = 0
    env.broadcast_phase({})
    env.advance(waits(env))
    assert all(agent.resource == 1 for agent in env.state.agents.values())


def test_action_phase_requires_broadcast_phase() -> None:
    env = ArenaEpisodeEnv(config=EpisodeConfig(horizon=2))
    env.reset(15)
    try:
        env.advance(waits(env))
    except RuntimeError as error:
        assert "broadcasts" in str(error)
    else:
        raise AssertionError("action phase should require broadcasts")


def test_episode_evaluator_runs_strict_intervention_rollout() -> None:
    class FirstOptionModel:
        name = "first-option"

        def respond(self, messages: list[dict[str, str]], oracle_target: str) -> str:
            del oracle_target
            phase = json.loads(messages[-1]["content"])["phase"]
            if phase == "BROADCAST":
                return '{"facts":[],"intent":null,"request_resource":0}'
            return '{"action_id":"A0"}'

    row = evaluate_episode(FirstOptionModel(), EPISODE_EVAL_CASES[0], "generated")
    assert row["broadcast_protocol_rate"] == 1.0
    assert row["broadcast_grounded_rate"] == 1.0
    assert row["action_protocol_rate"] == 1.0
    assert len(row["turns"]) <= EPISODE_EVAL_CASES[0][2]


def test_crossplay_controls_all_eight_agents_and_preserves_private_history() -> None:
    class FirstOptionModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def respond(self, messages: list[dict[str, str]], oracle_target: str) -> str:
            del oracle_target
            phase = json.loads(messages[-1]["content"])["phase"]
            if phase == "BROADCAST":
                return '{"facts":[],"intent":null,"request_resource":0}'
            return '{"action_id":"A0"}'

    row = evaluate_crossplay(
        {f"blue-{index}": FirstOptionModel(f"blue-policy-{index}") for index in range(4)},
        {f"red-{index}": FirstOptionModel(f"red-policy-{index}") for index in range(4)},
        development_cases(1)[0],
    )
    assert row["metrics"]["BLUE"]["action_protocol_rate"] == 1.0
    assert row["metrics"]["RED"]["action_protocol_rate"] == 1.0
    assert row["prompt_version"] == EPISODE_PROMPT_VERSION
    assert all(len(turn["actions"]) == 8 for turn in row["turns"])
    assert set(row["blue_agent_models"]) == {f"blue-{index}" for index in range(4)}
    assert len(set(row["blue_agent_models"].values())) == 4
    assert parse_conditions("generated:generated,dropped:generated") == (
        ("generated", "generated"),
        ("dropped", "generated"),
    )
    assert len(FROZEN_CROSSPLAY_CASES) == 24
    assert {size for _, size, _ in FROZEN_CROSSPLAY_CASES} == {14, 16}
    assert {horizon for _, _, horizon in FROZEN_CROSSPLAY_CASES} == {6, 8}


def test_side_swapped_summary_removes_map_side_bias() -> None:
    def row(seed: int, blue: str, red: str, blue_return: float, condition: str) -> dict:
        team_metrics = {
            "broadcast_protocol_rate": 1.0,
            "broadcast_grounded_rate": 1.0,
            "action_protocol_rate": 1.0,
            "communication_spend": 2.0,
            "invalid_broadcasts": 0,
            "invalid_actions": 0,
            "duplicate_target_turn_rate": 0.25,
        }
        return {
            "seed": seed,
            "blue_model": blue,
            "red_model": red,
            "blue_condition": condition if blue == "adapter" else "generated",
            "red_condition": condition if red == "adapter" else "generated",
            "metrics": {
                "BLUE": {"terminal_return": blue_return, **team_metrics},
                "RED": {"terminal_return": -blue_return, **team_metrics},
            },
            "inference": {"requests": 8, "wall_seconds": 2.0, "completion_tokens": 80},
        }

    rows = [
        row(1, "adapter", "base", 2.0, "generated"),
        row(1, "base", "adapter", -4.0, "generated"),
        row(1, "adapter", "base", 1.0, "dropped"),
        row(1, "base", "adapter", -1.0, "dropped"),
    ]
    summary = summarize_side_swapped(rows)
    assert summary["conditions"]["generated:generated"]["focal_mean_side_averaged_return"] == 3.0
    assert summary["communication_effects"]["generated_minus_dropped"]["mean_return_difference"] == 2.0
    assert summary["conditions"]["generated:generated"]["focal_metrics"][
        "broadcast_protocol_rate"
    ] == 1.0
    assert summary["inference"]["completion_tokens_per_second"] == 40.0


def test_crossplay_resume_requires_an_identical_manifest(tmp_path: Path) -> None:
    manifest = {"version": "test", "blue_model": "adapter", "red_model": "base"}
    digest = prepare_manifest(tmp_path, manifest, resume=False)
    assert prepare_manifest(tmp_path, manifest, resume=True) == digest

    changed = {**manifest, "red_model": "different-base"}
    try:
        prepare_manifest(tmp_path, changed, resume=True)
    except ValueError as error:
        assert "manifest mismatch" in str(error)
    else:
        raise AssertionError("a changed model pair must not resume into the same rows")


def test_dynamic_protocol_schema_only_allows_grounded_budgeted_broadcasts() -> None:
    env = ArenaEpisodeEnv(seed=21, size=12, config=EpisodeConfig(message_budget_per_agent=2))
    env.reset()
    messages, _ = episode_broadcast_prompt(env, "blue-0")
    body = json.loads(messages[-1]["content"])
    schema = broadcast_json_schema(body)
    branches = schema["anyOf"]
    assert branches
    assert all(
        branch["properties"]["facts"].get("maxItems", 0)
        + int(branch["properties"]["intent"]["const"] is not None)
        + branch["properties"]["request_resource"]["const"]
        <= 1
        for branch in branches
        if branch["properties"]["facts"].get("const") != []
        or branch["properties"]["intent"]["const"] is not None
        or branch["properties"]["request_resource"]["const"]
    )
    known = body["observation"]["known_nodes"]
    for branch in branches:
        enumerated = branch["properties"]["facts"].get("enum", [])
        for facts in enumerated:
            assert len({fact["node"] for fact in facts}) == len(facts)
            assert all(fact in known for fact in facts)
        assert branch["properties"]["intent"]["const"] in [None, *body["legal_intents"]]


def test_dynamic_action_schema_enumerates_only_displayed_action_ids() -> None:
    env = ArenaEpisodeEnv(seed=22, size=12)
    env.reset()
    env.broadcast_phase({})
    messages, _ = episode_action_prompt(env, "red-0")
    body = json.loads(messages[-1]["content"])
    schema = action_json_schema(body)
    assert schema["properties"]["action_id"]["enum"] == [
        row["id"] for row in body["legal_actions"]
    ]


def test_protocol_choice_trie_reconstructs_exact_legal_action_distribution() -> None:
    env = ArenaEpisodeEnv(seed=22, size=12)
    env.reset()
    env.broadcast_phase({})
    messages, _ = episode_action_prompt(env, "red-0")
    choices = protocol_choices(messages)
    assert set(choices) == {
        json.dumps({"action_id": row["id"]}, sort_keys=True, separators=(",", ":"))
        for row in json.loads(messages[-1]["content"])["legal_actions"]
    }
    encoded = [list(choice.encode()) for choice in choices]
    selected = encoded[-1]
    rows = completion_allowed_token_ids(selected, encoded)
    assert len(rows) == len(selected)
    assert all(token_id in allowed for token_id, allowed in zip(selected, rows, strict=True))


def test_rl_prompt_states_that_communication_has_no_reward_cost() -> None:
    env = ArenaRLEnv(config=EpisodeConfig(
        horizon=2,
        communication_cost=0.0,
        invalid_broadcast_cost=0.0,
        invalid_action_cost=0.0,
    ))
    env.reset(29)
    messages, _ = episode_broadcast_prompt(env, "blue-0")
    body = json.loads(messages[-1]["content"])
    assert body["reward_contract"] == {
        "terminal_only": True,
        "objective": "maximize normalized terminal controlled-node margin change",
        "communication_has_reward_cost": False,
        "invalid_outputs_are_rewarded": False,
    }


def test_rl_v3_reward_is_only_normalized_terminal_control_delta() -> None:
    env = ArenaRLEnv(config=EpisodeConfig(
        horizon=2,
        communication_cost=0.0,
        invalid_broadcast_cost=0.0,
        invalid_action_cost=0.0,
    ))
    env.reset(31)
    assert env._initial_state is not None
    env.broadcast_phase({})
    first = env.advance(waits(env))
    assert first.rewards == {"BLUE": 0.0, "RED": 0.0}
    env.broadcast_phase({})
    final = env.advance(waits(env))
    expected = terminal_control_delta(env._initial_state, env._require_state(), "BLUE")
    assert final.rewards == {"BLUE": expected, "RED": -expected}
    assert final.info["reward_definition"] == "normalized terminal control delta"


def test_certified_curriculum_pairs_critical_information_with_zero_value_decoy() -> None:
    pair = generate_pair(3_000_000, 12, role_pair=("blue-2", "blue-3"))
    assert pair is not None
    critical, decoy = pair
    assert (critical.sender, critical.receiver) == ("blue-2", "blue-3")
    assert critical.minimum_advantage > 0.1
    assert decoy.minimum_advantage == 0.0
    informed = informed_state(critical.state, critical.sender, critical.team, critical.target)
    assert Action("CAPTURE", critical.target) not in legal_actions(
        critical.state, critical.receiver
    )
    assert Action("CAPTURE", critical.target) in legal_actions(informed, critical.receiver)
    serialized_critical = json.loads(json.dumps(critical.manifest_row()))
    serialized_decoy = json.loads(json.dumps(decoy.manifest_row()))
    assert reconstruct_manifest_scenario(serialized_critical) == critical
    assert reconstruct_manifest_scenario(serialized_decoy) == decoy


def test_curriculum_manifest_balances_every_ordered_role_pair() -> None:
    manifest = generate_manifest(count=12, seed_start=3_100_000)
    roles = {
        (pair["critical"]["sender"], pair["critical"]["receiver"])
        for pair in manifest["pairs"]
    }
    assert len(roles) == 12
    assert all(pair["critical"]["minimum_advantage"] > 0 for pair in manifest["pairs"])
    assert all(pair["decoy"]["minimum_advantage"] == 0 for pair in manifest["pairs"])


def test_final_eval_runner_supports_four_policy_rosters_and_true_side_swap() -> None:
    class FirstOptionModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def respond(self, messages: list[dict[str, str]], oracle_target: str) -> str:
            del oracle_target
            phase = json.loads(messages[-1]["content"])["phase"]
            if phase == "BROADCAST":
                return '{"facts":[],"intent":null,"request_resource":0}'
            return '{"action_id":"A0"}'

    pair = generate_pair(3_200_003, 12, role_pair=("blue-0", "blue-1"))
    assert pair is not None
    critical, _ = pair
    assert state_to_dict(swap_team_labels(swap_team_labels(critical.state))) == state_to_dict(
        critical.state
    )
    identity = FinalEvalIdentity(
        "critical-1",
        "critical",
        "candidate_rl",
        "candidate-revision",
        "identity",
        "identity",
        "canonical",
        "opponent",
        "opponent-revision",
        "sample-1",
    )
    focal = tuple(FirstOptionModel(f"focal-{index}") for index in range(4))
    opponent = tuple(FirstOptionModel(f"opponent-{index}") for index in range(4))
    for side in ("BLUE", "RED"):
        row, raw = evaluate_final_case(
            focal,
            opponent,
            (critical.seed, critical.size, 2),
            identity,
            focal_side=side,
            condition="normal",
            initial_state=critical.state,
            critical_target=critical.target,
        )
        assert row["side"] == side
        assert row["horizon"] == 2
        assert len(raw["turns"]) == 1
        assert len(set(raw[f"{side.lower()}_agent_models"].values())) == 4
    permuted_identity = FinalEvalIdentity(
        "critical-1",
        "critical",
        "candidate_rl",
        "candidate-revision",
        "perm-2301",
        "perm-1032",
        "permuted-2",
        "opponent",
        "opponent-revision",
        "sample-1",
    )
    _, raw = evaluate_final_case(
        focal,
        opponent,
        (critical.seed, critical.size, 2),
        permuted_identity,
        focal_side="BLUE",
        condition="normal",
        initial_state=critical.state,
        critical_target=critical.target,
    )
    assert raw["blue_agent_models"]["blue-1"] == "focal-2"


def test_four_policy_contract_assigns_distinct_replacement_credit_to_owned_spans() -> None:
    bindings = tuple(
        AgentPolicy(
            f"{team.lower()}-{index}",
            team,
            f"{team.lower()}-policy-{index}",
            team == "BLUE",
        )
        for team in ("BLUE", "RED")
        for index in range(4)
    )
    validate_policy_roster(bindings, "BLUE")
    spans = tuple(
        AgentTokenSpan(
            "game-1",
            f"blue-{index}",
            f"blue-policy-{index}",
            "BLUE",
            0,
            "BROADCAST",
            index,
            100,
            8,
        )
        for index in range(4)
    )
    validate_token_spans(spans, bindings)
    credits = replacement_credits(
        0.5,
        {"blue-0": 0.1, "blue-1": 0.3, "blue-2": 0.6, "blue-3": 0.5},
        bindings,
        "BLUE",
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            (credit.advantage for credit in credits),
            (0.4, 0.2, -0.1, 0.0),
            strict=True,
        )
    )
    attached = attach_credits_to_spans(spans, credits)
    assert [attached[key]["policy_id"] for key in sorted(attached)] == [
        "blue-policy-0",
        "blue-policy-1",
        "blue-policy-2",
        "blue-policy-3",
    ]


def test_prime_bridge_routes_actual_tokens_and_checks_logprob_parity() -> None:
    bindings = tuple(
        AgentPolicy(
            f"{team.lower()}-{index}",
            team,
            f"{team.lower()}-policy-{index}",
            team == "BLUE",
        )
        for team in ("BLUE", "RED")
        for index in range(4)
    )
    credits = replacement_credits(
        0.4,
        {"blue-0": 0.1, "blue-1": 0.2, "blue-2": 0.3, "blue-3": 0.4},
        bindings,
        "BLUE",
    )

    def decision(
        agent_id: str,
        index: int,
        branch: str = "actual",
        replaced_agent: str | None = None,
    ) -> RolloutDecision:
        team = "BLUE" if agent_id.startswith("blue") else "RED"
        return RolloutDecision(
            "game-7",
            branch,
            replaced_agent,
            agent_id,
            f"{agent_id.split('-')[0]}-policy-{agent_id[-1]}",
            "revision-1",
            team,
            0,
            "ACT",
            index,
            (1, 2),
            (3,),
            (-0.25,),
            "a" * 64,
            "sample-7",
            "c" * 64,
            "e" * 64,
            "d" * 64,
        )

    actual = tuple(
        decision(f"{team}-{index}", offset * 4 + index)
        for offset, team in enumerate(("blue", "red"))
        for index in range(4)
    )
    replacements = tuple(
        decision(
            f"{team}-{agent_index}",
            8 + replaced_index * 8 + team_index * 4 + agent_index,
            "replacement",
            f"blue-{replaced_index}",
        )
        for replaced_index in range(4)
        for team_index, team in enumerate(("blue", "red"))
        for agent_index in range(4)
    )
    decisions = actual + replacements
    envelopes = build_training_envelopes(decisions, bindings, credits, "BLUE")
    assert [row.policy_id for row in envelopes] == [f"blue-policy-{index}" for index in range(4)]
    blue_actual = tuple(row for row in actual if row.team == "BLUE")
    trainer = {row.decision_id: row.rollout_logprobs for row in blue_actual}
    report = verify_trainer_logprob_parity(
        actual,
        trainer,
        frozenset(f"blue-policy-{index}" for index in range(4)),
    )
    assert report == {
        "status": "passed",
        "decisions": 4,
        "tokens": 4,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "p99_abs_error": 0.0,
        "max_probability_error": 0.0,
        "p99_probability_error": 0.0,
        "probability_tail_fraction": 0.0,
        "mean_mismatch_kl": 0.0,
        "max_mismatch_kl": 0.0,
    }


def test_final_eval_pairs_by_seed_and_separates_capability_from_communication() -> None:
    rows = []

    def add(
        case_id: str,
        suite: str,
        opponent: str,
        side: str,
        variant: str,
        assignment: str,
        condition: str,
        value: float,
        role_assignment: str = "identity",
        option_order: str = "canonical",
    ) -> None:
        rows.append(
            {
                "case_id": case_id,
                "suite": suite,
                "opponent_id": opponent,
                "opponent_revision": "commit-1234567",
                "side": side,
                "policy_variant": variant,
                "policy_revision": f"revision-{variant}",
                "policy_assignment": assignment,
                "role_assignment": role_assignment,
                "option_order": option_order,
                "condition": condition,
                "sampling_key": f"sampling-{case_id}",
                "terminal_return": value,
                "messages_nonempty": int(condition == "normal"),
                "critical_capture": condition == "normal",
            }
        )

    for case_id in ("seed-1", "seed-2"):
        for opponent in ("base@1", "sft@1", "league@1"):
            for side in ("BLUE", "RED"):
                add(case_id, "ordinary_ood", opponent, side, "candidate_rl", "identity", "normal", 0.4)
                add(case_id, "ordinary_ood", opponent, side, "sft_init", "identity", "normal", 0.1)
                add(case_id, "ordinary_ood", opponent, side, "action_only_rl", "identity", "normal", 0.3)
                add(case_id, "ordinary_ood", opponent, side, "candidate_rl", "shuffle-1", "normal", 0.2)
                add(case_id, "ordinary_ood", opponent, side, "candidate_rl", "identity", "normal", 0.4, "perm-1032")
                add(
                    case_id,
                    "ordinary_ood",
                    opponent,
                    side,
                    "candidate_rl",
                    "identity",
                    "normal",
                    0.4,
                    "identity",
                    "permuted-1",
                )
                for condition in ("normal", "dropped", "sender_shuffled", "delayed", "zero_budget"):
                    add(
                        case_id,
                        "critical",
                        opponent,
                        side,
                        "candidate_rl",
                        "identity",
                        condition,
                        0.5 if condition == "normal" else 0.0,
                    )
                add(case_id, "critical", opponent, side, "candidate_rl", "shuffle-1", "normal", 0.1)
                add(case_id, "critical", opponent, side, "candidate_rl", "identity", "normal", 0.5, "perm-1032")
                add(
                    case_id,
                    "critical",
                    opponent,
                    side,
                    "candidate_rl",
                    "identity",
                    "normal",
                    0.5,
                    "identity",
                    "permuted-1",
                )
                add(case_id, "decoy", opponent, side, "candidate_rl", "identity", "normal", 0.0)
                add(case_id, "decoy", opponent, side, "candidate_rl", "identity", "dropped", 0.0)

    summary = summarize_final_eval(rows)
    assert summary["capability_rl_minus_sft"]["independent_seed_units"] == 2
    assert math.isclose(summary["capability_rl_minus_sft"]["mean_difference"], 0.3)
    assert summary["communication_effects"]["normal_minus_dropped"]["independent_seed_units"] == 2
    assert summary["claim_checks"]["communication_claim_passed"]
    assert summary["claim_checks"]["specialization_interval_positive"]
    assert summary["claim_checks"]["role_label_equivalence_within_0_02"]
    assert summary["claim_checks"]["option_order_equivalence_within_0_02"]


def test_collapse_audit_separates_return_gain_from_message_gain() -> None:
    rows = []
    for game_index in range(20):
        opponent = ("base@1", "sft@1", "league@1")[game_index % 3]
        for agent_index in range(4):
            common = {
                "game_id": f"game-{game_index}",
                "agent_id": f"blue-{agent_index}",
                "policy_id": f"blue-policy-{agent_index}",
                "opponent_id": opponent,
                "side": "BLUE" if game_index % 2 == 0 else "RED",
                "message_nonempty": game_index % 2 == 0,
                "message_target": f"node-{game_index % 5}",
                "action_signature": f"action-{game_index % 4}",
                "kl": 0.01,
            }
            rows.append(
                dict(common, checkpoint_id="sft", condition="normal", terminal_return=0.1)
            )
            rows.append(
                dict(common, checkpoint_id="rl", condition="normal", terminal_return=0.2)
            )
            rows.append(
                dict(common, checkpoint_id="rl", condition="dropped", terminal_return=0.19)
            )
    report = audit_training_collapse(
        rows,
        candidate_checkpoint="rl",
        baseline_checkpoint="sft",
    )
    assert report["passed"]
    for row in rows:
        if row["checkpoint_id"] == "rl" and row["condition"] == "dropped":
            row["terminal_return"] = 0.2
    report = audit_training_collapse(
        rows,
        candidate_checkpoint="rl",
        baseline_checkpoint="sft",
    )
    assert report["flags"]["return_gain_without_message_gain"]


def test_fail_closed_supervisor_replays_all_branches_and_hash_chains_approvals() -> None:
    config = EpisodeConfig(
        horizon=2,
        communication_cost=0.0,
        invalid_broadcast_cost=0.0,
        invalid_action_cost=0.0,
    )
    source = ArenaRLEnv(seed=41, size=12, config=config)
    source.reset()
    initial = source._require_state().clone()

    def replay(
        replaced_agent: str | None,
    ) -> tuple[
        BranchReplay,
        dict[tuple[str, int, str], str],
        dict[tuple[str, int, str], str],
    ]:
        env = ArenaRLEnv(size=12, config=config)
        env.reset_from_state(initial)
        turns = []
        contexts = {}
        outputs = {}
        final = None
        for turn in range(2):
            pre = canonical_sha256(state_to_dict(env._require_state()))
            contexts.update(
                {
                    (agent_id, turn, "BROADCAST"): canonical_sha256(observation)
                    for agent_id, observation in env.observations().items()
                }
            )
            env.broadcast_phase({})
            outputs.update(
                {
                    (agent_id, turn, "BROADCAST"): canonical_sha256(
                        broadcast.to_dict()
                    )
                    for agent_id, broadcast in env._phase.accepted.items()
                }
            )
            contexts.update(
                {
                    (agent_id, turn, "ACT"): canonical_sha256(observation)
                    for agent_id, observation in env.action_observations().items()
                }
            )
            actions = waits(env)
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
                    (),
                    (),
                    tuple(sorted(actions.items())),
                    pre,
                    canonical_sha256(state_to_dict(env._require_state())),
                )
            )
        assert final is not None
        return (
            BranchReplay(replaced_agent, tuple(turns), final.rewards["BLUE"]),
            contexts,
            outputs,
        )

    bindings = tuple(
        AgentPolicy(
            f"{team.lower()}-{index}",
            team,
            f"{team.lower()}-policy-{index}",
            team == "BLUE",
        )
        for team in ("BLUE", "RED")
        for index in range(4)
    )
    constraint = "b" * 64
    lock = RunLock(
        "run-1",
        "commit-1234567",
        "arena-rl-v3-terminal-control",
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "base-revision",
        tuple((f"blue-policy-{index}", "trainable-revision") for index in range(4)),
        (
            *((f"red-policy-{index}", "opponent-revision") for index in range(4)),
            ("sft-replacement", "sft-revision"),
        ),
        "sft-replacement",
        "opponent",
        "opponent-revision",
        (constraint,),
    )

    def decision(
        agent_id: str,
        turn: int,
        phase: str,
        trajectory_index: int,
        replaced_agent: str | None,
        context_sha256: str,
        output_sha256: str,
    ) -> RolloutDecision:
        branch = "actual" if replaced_agent is None else "replacement"
        prefix, index = agent_id.split("-")
        policy_id = f"{prefix}-policy-{index}"
        revision = "trainable-revision" if prefix == "blue" else "opponent-revision"
        if agent_id == replaced_agent:
            policy_id = "sft-replacement"
            revision = "sft-revision"
        return RolloutDecision(
            "game-1",
            branch,
            replaced_agent,
            agent_id,
            policy_id,
            revision,
            "BLUE" if prefix == "blue" else "RED",
            turn,
            phase,
            trajectory_index,
            (1, 2),
            (3,),
            (-0.25,),
            constraint,
            f"game-1:{agent_id}:{turn}:{phase}",
            context_sha256,
            canonical_sha256(
                {
                    "sampling_key": f"game-1:{agent_id}:{turn}:{phase}",
                    "context_sha256": context_sha256,
                    "policy_id": policy_id,
                    "revision": revision,
                }
            ),
            output_sha256,
        )

    actual_replay_data = replay(None)
    actual_replay, actual_contexts, _ = actual_replay_data
    replacement_data = {
        f"blue-{index}": replay(f"blue-{index}") for index in range(4)
    }
    actual_decisions = tuple(
        decision(
            f"{team}-{index}",
            turn,
            phase,
            turn * 16 + phase_index * 8 + team_index * 4 + index,
            None,
            actual_contexts[(f"{team}-{index}", turn, phase)],
            actual_replay_data[2][(f"{team}-{index}", turn, phase)],
        )
        for turn in range(2)
        for phase_index, phase in enumerate(("BROADCAST", "ACT"))
        for team_index, team in enumerate(("blue", "red"))
        for index in range(4)
    )
    replacement_decisions = tuple(
        decision(
            f"{team}-{index}",
            turn,
            phase,
            32 + replaced_index * 32 + turn * 16 + phase_index * 8 + team_index * 4 + index,
            f"blue-{replaced_index}",
            replacement_data[f"blue-{replaced_index}"][1][
                (f"{team}-{index}", turn, phase)
            ],
            replacement_data[f"blue-{replaced_index}"][2][
                (f"{team}-{index}", turn, phase)
            ],
        )
        for replaced_index in range(4)
        for turn in range(2)
        for phase_index, phase in enumerate(("BROADCAST", "ACT"))
        for team_index, team in enumerate(("blue", "red"))
        for index in range(4)
    )
    evidence = CreditGroupEvidence(
        lock.sha256,
        "game-1",
        initial,
        canonical_sha256(state_to_dict(initial)),
        config,
        actual_replay,
        tuple(replacement_data[f"blue-{index}"][0] for index in range(4)),
        actual_decisions + replacement_decisions,
        {
            row.decision_id: row.rollout_logprobs
            for row in actual_decisions
            if row.team == "BLUE"
        },
    )
    signing_key = b"supervisor-test-key-32-bytes-long!!"
    approval = approve_credit_group(lock, evidence, bindings, "BLUE", signing_key)
    assert len(approval.envelopes) == 4
    assert approval.logprob_max_abs_error == 0.0
    verify_approval_signature(approval, signing_key)
    try:
        verify_approval_signature(
            replace(approval, replay_return=approval.replay_return + 0.1),
            signing_key,
        )
    except ValueError as error:
        assert "signature" in str(error)
    else:
        raise AssertionError("tampered approval must fail signature verification")

    bad_actual = replace(evidence.actual, terminal_return=evidence.actual.terminal_return + 0.1)
    try:
        approve_credit_group(
            lock,
            replace(evidence, actual=bad_actual),
            bindings,
            "BLUE",
            signing_key,
        )
    except ValueError as error:
        assert "return disagrees" in str(error)
    else:
        raise AssertionError("tampered reward must fail closed")

    bad_decision = replace(evidence.decisions[0], output_sha256="f" * 64)
    try:
        approve_credit_group(
            lock,
            replace(evidence, decisions=(bad_decision, *evidence.decisions[1:])),
            bindings,
            "BLUE",
            signing_key,
        )
    except ValueError as error:
        assert "does not match replayed action" in str(error)
    else:
        raise AssertionError("model-output/replay mismatch must fail closed")

    bad_revision = replace(evidence.decisions[0], policy_revision="stale-revision")
    try:
        approve_credit_group(
            lock,
            replace(evidence, decisions=(bad_revision, *evidence.decisions[1:])),
            bindings,
            "BLUE",
            signing_key,
        )
    except ValueError as error:
        assert "stale or unexpected" in str(error)
    else:
        raise AssertionError("stale policy evidence must fail closed")

    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "approvals.jsonl"
        append_hash_chained_record(trace, {"game": "game-1", "status": "approved"})
        append_hash_chained_record(trace, {"game": "game-2", "status": "rejected"})
        assert len(verify_hash_chain(trace)) == 2
        records = trace.read_text(encoding="utf-8").replace("approved", "tampered")
        trace.write_text(records, encoding="utf-8")
        try:
            verify_hash_chain(trace)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("tampered audit trace must fail verification")


def test_live_credit_group_routes_only_after_bound_trainer_parity_gate() -> None:
    seen_sampling_keys: list[str] = []

    class FirstChoiceGenerator:
        async def generate(
            self,
            endpoint: PolicyEndpoint,
            messages: list[dict[str, str]],
            *,
            sampling_key: str,
        ) -> ChoiceCompletion:
            seen_sampling_keys.append(sampling_key)
            text = protocol_choices(messages)[0]
            request_sha256 = canonical_sha256(
                {
                    "policy_id": endpoint.policy_id,
                    "revision": endpoint.revision,
                    "messages": messages,
                    "sampling_key": sampling_key,
                }
            )
            return ChoiceCompletion(
                (10,),
                (11,),
                (0.0,),
                ((11,),),
                text,
                request_sha256,
            )

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
    endpoints = tuple(
        PolicyEndpoint(f"blue-policy-{index}", "trainable-r0", f"blue-{index}", ("http://unused",))
        for index in range(4)
    ) + (
        PolicyEndpoint("red-opponent", "opponent-r0", "red", ("http://unused",)),
        PolicyEndpoint("sft-replacement", "sft-r0", "sft", ("http://unused",)),
    )
    gate_sha256 = "c" * 64
    lock = RunLock(
        "live-smoke",
        "commit-1234567",
        "arena-rl-v3-terminal-control",
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "base-r0",
        tuple((f"blue-policy-{index}", "trainable-r0") for index in range(4)),
        (("red-opponent", "opponent-r0"), ("sft-replacement", "sft-r0")),
        "sft-replacement",
        "opponent",
        "opponent-r0",
        (
            protocol_constraint_sha256("BROADCAST"),
            protocol_constraint_sha256("ACT"),
        ),
        gate_sha256,
    )
    rl_config = EpisodeConfig(
        horizon=2,
        communication_cost=0.0,
        invalid_broadcast_cost=0.0,
        invalid_action_cost=0.0,
    )
    initial_env = ArenaRLEnv(seed=101, size=12, config=rl_config)
    initial_env.reset(101)
    supplied_initial_state = initial_env._require_state().clone()
    group = asyncio.run(
        build_live_credit_group(
            FirstChoiceGenerator(),  # type: ignore[arg-type]
            game_id="live-game-1",
            seed=101,
            size=12,
            config=rl_config,
            bindings=bindings,
            policies=endpoints,
            replacement_policy_id="sft-replacement",
            run_lock_sha256=lock.sha256,
            initial_state=supplied_initial_state,
            sampling_namespace="matched-pair-7",
        )
    )
    assert seen_sampling_keys
    assert all(key.startswith("matched-pair-7:") for key in seen_sampling_keys)
    signing_key = b"live-supervisor-key-at-least-32-bytes"
    approval = approve_credit_group(lock, group.evidence, bindings, "BLUE", signing_key)
    assert approval.parity_mode == "trainer_pre_step"
    assert approval.logprob_max_abs_error is None
    assert approval.mismatch_kl_max is None
    routes = tuple(
        PolicyRunRoute(f"blue-policy-{index}", f"run_blue_{index}")
        for index in range(4)
    )
    try:
        route_approved_samples(
            approval,
            group.owned_samples,
            routes,
            step=0,
            signing_key=signing_key,
        )
    except ValueError as error:
        assert "active trainer pre-step gate" in str(error)
    else:
        raise AssertionError("deferred parity must be bound to the trainer gate")
    batches = route_approved_samples(
        approval,
        group.owned_samples,
        routes,
        step=0,
        signing_key=signing_key,
        trainer_parity_gate_sha256=gate_sha256,
    )
    assert set(batches) == {f"run_blue_{index}" for index in range(4)}
    assert all(len(batch.examples) == 4 for batch in batches.values())
    merged = merge_routed_batch_groups((batches, batches), step=0)
    assert set(merged) == set(batches)
    assert all(len(batch.examples) == 8 for batch in merged.values())


def test_serving_constraint_rows_reject_a_different_normalization_mask() -> None:
    with pytest.raises(ValueError, match="structured mask mismatch"):
        _verify_serving_constraint_rows(
            [11],
            [[11, 12]],
            [
                {
                    "top_logprobs": [
                        {"token": "token_id:11", "logprob": -0.1},
                        {"token": "token_id:13", "logprob": -2.0},
                    ]
                }
            ],
        )


def test_serving_constraint_rows_ignore_vllm_masked_sentinels() -> None:
    _verify_serving_constraint_rows(
        [11],
        [[11, 12]],
        [
            {
                "top_logprobs": [
                    {"token": "token_id:11", "logprob": -0.1},
                    {"token": "token_id:12", "logprob": -2.0},
                    {"token": "token_id:13", "logprob": -9999.0},
                ]
            }
        ],
    )


def test_vllm_generator_coalesces_exact_requests_within_one_group() -> None:
    config = EpisodeConfig(
        horizon=2,
        communication_cost=0.0,
        invalid_broadcast_cost=0.0,
        invalid_action_cost=0.0,
    )
    env = ArenaRLEnv(seed=13, size=12, config=config)
    env.reset(13)
    messages, _ = episode_broadcast_prompt(env, "blue-0", permutation=0)
    expected = protocol_choices(messages)[0]

    class FakeTokenizer:
        eos_token_id = 0

        def decode(self, token_ids, *, skip_special_tokens):
            del token_ids, skip_special_tokens
            return expected

        def encode(self, text, *, add_special_tokens):
            del text, add_special_tokens
            return [11]

    class FakeRenderer:
        def render_ids(self, rendered_messages, *, add_generation_prompt):
            del rendered_messages, add_generation_prompt
            return [1, 2, 3]

    class FakeChoiceMask:
        def allowed_token_ids(self, choices, completion_ids):
            assert choices[0] == expected
            assert completion_ids == [11]
            return [[11]]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "token_ids": [11],
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "token_id:11", "logprob": -0.1}
                                    ],
                                }
                            ]
                        },
                    }
                ]
            }

    class FakeClient:
        def __init__(self):
            self.posts = 0

        async def post(self, path, *, json):
            assert path == "/inference/v1/generate"
            sampling = json["sampling_params"]
            assert sampling["temperature"] == 1.0
            assert sampling["top_p"] == 1.0
            assert sampling["top_k"] == 0
            assert sampling["min_p"] == 0.0
            assert sampling["logprobs"] == 20
            assert sampling["structured_outputs"]["choice"][0] == expected
            self.posts += 1
            await asyncio.sleep(0)
            return FakeResponse()

    generator = object.__new__(VLLMChoiceGenerator)
    generator.tokenizer = FakeTokenizer()
    generator.renderer = FakeRenderer()
    generator.choice_mask = FakeChoiceMask()
    generator.timeout = 1.0
    client = FakeClient()
    generator._clients = {"http://fake": client}
    generator._group_requests = None
    endpoint = PolicyEndpoint("blue-policy-0", "revision-1", "model", ("http://fake",))

    async def exercise() -> None:
        async with generator.coalesced_request_group():
            first, second = await asyncio.gather(
                generator.generate(endpoint, messages, sampling_key="shared-key"),
                generator.generate(endpoint, messages, sampling_key="shared-key"),
            )
            assert first == second
            assert client.posts == 1
        assert generator._group_requests is None
        async with generator.coalesced_request_group():
            await generator.generate(endpoint, messages, sampling_key="shared-key")
        assert client.posts == 2

    asyncio.run(exercise())


def test_vllm_generator_retries_only_an_identical_transport_request() -> None:
    class FakeTokenizer:
        eos_token_id = 0

        def decode(self, token_ids, *, skip_special_tokens):
            del token_ids, skip_special_tokens
            return "WAIT"

    class FakeChoiceMask:
        def allowed_token_ids(self, choices, completion_ids):
            assert choices == ("WAIT",)
            assert completion_ids == [11]
            return [[11]]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "token_ids": [11],
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {
                                    "logprob": 0.0,
                                    "top_logprobs": [
                                        {"token": "token_id:11", "logprob": 0.0}
                                    ],
                                }
                            ]
                        },
                    }
                ]
            }

    class FlakyClient:
        def __init__(self):
            self.bodies = []

        async def post(self, path, *, json):
            assert path == "/inference/v1/generate"
            self.bodies.append(json)
            if len(self.bodies) == 1:
                raise httpx.ReadError("connection reset")
            return FakeResponse()

    generator = object.__new__(VLLMChoiceGenerator)
    generator.tokenizer = FakeTokenizer()
    generator.choice_mask = FakeChoiceMask()
    generator.timeout = 1.0
    client = FlakyClient()
    generator._clients = {"http://fake": client}

    async def exercise() -> None:
        completion = await generator._complete_request(
            base_url="http://fake",
            request_body={"seed": 17},
            choices=("WAIT",),
            prompt_ids=(1, 2, 3),
            request_sha256="a" * 64,
        )
        assert completion.transport_attempts == 2
        assert client.bodies == [{"seed": 17}, {"seed": 17}]

    asyncio.run(exercise())
