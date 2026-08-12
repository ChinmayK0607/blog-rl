from __future__ import annotations

import json
from pathlib import Path

from swarm_ctf_eval.arena import WAIT, Action
from swarm_ctf_eval.arena_protocol import Broadcast
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
from swarm_ctf_eval.episode_protocol import episode_action_prompt, episode_broadcast_prompt
from swarm_ctf_eval.episode_splits import EPISODE_EVAL_CASES
from swarm_ctf_eval.structured_protocol import action_json_schema, broadcast_json_schema


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
        FirstOptionModel("blue"),
        FirstOptionModel("red"),
        development_cases(1)[0],
    )
    assert row["metrics"]["BLUE"]["action_protocol_rate"] == 1.0
    assert row["metrics"]["RED"]["action_protocol_rate"] == 1.0
    assert all(len(turn["actions"]) == 8 for turn in row["turns"])
    assert parse_conditions("generated:generated,dropped:generated") == (
        ("generated", "generated"),
        ("dropped", "generated"),
    )
    assert len(FROZEN_CROSSPLAY_CASES) == 24
    assert {size for _, size, _ in FROZEN_CROSSPLAY_CASES} == {14, 16}
    assert {horizon for _, _, horizon in FROZEN_CROSSPLAY_CASES} == {6, 8}


def test_side_swapped_summary_removes_map_side_bias() -> None:
    def row(seed: int, blue: str, red: str, blue_return: float, condition: str) -> dict:
        return {
            "seed": seed,
            "blue_model": blue,
            "red_model": red,
            "blue_condition": condition if blue == "adapter" else "generated",
            "red_condition": condition if red == "adapter" else "generated",
            "metrics": {
                "BLUE": {"terminal_return": blue_return},
                "RED": {"terminal_return": -blue_return},
            },
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
