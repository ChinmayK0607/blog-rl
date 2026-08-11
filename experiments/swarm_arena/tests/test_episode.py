from __future__ import annotations

import json

from swarm_ctf_eval.arena import WAIT, Action
from swarm_ctf_eval.arena_protocol import Broadcast
from swarm_ctf_eval.episode import EMPTY_BROADCAST, ArenaEpisodeEnv, EpisodeConfig, message_units
from swarm_ctf_eval.episode_model_eval import evaluate_episode
from swarm_ctf_eval.episode_splits import EPISODE_EVAL_CASES


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
