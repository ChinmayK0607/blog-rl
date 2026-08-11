from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_ctf_eval.arena import Action, GameState, legal_actions, state_to_dict, step, team_value
from swarm_ctf_eval.arena_oracle import deterministic_policy
from swarm_ctf_eval.episode import EMPTY_BROADCAST, ArenaEpisodeEnv, EpisodeConfig

SEED = 0
SIZE = 10
HORIZON = 2
TEAM = "BLUE"
OPPONENT_STYLE = "balanced"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results/replay/seed0-full-match.json"
TOLERANCE = 1e-12

SerializedTurn = tuple[tuple[str, Action], ...]
Trajectory = tuple[SerializedTurn, ...]

ROLE_PROPOSALS: dict[int, dict[str, dict[str, Any]]] = {
    0: {
        "blue-0": {
            "broadcast_rationale": "With resource 1 and no message budget, blue-0 can capture adjacent neutral critical node V43 without broadcasting.",
            "action_id": "A0",
            "action_rationale": "Capturing adjacent neutral critical node V43 targets the highest observed value with the available resource.",
        },
        "blue-1": {
            "broadcast_rationale": "Fortifying V71 uses an available action to strengthen blue-1's secure position while preserving the no-broadcast constraint.",
            "action_id": "A0",
            "action_rationale": "Fortifying V71 strengthens blue-1's secure position using an available legal action.",
        },
        "blue-2": {
            "broadcast_rationale": "With no message budget, I cannot broadcast information about the unknown neighbor V55.",
            "action_id": "A1",
            "action_rationale": "Scanning V55 reveals information about the only unknown neighboring node.",
        },
        "blue-3": {
            "broadcast_rationale": "No broadcast is possible because the message budget is zero.",
            "action_id": "A0",
            "action_rationale": "Capturing the adjacent red secure node uses the available resource advantage.",
        },
        "red-0": {
            "broadcast_rationale": "I cannot send a non-empty broadcast because the message budget is zero.",
            "action_id": "A0",
            "action_rationale": "Capturing the known neutral critical node V43 directly contests its value using my available resources.",
        },
        "red-1": {
            "broadcast_rationale": "With no message budget, I keep the broadcast empty while preserving my sole resource for the action phase.",
            "action_id": "A2",
            "action_rationale": "Fortifying my exposed red node is the safest locally supported use of my single resource.",
        },
        "red-2": {
            "broadcast_rationale": "With no message budget, I will preserve my resource and assess the best legal move privately.",
            "action_id": "A2",
            "action_rationale": "Probing the adjacent fortified enemy reveals useful information without risking a likely unfavorable capture.",
        },
        "red-3": {
            "broadcast_rationale": "No broadcast is legal with a zero message budget.",
            "action_id": "A0",
            "action_rationale": "Capturing the adjacent enemy node offers the strongest immediate gain with only two turns remaining.",
        },
    },
    1: {
        "blue-0": {
            "broadcast_rationale": "With zero message budget, blue-0 cannot broadcast information or intent.",
            "action_id": "A0",
            "action_rationale": "Scanning unknown neighbor V15 gathers local information while blue-0 has no resource for capture actions.",
        },
        "blue-1": {
            "broadcast_rationale": "With no message budget, blue-1 cannot broadcast the observed exposed red node V63.",
            "action_id": "A0",
            "action_rationale": "Capturing the exposed red node V63 is a legal action with one turn remaining.",
        },
        "blue-2": {
            "broadcast_rationale": "The zero message budget prevents broadcasting the observed RED node V55 or threatened BLUE nodes.",
            "action_id": "A0",
            "action_rationale": "Capturing the adjacent secure RED node V55 can convert its value for BLUE on the final turn.",
        },
        "blue-3": {
            "broadcast_rationale": "No broadcast is possible because the remaining message budget is zero.",
            "action_id": "A0",
            "action_rationale": "The final turn is best used capturing the exposed red node with the available resource.",
        },
        "red-0": {
            "broadcast_rationale": "I cannot send a non-empty broadcast because the message budget is zero.",
            "action_id": "A0",
            "action_rationale": "With one turn remaining and two resources, I will attempt to capture the adjacent blue-held critical node V43.",
        },
        "red-1": {
            "broadcast_rationale": "With zero message budget, I cannot send a non-empty broadcast despite observing V71 exposed.",
            "action_id": "A0",
            "action_rationale": "Scanning the unknown neighboring node provides more local information than waiting on the final turn.",
        },
        "red-2": {
            "broadcast_rationale": "With no message budget or resources, I cannot communicate and will choose the most informative legal action privately.",
            "action_id": "A1",
            "action_rationale": "Scanning V71 is more informative because V59 is already known to contain a teammate and V71 was just exposed by red-1.",
        },
        "red-3": {
            "broadcast_rationale": "No broadcast is legal because the message budget remains zero.",
            "action_id": "A0",
            "action_rationale": "Scanning the unknown neighbor is the only action that can produce useful information on the final turn.",
        },
    },
}


@dataclass(frozen=True)
class Certificate:
    terminal_return: float
    trajectory: Trajectory
    optimal_trajectory_count: int
    first_turn_joint_actions: int
    second_turn_joint_actions: int

    @property
    def total_joint_actions(self) -> int:
        return self.first_turn_joint_actions + self.second_turn_joint_actions


def blue_joint_actions(state: GameState) -> itertools.product[tuple[Action, ...]]:
    members = sorted(agent.id for agent in state.agents.values() if agent.team == TEAM)
    return itertools.product(*(legal_actions(state, agent_id) for agent_id in members))


def serialize_turn(state: GameState, combination: tuple[Action, ...]) -> SerializedTurn:
    members = sorted(agent.id for agent in state.agents.values() if agent.team == TEAM)
    return tuple(zip(members, combination, strict=True))


def trajectory_key(trajectory: Trajectory) -> tuple[int, tuple[Action, ...]]:
    actions = tuple(action for turn in trajectory for _, action in turn)
    return sum(action.kind != "WAIT" for action in actions), actions


def regenerate_resources(state: GameState, period: int) -> None:
    if state.turn % period != 0:
        return
    for agent in state.agents.values():
        agent.resource = min(4, agent.resource + 1)
    state.validate()


def terminal_blue_return(state: GameState, initial_value: float) -> float:
    # Communication and invalid-action terms are zero in this certificate:
    # broadcasts are EMPTY and only enumerated legal actions are submitted.
    return team_value(state, TEAM) - initial_value


def solve_certificate() -> Certificate:
    """Exhaustively solve the minimum legal no-communication episode.

    RED is a fixed state-dependent deterministic policy. This is exact backward
    induction for BLUE's centralized best response, not minimax or equilibrium.
    A zero message budget makes EMPTY_BROADCAST the only non-penalized broadcast,
    so enumerating legal action trajectories covers every potentially optimal
    BLUE trajectory in both phases of the configured episode.
    """

    config = EpisodeConfig(horizon=HORIZON, message_budget_per_agent=0)
    env = ArenaEpisodeEnv(SEED, SIZE, config)
    env.reset()
    assert env.state is not None
    initial_state = env.state
    initial_value = env.initial_values[TEAM]
    red_first = deterministic_policy(initial_state, "RED", OPPONENT_STYLE)

    best_value = float("-inf")
    canonical: Trajectory | None = None
    optimal_count = 0
    first_explored = 0
    second_explored = 0

    for first_combination in blue_joint_actions(initial_state):
        first_explored += 1
        first_turn = serialize_turn(initial_state, first_combination)
        first_result = step(initial_state, {**red_first, **dict(first_turn)})
        state_after_first = first_result.state

        blue_nodes = sum(node.owner == "BLUE" for node in state_after_first.nodes.values())
        red_nodes = sum(node.owner == "RED" for node in state_after_first.nodes.values())
        if blue_nodes == 0 or red_nodes == 0:
            value = terminal_blue_return(state_after_first, initial_value)
            trajectory = (first_turn,)
            candidates = ((value, trajectory),)
        else:
            red_second = deterministic_policy(state_after_first, "RED", OPPONENT_STYLE)
            candidates_list: list[tuple[float, Trajectory]] = []
            for second_combination in blue_joint_actions(state_after_first):
                second_explored += 1
                second_turn = serialize_turn(state_after_first, second_combination)
                final_state = step(state_after_first, {**red_second, **dict(second_turn)}).state
                regenerate_resources(final_state, config.resource_regen_period)
                candidates_list.append((terminal_blue_return(final_state, initial_value), (first_turn, second_turn)))
            candidates = candidates_list

        for value, trajectory in candidates:
            if value > best_value + TOLERANCE:
                best_value = value
                canonical = trajectory
                optimal_count = 1
            elif abs(value - best_value) <= TOLERANCE:
                optimal_count += 1
                assert canonical is not None
                if trajectory_key(trajectory) < trajectory_key(canonical):
                    canonical = trajectory

    assert canonical is not None
    return Certificate(best_value, canonical, optimal_count, first_explored, second_explored)


def serialize_legal_actions(state: GameState) -> dict[str, list[dict[str, Any]]]:
    return {
        agent_id: [
            dict(action.to_dict(), id=f"A{index}") for index, action in enumerate(legal_actions(state, agent_id))
        ]
        for agent_id in sorted(state.agents)
    }


def action_from_dict(value: dict[str, Any]) -> Action:
    return Action(value["type"], value.get("target"), value.get("amount"))


def replay_certificate(certificate: Certificate) -> dict[str, Any]:
    config = EpisodeConfig(horizon=HORIZON, message_budget_per_agent=0)
    env = ArenaEpisodeEnv(SEED, SIZE, config)
    initial_observations = env.reset()
    assert env.state is not None
    initial_state = state_to_dict(env.state)
    initial_team_value = {team: team_value(env.state, team) for team in ("BLUE", "RED")}
    turns: list[dict[str, Any]] = []

    for index, planned_turn in enumerate(certificate.trajectory):
        assert env.state is not None
        pre_state = state_to_dict(env.state)
        pre_observations = env.observations()
        legal_before = serialize_legal_actions(env.state)

        phase = env.broadcast_phase({})
        assert all(message == EMPTY_BROADCAST for message in phase.accepted.values())
        assert all(not errors for errors in phase.errors.values())
        assert all(units == 0 for units in phase.message_units.values())
        action_observations = env.action_observations()
        legal_after = serialize_legal_actions(env.state)

        blue_actions = dict(planned_turn)
        for agent_id, action in blue_actions.items():
            assert action in legal_actions(env.state, agent_id)
        red_actions = deterministic_policy(env.state, "RED", OPPONENT_STYLE)
        executed = {**blue_actions, **red_actions}
        role_action_proposals: dict[str, dict[str, Any]] = {}
        for agent, proposal in ROLE_PROPOSALS[index].items():
            action_index = int(proposal["action_id"][1:])
            displayed = legal_after[agent]
            assert 0 <= action_index < len(displayed)
            selected = {key: value for key, value in displayed[action_index].items() if key != "id"}
            role_action_proposals[agent] = {
                "action_id": proposal["action_id"],
                "action": selected,
                "matches_executed": selected == executed[agent].to_dict(),
                "private_rationale": proposal["action_rationale"],
            }
        transition = env.advance(executed)
        assert env.state is not None

        turns.append(
            {
                "index": index,
                "pre": {
                    "state": pre_state,
                    "observations": pre_observations,
                    "legal_actions": legal_before,
                },
                "broadcast": {
                    "policy": "empty_broadcast",
                    "role_proposals": {
                        agent: {
                            "broadcast": EMPTY_BROADCAST.to_dict(),
                            "private_rationale": proposal["broadcast_rationale"],
                        }
                        for agent, proposal in ROLE_PROPOSALS[index].items()
                    },
                    "accepted": {agent: message.to_dict() for agent, message in phase.accepted.items()},
                    "delivered": {agent: message.to_dict() for agent, message in phase.delivered.items()},
                    "errors": {agent: list(errors) for agent, errors in phase.errors.items()},
                    "message_units": phase.message_units,
                    "remaining_budget": phase.remaining_budget,
                    "shared_fact_updates": phase.shared_fact_updates,
                    "inboxes": {agent: list(inbox) for agent, inbox in phase.inboxes.items()},
                },
                "act": {
                    "observations": action_observations,
                    "legal_actions": legal_after,
                    "role_proposals": role_action_proposals,
                    "executed": {agent: action.to_dict() for agent, action in sorted(executed.items())},
                    "blue": {agent: action.to_dict() for agent, action in sorted(blue_actions.items())},
                    "red": {agent: action.to_dict() for agent, action in sorted(red_actions.items())},
                },
                "resolution": {
                    "events": transition.info["events"],
                    "duplicate_targets": transition.info["duplicate_targets"],
                },
                "post": {
                    "state": state_to_dict(env.state),
                    "observations": transition.observations,
                    "team_value": transition.info["team_value"],
                    "rewards": transition.rewards,
                    "communication_spend": transition.info["communication_spend"],
                    "invalid_broadcasts": transition.info["invalid_broadcasts"],
                    "invalid_actions": transition.info["invalid_actions"],
                    "terminated": transition.terminated,
                    "truncated": transition.truncated,
                },
            }
        )

    final = turns[-1]["post"]
    assert final["terminated"] or final["truncated"]
    assert abs(final["rewards"][TEAM] - certificate.terminal_return) <= TOLERANCE
    return {
        "schema_version": "swarm-arena-full-match-certificate-v1",
        "episode": {
            "episode_version": "arena-episode-v2",
            "seed": SEED,
            "size": SIZE,
            "horizon": HORIZON,
            "config": {
                "message_budget_per_agent": config.message_budget_per_agent,
                "max_facts_per_message": config.max_facts_per_message,
                "communication_cost": config.communication_cost,
                "invalid_broadcast_cost": config.invalid_broadcast_cost,
                "invalid_action_cost": config.invalid_action_cost,
                "resource_regen_period": config.resource_regen_period,
            },
        },
        "optimality": {
            "definition": (
                "Exact centralized BLUE best response over every legal joint-action trajectory "
                "against state-dependent deterministic balanced RED. With message budget zero, "
                "EMPTY_BROADCAST is the only non-penalized broadcast."
            ),
            "terminal_blue_return": certificate.terminal_return,
            "optimal_trajectory_count": certificate.optimal_trajectory_count,
            "unique_optimum": certificate.optimal_trajectory_count == 1,
            "explored": {
                "first_turn_blue_joint_actions": certificate.first_turn_joint_actions,
                "reachable_second_turn_blue_joint_actions": certificate.second_turn_joint_actions,
                "total_blue_joint_actions": certificate.total_joint_actions,
            },
            "tie_break": "fewest non-WAIT actions, then lexicographic Action tuple",
            "limitation": (
                "This is not minimax, Nash equilibrium, communication-policy optimality at a positive "
                "budget, or an observation-respecting decentralized-policy certificate. RED is fixed, "
                "and the BLUE planner reads the full GameState while selecting individually legal actions."
            ),
        },
        "role_analysis": {
            "requested_model_available": False,
            "requested_model": "gpt-5.6-luna",
            "actual_model": "gpt-5.6-sol",
            "reasoning_effort": "medium",
            "independent_roles": sorted(ROLE_PROPOSALS[0]),
            "method": (
                "Eight fresh role contexts were run in waves. Each received only that agent's current private "
                "observation, legal actions, empty teammate inbox, and its own prior private role history."
            ),
            "note": "Role proposals are comparisons only; they are not called optimal or executed by the certificate.",
        },
        "initial": {
            "state": initial_state,
            "observations": initial_observations,
            "team_value": initial_team_value,
        },
        "turns": turns,
        "outcome": {
            "reason": "elimination" if final["terminated"] else "time_limit",
            "turns_played": len(turns),
            "terminal_return": final["rewards"],
            "final_team_value": final["team_value"],
            "communication_spend": final["communication_spend"],
            "invalid_broadcasts": final["invalid_broadcasts"],
            "invalid_actions": final["invalid_actions"],
        },
    }


def replay_hash(replay: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in replay.items() if key != "sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_replay(replay: dict[str, Any]) -> None:
    episode = replay["episode"]
    config = EpisodeConfig(horizon=episode["horizon"], **episode["config"])
    env = ArenaEpisodeEnv(episode["seed"], episode["size"], config)
    observations = env.reset()
    assert env.state is not None
    assert canonical_json(state_to_dict(env.state)) == canonical_json(replay["initial"]["state"])
    assert observations == replay["initial"]["observations"]

    for recorded in replay["turns"]:
        assert env.state is not None
        assert canonical_json(state_to_dict(env.state)) == canonical_json(recorded["pre"]["state"])
        phase = env.broadcast_phase({})
        assert {agent: message.to_dict() for agent, message in phase.accepted.items()} == recorded["broadcast"][
            "accepted"
        ]
        assert {agent: message.to_dict() for agent, message in phase.delivered.items()} == recorded["broadcast"][
            "delivered"
        ]
        assert {agent: list(errors) for agent, errors in phase.errors.items()} == recorded["broadcast"]["errors"]
        assert {agent: list(inbox) for agent, inbox in phase.inboxes.items()} == recorded["broadcast"]["inboxes"]
        assert env.action_observations() == recorded["act"]["observations"]
        assert serialize_legal_actions(env.state) == recorded["act"]["legal_actions"]
        actions = {agent: action_from_dict(action) for agent, action in recorded["act"]["executed"].items()}
        expected_red = deterministic_policy(env.state, "RED", OPPONENT_STYLE)
        assert {agent: action.to_dict() for agent, action in expected_red.items()} == recorded["act"]["red"]
        transition = env.advance(actions)
        assert canonical_json(state_to_dict(env.state)) == canonical_json(recorded["post"]["state"])
        assert transition.observations == recorded["post"]["observations"]
        assert transition.info["events"] == recorded["resolution"]["events"]
        assert transition.rewards == recorded["post"]["rewards"]
        assert transition.terminated == recorded["post"]["terminated"]
        assert transition.truncated == recorded["post"]["truncated"]

    assert replay["sha256"] == replay_hash(replay)
    assert env.communication_spend == {"BLUE": 0, "RED": 0}
    assert env.invalid_broadcasts == {"BLUE": 0, "RED": 0}
    assert env.invalid_actions == {"BLUE": 0, "RED": 0}
    assert replay["outcome"]["terminal_return"][TEAM] == replay["optimality"]["terminal_blue_return"]
    assert replay["optimality"]["unique_optimum"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an exact minimum full-match certificate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    certificate = solve_certificate()
    replay = replay_certificate(certificate)
    replay["sha256"] = replay_hash(replay)
    validate_replay(replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": replay["sha256"],
                "terminal_blue_return": certificate.terminal_return,
                "optimal_trajectory_count": certificate.optimal_trajectory_count,
                "unique_optimum": certificate.optimal_trajectory_count == 1,
                "explored": replay["optimality"]["explored"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
