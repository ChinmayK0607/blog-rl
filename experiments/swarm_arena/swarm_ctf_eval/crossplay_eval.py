from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .arena import TEAMS, WAIT, Action, GameState, NodeObservation, Team
from .arena_eval import ArenaModel, OpenAIArenaModel, _respond_many
from .arena_protocol import Broadcast, parse_action, parse_broadcast
from .episode import (
    EMPTY_BROADCAST,
    EPISODE_VERSION,
    ArenaEpisodeEnv,
    EpisodeConfig,
    message_units,
    validate_message,
)
from .episode_protocol import (
    EPISODE_PROMPT_VERSION,
    ActionPromptProfile,
    episode_action_prompt,
    episode_broadcast_prompt,
)
from .message_interventions import target_swapped_broadcast
from .providers import OpenAICompatibleProvider
from .structured_protocol import STRUCTURED_PROTOCOL_VERSION, protocol_response_format

CROSSPLAY_VERSION = "arena-crossplay-v5-source-prompt-bound"
CONDITIONS = (
    "generated",
    "dropped",
    "sender_shuffled",
    "delayed",
    "zero_budget",
    "target_swapped",
)
Case = tuple[int, int, int]
ModelRoster = ArenaModel | dict[str, ArenaModel]
FROZEN_CROSSPLAY_CASES: tuple[Case, ...] = tuple(
    (
        3_000_003 + 193 * index,
        14 if index < 12 else 16,
        6 if index % 2 == 0 else 8,
    )
    for index in range(24)
)


def development_cases(count: int, seed_base: int = 1_000_003) -> tuple[Case, ...]:
    if count < 1:
        raise ValueError("case count must be positive")
    return tuple(
        (seed_base + 193 * index, 12 if index % 2 == 0 else 13, 4 if index % 2 == 0 else 6)
        for index in range(count)
    )


def _members(env: ArenaEpisodeEnv, team: Team) -> list[str]:
    state = env._require_state()
    return sorted(agent.id for agent in state.agents.values() if agent.team == team)


def _resolve_roster(
    env: ArenaEpisodeEnv,
    team: Team,
    roster: ModelRoster,
) -> dict[str, ArenaModel]:
    members = _members(env, team)
    if not isinstance(roster, dict):
        return {agent_id: roster for agent_id in members}
    if set(roster) != set(members):
        raise ValueError(
            f"{team} model roster must cover exactly {members}; got {sorted(roster)}"
        )
    return dict(roster)


def _roster_name(roster: dict[str, ArenaModel]) -> str:
    names = [model.name for _, model in sorted(roster.items())]
    return names[0] if len(set(names)) == 1 else "roster[" + ",".join(names) + "]"


def _referee_state(env: ArenaEpisodeEnv) -> dict[str, Any]:
    """Compact global snapshot for deterministic replay and post-hoc auditing."""
    state = env._require_state()
    return {
        "turn": state.turn,
        "nodes": {
            node_id: {
                "neighbors": list(node.neighbors),
                "owner": node.owner,
                "value": node.value,
                "critical": node.critical,
                "fortification": node.fortification,
                "exposed": node.exposed,
                "compromised": node.compromised,
            }
            for node_id, node in sorted(state.nodes.items())
        },
        "agents": {
            agent_id: {
                "team": agent.team,
                "position": agent.position,
                "resource": agent.resource,
            }
            for agent_id, agent in sorted(state.agents.items())
        },
    }


def _with_history(messages: list[dict[str, str]], history: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not history:
        return messages
    body = json.loads(messages[-1]["content"])
    body["private_history"] = history
    return [*messages[:-1], {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"))}]


def _respond_agents(
    jobs: list[tuple[str, ArenaModel, list[dict[str, str]]]],
) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[int, tuple[ArenaModel, list[tuple[str, list[dict[str, str]]]]]] = {}
    for agent_id, model, prompt in jobs:
        key = id(model)
        if key not in grouped:
            grouped[key] = (model, [])
        grouped[key][1].append((agent_id, prompt))

    def run_group(
        group: tuple[ArenaModel, list[tuple[str, list[dict[str, str]]]]]
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        model, entries = group
        started = time.perf_counter()
        responses = _respond_many(model, [prompt for _, prompt in entries], ["{}"] * len(entries))
        elapsed = time.perf_counter() - started
        stats = getattr(model, "last_batch_stats", None)
        if not isinstance(stats, dict):
            stats = {
                "requests": len(entries),
                "wall_seconds": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "completion_tokens_per_second": None,
            }
        return (
            [(agent_id, response) for (agent_id, _), response in zip(entries, responses, strict=True)],
            {"model": model.name, **stats},
        )

    groups = list(grouped.values())
    started = time.perf_counter()
    if len(groups) == 1:
        results = [run_group(groups[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            results = list(executor.map(run_group, groups))
    elapsed = time.perf_counter() - started
    responses = {
        agent_id: response
        for entries, _ in results
        for agent_id, response in entries
    }
    group_stats = [stats for _, stats in results]
    known_completion_tokens = [stats["completion_tokens"] for stats in group_stats]
    completion_tokens = (
        sum(known_completion_tokens)
        if all(value is not None for value in known_completion_tokens)
        else None
    )
    metrics = {
        "requests": len(jobs),
        "wall_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": (
            completion_tokens / elapsed if completion_tokens is not None and elapsed else None
        ),
        "model_groups": group_stats,
    }
    return responses, metrics


def _preview_accepted(env: ArenaEpisodeEnv, messages: dict[str, Broadcast]) -> dict[str, Broadcast]:
    state = env._require_state()
    accepted = {}
    for agent_id in sorted(state.agents):
        message = messages.get(agent_id, EMPTY_BROADCAST)
        errors = validate_message(state, agent_id, message, max_facts=env.config.max_facts_per_message)
        if not errors and message_units(message) > env.remaining_budget[agent_id]:
            errors = ("message_budget_exceeded",)
        accepted[agent_id] = EMPTY_BROADCAST if errors else message
    return accepted


def _rotate(messages: dict[str, Broadcast], members: list[str]) -> dict[str, Broadcast]:
    return {
        agent_id: messages[members[(index + 1) % len(members)]]
        for index, agent_id in enumerate(members)
    }


def _delivered_for_team(
    condition: str,
    members: list[str],
    current: dict[str, Broadcast],
    previous: dict[str, Broadcast],
) -> dict[str, Broadcast]:
    if condition == "generated":
        return {agent_id: current[agent_id] for agent_id in members}
    if condition in {"dropped", "zero_budget"}:
        return {agent_id: EMPTY_BROADCAST for agent_id in members}
    if condition == "sender_shuffled":
        return _rotate(current, members)
    if condition == "delayed":
        return {agent_id: previous.get(agent_id, EMPTY_BROADCAST) for agent_id in members}
    if condition == "target_swapped":
        return {agent_id: current[agent_id] for agent_id in members}
    raise ValueError(f"unknown condition: {condition}")


def _with_required_fact(message: Broadcast, required: NodeObservation) -> Broadcast:
    if any(fact.node == required.node for fact in message.facts):
        return message
    remaining = tuple(fact for fact in message.facts if fact.node != required.node)
    return Broadcast(
        (required, *remaining)[:3],
        message.intent,
        message.request_resource,
    )


def evaluate_crossplay(
    blue_model: ModelRoster,
    red_model: ModelRoster,
    case: Case,
    *,
    blue_condition: str = "generated",
    red_condition: str = "generated",
    blue_history_window: int = 3,
    red_history_window: int = 3,
    initial_state: GameState | None = None,
    env: ArenaEpisodeEnv | None = None,
    action_permutation_offset: int = 0,
    turn_zero_required_facts: dict[str, NodeObservation] | None = None,
    action_prompt_profiles: dict[str, ActionPromptProfile] | None = None,
    target_swap_interventions: dict[Team, tuple[str, tuple[str, str], str]] | None = None,
) -> dict[str, Any]:
    if blue_condition not in CONDITIONS or red_condition not in CONDITIONS:
        raise ValueError("unknown communication condition")
    if blue_history_window < 0 or red_history_window < 0:
        raise ValueError("history windows cannot be negative")

    seed, size, horizon = case
    if env is None:
        env = ArenaEpisodeEnv(seed, size, EpisodeConfig(horizon=horizon))
    elif env.config.horizon != horizon:
        raise ValueError("provided environment horizon does not match the case")
    if initial_state is None:
        env.reset(seed)
    else:
        if len(initial_state.nodes) != size:
            raise ValueError("provided initial state size does not match the case")
        env.reset_from_state(initial_state)
    blue_roster = _resolve_roster(env, "BLUE", blue_model)
    red_roster = _resolve_roster(env, "RED", red_model)
    models_by_agent = {**blue_roster, **red_roster}
    resolved_action_prompt_profiles = action_prompt_profiles or {}
    unknown_profile_agents = set(resolved_action_prompt_profiles) - set(models_by_agent)
    if unknown_profile_agents:
        raise ValueError(
            "action prompt profiles name unknown agents: "
            f"{sorted(unknown_profile_agents)}"
        )
    conditions: dict[Team, str] = {"BLUE": blue_condition, "RED": red_condition}
    history_windows: dict[Team, int] = {
        "BLUE": blue_history_window,
        "RED": red_history_window,
    }
    histories: dict[str, list[dict[str, Any]]] = {
        agent_id: [] for agent_id in env._require_state().agents
    }
    previous_accepted: dict[str, Broadcast] = {}
    attempts = {
        team: {"broadcast": 0, "broadcast_protocol": 0, "broadcast_grounded": 0, "action": 0, "action_protocol": 0}
        for team in TEAMS
    }
    turns = []
    final = None
    initial_state = _referee_state(env)

    for turn in range(horizon):
        state = env._require_state()
        pre_state = _referee_state(env)
        broadcast_jobs = []
        broadcast_prompts: dict[str, list[dict[str, str]]] = {}
        for team in TEAMS:
            if conditions[team] == "zero_budget":
                continue
            for index, agent_id in enumerate(_members(env, team)):
                prompt, _ = episode_broadcast_prompt(env, agent_id, seed + turn * 19 + index)
                window = history_windows[team]
                prompt = _with_history(prompt, histories[agent_id][-window:] if window else [])
                broadcast_prompts[agent_id] = prompt
                broadcast_jobs.append((agent_id, models_by_agent[agent_id], prompt))
        if broadcast_jobs:
            raw_broadcasts, broadcast_inference = _respond_agents(broadcast_jobs)
        else:
            raw_broadcasts = {}
            broadcast_inference = {
                "requests": 0,
                "wall_seconds": 0.0,
                "completion_tokens": 0,
                "completion_tokens_per_second": None,
                "model_groups": [],
            }

        parsed_messages: dict[str, Broadcast] = {}
        broadcast_rows = []
        for agent_id in sorted(state.agents):
            team = state.agents[agent_id].team
            queried = agent_id in raw_broadcasts
            raw = raw_broadcasts.get(agent_id)
            if queried:
                assert raw is not None
                parsed = parse_broadcast(raw, state, agent_id)
                message = parsed.value if parsed.valid else EMPTY_BROADCAST
                assert isinstance(message, Broadcast)
                protocol_valid: bool | None = parsed.valid
                protocol_errors = list(parsed.errors)
                attempts[team]["broadcast"] += 1
                attempts[team]["broadcast_protocol"] += int(parsed.valid)
            else:
                if conditions[team] != "zero_budget":
                    raise RuntimeError(f"missing broadcast response for {agent_id}")
                message = EMPTY_BROADCAST
                protocol_valid = None
                protocol_errors = []
            parsed_messages[agent_id] = message
            broadcast_rows.append(
                {
                    "agent_id": agent_id,
                    "team": team,
                    "condition": conditions[team],
                    "queried": queried,
                    "prompt_messages": broadcast_prompts.get(agent_id),
                    "raw_response": raw,
                    "protocol_valid": protocol_valid,
                    "protocol_errors": protocol_errors,
                    "parsed_message": message.to_dict(),
                }
            )

        if turn == 0 and turn_zero_required_facts:
            unknown = set(turn_zero_required_facts) - set(parsed_messages)
            if unknown:
                raise ValueError(f"turn-zero required facts name unknown agents: {sorted(unknown)}")
            for agent_id, required in turn_zero_required_facts.items():
                parsed_messages[agent_id] = _with_required_fact(
                    parsed_messages[agent_id], required
                )
            for row in broadcast_rows:
                required = turn_zero_required_facts.get(row["agent_id"])
                row["required_fact"] = (
                    {
                        "node": required.node,
                        "owner": required.owner,
                        "status": required.status,
                        "value": required.value,
                        "critical": required.critical,
                        "observed_turn": required.observed_turn,
                    }
                    if required is not None
                    else None
                )

        preview = _preview_accepted(env, parsed_messages)
        delivered: dict[str, Broadcast] = {}
        for team in TEAMS:
            members = _members(env, team)
            team_delivered = _delivered_for_team(
                conditions[team], members, preview, previous_accepted
            )
            if conditions[team] == "target_swapped":
                if target_swap_interventions is None or team not in target_swap_interventions:
                    raise ValueError("target-swapped condition requires a certified intervention")
                sender, targets, active_target = target_swap_interventions[team]
                if sender not in team_delivered:
                    raise ValueError("target-swapped condition names an unknown sender")
                team_delivered[sender] = target_swapped_broadcast(
                    team_delivered[sender],
                    candidate_targets=targets,
                    active_target=active_target,
                )
            delivered.update(team_delivered)
        phase = env.broadcast_phase(parsed_messages, delivered_broadcasts=delivered)
        previous_accepted = dict(phase.accepted)
        for row in broadcast_rows:
            agent_id = row["agent_id"]
            team = row["team"]
            row["environment_errors"] = list(phase.errors[agent_id])
            row["message_units"] = phase.message_units[agent_id]
            row["accepted_message"] = phase.accepted[agent_id].to_dict()
            row["delivered_message"] = phase.delivered[agent_id].to_dict()
            row["remaining_budget"] = phase.remaining_budget[agent_id]
            if row["queried"]:
                attempts[team]["broadcast_grounded"] += int(not phase.errors[agent_id])

        action_jobs = []
        displayed_by_agent: dict[str, tuple[Action, ...]] = {}
        action_prompts: dict[str, list[dict[str, str]]] = {}
        for team in TEAMS:
            for index, agent_id in enumerate(_members(env, team)):
                prompt, displayed = episode_action_prompt(
                    env,
                    agent_id,
                    permutation=seed + turn * 23 + index + action_permutation_offset,
                    profile=resolved_action_prompt_profiles.get(agent_id, "full"),
                )
                window = history_windows[team]
                prompt = _with_history(prompt, histories[agent_id][-window:] if window else [])
                displayed_by_agent[agent_id] = displayed
                action_prompts[agent_id] = prompt
                action_jobs.append((agent_id, models_by_agent[agent_id], prompt))
        raw_actions, action_inference = _respond_agents(action_jobs)

        selected: dict[str, Action] = {}
        action_rows = []
        for agent_id, raw in sorted(raw_actions.items()):
            team = state.agents[agent_id].team
            parsed = parse_action(raw, displayed_by_agent[agent_id])
            action = parsed.value if parsed.valid else WAIT
            assert isinstance(action, Action)
            selected[agent_id] = action
            attempts[team]["action"] += 1
            attempts[team]["action_protocol"] += int(parsed.valid)
            action_rows.append(
                {
                    "agent_id": agent_id,
                    "team": team,
                    "prompt_messages": action_prompts[agent_id],
                    "raw_response": raw,
                    "protocol_valid": parsed.valid,
                    "protocol_errors": list(parsed.errors),
                    "displayed_legal_actions": [
                        {"action_id": f"A{index}", **action.to_dict()}
                        for index, action in enumerate(displayed_by_agent[agent_id])
                    ],
                    "selected_action": action.to_dict(),
                }
            )

        final = env.advance(selected)
        for agent_id in sorted(selected):
            histories[agent_id].append(
                {
                    "turn": turn,
                    "accepted_broadcast": phase.accepted[agent_id].to_dict(),
                    "received_broadcasts": list(phase.inboxes[agent_id]),
                    "selected_action": selected[agent_id].to_dict(),
                    "local_events": final.observations[agent_id]["last_local_events"],
                }
            )
        turns.append(
            {
                "turn": turn,
                "broadcasts": broadcast_rows,
                "actions": action_rows,
                "shared_fact_updates": phase.shared_fact_updates,
                "events": final.info["events"],
                "duplicate_targets": final.info["duplicate_targets"],
                "team_value": final.info["team_value"],
                "pre_state": pre_state,
                "post_state": _referee_state(env),
                "inference": {
                    "broadcast": broadcast_inference,
                    "action": action_inference,
                },
            }
        )
        if final.terminated or final.truncated:
            break

    assert final is not None and (final.terminated or final.truncated)

    def rate(team: Team, numerator: str, denominator: str) -> float | None:
        total = attempts[team][denominator]
        return attempts[team][numerator] / total if total else None

    metrics = {}
    for team in TEAMS:
        metrics[team] = {
            "terminal_return": final.rewards[team],
            "broadcast_protocol_rate": rate(team, "broadcast_protocol", "broadcast"),
            "broadcast_grounded_rate": rate(team, "broadcast_grounded", "broadcast"),
            "action_protocol_rate": rate(team, "action_protocol", "action"),
            "communication_spend": final.info["communication_spend"][team],
            "invalid_broadcasts": final.info["invalid_broadcasts"][team],
            "invalid_actions": final.info["invalid_actions"][team],
            "duplicate_target_turn_rate": statistics.mean(
                bool(row["duplicate_targets"][team]) for row in turns
            ),
        }
    inference_seconds = sum(
        turn["inference"][phase]["wall_seconds"]
        for turn in turns
        for phase in ("broadcast", "action")
    )
    completion_values = [
        turn["inference"][phase]["completion_tokens"]
        for turn in turns
        for phase in ("broadcast", "action")
    ]
    completion_tokens = (
        sum(completion_values) if all(value is not None for value in completion_values) else None
    )
    return {
        "crossplay_version": CROSSPLAY_VERSION,
        "episode_version": EPISODE_VERSION,
        "prompt_version": EPISODE_PROMPT_VERSION,
        "seed": seed,
        "size": size,
        "horizon": horizon,
        "blue_model": _roster_name(blue_roster),
        "red_model": _roster_name(red_roster),
        "blue_agent_models": {
            agent_id: model.name for agent_id, model in sorted(blue_roster.items())
        },
        "red_agent_models": {
            agent_id: model.name for agent_id, model in sorted(red_roster.items())
        },
        "blue_condition": blue_condition,
        "red_condition": red_condition,
        "blue_history_window": blue_history_window,
        "red_history_window": red_history_window,
        "initial_team_value": dict(env.initial_values),
        "initial_state": initial_state,
        "inference": {
            "requests": sum(
                turn["inference"][phase]["requests"]
                for turn in turns
                for phase in ("broadcast", "action")
            ),
            "wall_seconds": inference_seconds,
            "completion_tokens": completion_tokens,
            "completion_tokens_per_second": (
                completion_tokens / inference_seconds
                if completion_tokens is not None and inference_seconds
                else None
            ),
        },
        "metrics": metrics,
        "turns": turns,
    }


def _mean_ci(values: list[float], *, trials: int = 20_000, seed: int = 0) -> list[float]:
    """Deterministic non-parametric bootstrap interval for a seed-level mean."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    generator = random.Random(seed)
    size = len(values)
    samples = sorted(
        statistics.mean(values[generator.randrange(size)] for _ in range(size))
        for _ in range(trials)
    )
    return [samples[int(0.025 * (trials - 1))], samples[int(0.975 * (trials - 1))]]


def _paired_randomization_p(values: list[float], *, trials: int = 100_000, seed: int = 0) -> float:
    """Two-sided paired sign-randomization test; exact for up to 16 seeds."""
    observed = abs(statistics.mean(values))
    if observed == 0:
        return 1.0
    if len(values) <= 16:
        total = 1 << len(values)
        extreme = sum(
            abs(
                statistics.mean(
                    value if mask & (1 << index) else -value
                    for index, value in enumerate(values)
                )
            )
            >= observed - 1e-12
            for mask in range(total)
        )
        return extreme / total
    generator = random.Random(seed)
    extreme = sum(
        abs(statistics.mean(value if generator.getrandbits(1) else -value for value in values))
        >= observed - 1e-12
        for _ in range(trials)
    )
    return (extreme + 1) / (trials + 1)


def summarize(rows: list[dict[str, Any]], manifest_sha256: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("cross-play rows cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['blue_condition']}:{row['red_condition']}"].append(row)
    conditions = {}
    for condition, group in sorted(grouped.items()):
        blue_returns = [float(row["metrics"]["BLUE"]["terminal_return"]) for row in group]
        conditions[condition] = {
            "matches": len(group),
            "blue_mean_return": statistics.mean(blue_returns),
            "blue_mean_return_95": _mean_ci(blue_returns),
            "blue_win_rate": statistics.mean(value > 0 for value in blue_returns),
            "draw_rate": statistics.mean(value == 0 for value in blue_returns),
            "metrics": {
                team: {
                    key: statistics.mean(
                        row["metrics"][team][key]
                        for row in group
                        if row["metrics"][team][key] is not None
                    )
                    for key in (
                        "broadcast_protocol_rate",
                        "broadcast_grounded_rate",
                        "action_protocol_rate",
                        "communication_spend",
                        "invalid_broadcasts",
                        "invalid_actions",
                        "duplicate_target_turn_rate",
                    )
                    if any(row["metrics"][team][key] is not None for row in group)
                }
                for team in TEAMS
            },
        }
    return {
        "crossplay_version": CROSSPLAY_VERSION,
        "manifest_sha256": manifest_sha256,
        "blue_model": rows[0]["blue_model"],
        "red_model": rows[0]["red_model"],
        "matches": len(rows),
        "interval_method": "seed-level nonparametric bootstrap, 20000 deterministic resamples",
        "conditions": conditions,
    }


def summarize_side_swapped(
    rows: list[dict[str, Any]],
    focal_policy: str | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("side-swapped rows cannot be empty")
    policies = sorted({row["blue_model"] for row in rows} | {row["red_model"] for row in rows})
    if len(policies) != 2:
        raise ValueError("side-swapped summary requires exactly two distinct policies")
    focal = focal_policy or rows[0]["blue_model"]
    if focal not in policies:
        raise ValueError(f"focal policy is absent from rows: {focal}")
    opponent = next(policy for policy in policies if policy != focal)
    by_key: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if {row["blue_model"], row["red_model"]} != {focal, opponent}:
            raise ValueError("all rows must use the same policy pair")
        focal_condition = (
            row["blue_condition"] if row["blue_model"] == focal else row["red_condition"]
        )
        opponent_condition = (
            row["red_condition"] if row["blue_model"] == focal else row["blue_condition"]
        )
        orientation = "focal_blue" if row["blue_model"] == focal else "focal_red"
        key = (row["seed"], focal_condition, opponent_condition)
        if orientation in by_key[key]:
            raise ValueError(f"duplicate side assignment for {key}: {orientation}")
        by_key[key][orientation] = row

    paired = []
    metric_keys = (
        "broadcast_protocol_rate",
        "broadcast_grounded_rate",
        "action_protocol_rate",
        "communication_spend",
        "invalid_broadcasts",
        "invalid_actions",
        "duplicate_target_turn_rate",
    )
    for (seed, focal_condition, opponent_condition), orientations in sorted(by_key.items()):
        if set(orientations) != {"focal_blue", "focal_red"}:
            continue
        focal_blue = orientations["focal_blue"]
        focal_red = orientations["focal_red"]
        focal_return = statistics.mean(
            (
                float(focal_blue["metrics"]["BLUE"]["terminal_return"]),
                float(focal_red["metrics"]["RED"]["terminal_return"]),
            )
        )
        focal_metrics = {
            metric: statistics.mean(
                (
                    float(focal_blue["metrics"]["BLUE"][metric]),
                    float(focal_red["metrics"]["RED"][metric]),
                )
            )
            for metric in metric_keys
        }
        paired.append(
            {
                "seed": seed,
                "focal_condition": focal_condition,
                "opponent_condition": opponent_condition,
                "focal_side_averaged_return": focal_return,
                "focal_side_averaged_metrics": focal_metrics,
            }
        )
    if not paired:
        raise ValueError("no complete side-swapped pairs")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(row["focal_condition"], row["opponent_condition"])].append(row)
    conditions = {}
    for condition, group in sorted(grouped.items()):
        values = [float(row["focal_side_averaged_return"]) for row in group]
        conditions[":".join(condition)] = {
            "paired_seeds": len(group),
            "focal_mean_side_averaged_return": statistics.mean(values),
            "focal_mean_side_averaged_return_95": _mean_ci(values),
            "randomization_p_two_sided": _paired_randomization_p(values),
            "focal_metrics": {
                metric: statistics.mean(
                    float(row["focal_side_averaged_metrics"][metric]) for row in group
                )
                for metric in metric_keys
            },
        }

    effects = {}
    generated = {
        row["seed"]: float(row["focal_side_averaged_return"])
        for row in paired
        if row["focal_condition"] == "generated" and row["opponent_condition"] == "generated"
    }
    for intervention in ("dropped", "sender_shuffled", "delayed", "zero_budget"):
        intervened = {
            row["seed"]: float(row["focal_side_averaged_return"])
            for row in paired
            if row["focal_condition"] == intervention and row["opponent_condition"] == "generated"
        }
        common = sorted(set(generated) & set(intervened))
        if not common:
            continue
        differences = [generated[seed] - intervened[seed] for seed in common]
        effects[f"generated_minus_{intervention}"] = {
            "paired_seeds": len(common),
            "mean_return_difference": statistics.mean(differences),
            "mean_return_difference_95": _mean_ci(differences),
            "randomization_p_two_sided": _paired_randomization_p(differences),
            "positive_seed_rate": statistics.mean(value > 0 for value in differences),
            "seed_differences": [
                {"seed": seed, "return_difference": difference}
                for seed, difference in zip(common, differences, strict=True)
            ],
        }
    total_inference_seconds = sum(float(row["inference"]["wall_seconds"]) for row in rows)
    completion_values = [row["inference"]["completion_tokens"] for row in rows]
    total_completion_tokens = (
        sum(int(value) for value in completion_values)
        if all(value is not None for value in completion_values)
        else None
    )
    return {
        "crossplay_version": CROSSPLAY_VERSION,
        "focal_policy": focal,
        "opponent_policy": opponent,
        "complete_side_swapped_pairs": len(paired),
        "interval_method": "paired seed-level nonparametric bootstrap, 20000 deterministic resamples",
        "test_method": "paired two-sided sign randomization; exact through 16 seeds, otherwise 100000 trials",
        "conditions": conditions,
        "communication_effects": effects,
        "inference": {
            "rows": len(rows),
            "requests": sum(int(row["inference"]["requests"]) for row in rows),
            "wall_seconds": total_inference_seconds,
            "completion_tokens": total_completion_tokens,
            "completion_tokens_per_second": (
                total_completion_tokens / total_inference_seconds
                if total_completion_tokens is not None and total_inference_seconds
                else None
            ),
        },
    }


def parse_conditions(value: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in value.split(","):
        blue, separator, red = item.partition(":")
        if not separator or blue not in CONDITIONS or red not in CONDITIONS:
            raise ValueError(f"invalid condition pair: {item}")
        pairs.append((blue, red))
    if not pairs:
        raise ValueError("at least one condition pair is required")
    return tuple(pairs)


def prepare_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
) -> str:
    """Bind an output directory to one exact protocol before writing rows."""
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # Canonicalize tuples to their on-disk JSON representation so equality on
    # resume compares like with like instead of rejecting identical arguments.
    record = json.loads(json.dumps({**manifest, "sha256": manifest_sha256}, sort_keys=True))
    manifest_path = output_dir / "manifest.json"
    rows_path = output_dir / "rows.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite an existing cross-play run: {output_dir}"
            )
        if existing != record:
            raise ValueError(
                "resume manifest mismatch; use the original arguments or a new output directory"
            )
    elif rows_path.is_file():
        raise ValueError(f"rows exist without a manifest: {rows_path}")
    else:
        manifest_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable two-model 4v4 Swarm Arena cross-play.")
    parser.add_argument("--blue-base-url", required=True)
    parser.add_argument("--blue-model", required=True)
    parser.add_argument(
        "--blue-artifact-id",
        required=True,
        help="Immutable HF revision or local adapter SHA-256 for the served BLUE model",
    )
    parser.add_argument("--red-base-url", required=True)
    parser.add_argument("--red-model", required=True)
    parser.add_argument(
        "--red-artifact-id",
        required=True,
        help="Immutable HF revision or local adapter SHA-256 for the served RED model",
    )
    parser.add_argument("--api-key", default="local")
    parser.add_argument(
        "--constrain-protocol",
        action="store_true",
        help="Dynamically constrain JSON to observed facts, legal intents/actions, and live message budget",
    )
    parser.add_argument("--split", choices=("development", "frozen"), default="development")
    parser.add_argument("--cases", type=int)
    parser.add_argument("--seed-base", type=int, default=1_000_003)
    parser.add_argument("--blue-history-window", type=int, default=3)
    parser.add_argument("--red-history-window", type=int, default=3)
    parser.add_argument(
        "--conditions",
        default="generated:generated,dropped:generated,sender_shuffled:generated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--swap-sides", action="store_true")
    args = parser.parse_args()

    blue_provider = OpenAICompatibleProvider(
        args.blue_base_url,
        args.blue_model,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=160,
        enable_thinking=False,
        response_format_factory=protocol_response_format if args.constrain_protocol else None,
    )
    same_endpoint = args.blue_base_url == args.red_base_url and args.blue_model == args.red_model
    blue_model: ArenaModel = OpenAIArenaModel(blue_provider, args.blue_model)
    if same_endpoint:
        red_model = blue_model
    else:
        red_model = OpenAIArenaModel(
            OpenAICompatibleProvider(
                args.red_base_url,
                args.red_model,
                api_key=args.api_key,
                temperature=0.0,
                max_tokens=160,
                enable_thinking=False,
                response_format_factory=protocol_response_format if args.constrain_protocol else None,
            ),
            args.red_model,
        )

    if args.split == "frozen":
        if args.cases is not None:
            parser.error("--cases cannot truncate the frozen cross-play split")
        cases = FROZEN_CROSSPLAY_CASES
    else:
        cases = development_cases(args.cases or 8, args.seed_base)
    condition_pairs = parse_conditions(args.conditions)
    repository_root = Path(__file__).resolve().parents[3]
    source_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    manifest = {
        "version": CROSSPLAY_VERSION,
        "source_commit": source_commit,
        "prompt_version": EPISODE_PROMPT_VERSION,
        "blue_model": args.blue_model,
        "blue_artifact_id": args.blue_artifact_id,
        "red_model": args.red_model,
        "red_artifact_id": args.red_artifact_id,
        "generation": {"temperature": 0.0, "max_tokens": 160, "enable_thinking": False},
        "protocol_constraint": {
            "enabled": args.constrain_protocol,
            "version": STRUCTURED_PROTOCOL_VERSION if args.constrain_protocol else None,
        },
        "analysis": {
            "unit": "seed after side averaging",
            "bootstrap_resamples": 20_000,
            "randomization_trials": 100_000,
            "random_seed": 0,
        },
        "split": args.split,
        "cases": cases,
        "conditions": condition_pairs,
        "blue_history_window": args.blue_history_window,
        "red_history_window": args.red_history_window,
        "swap_sides": args.swap_sides,
    }
    manifest_sha256 = prepare_manifest(args.output_dir, manifest, resume=args.resume)
    rows_path = args.output_dir / "rows.jsonl"
    rows = []
    if args.resume and rows_path.is_file():
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    completed = {
        (
            row["seed"],
            row["blue_model"],
            row["red_model"],
            row["blue_condition"],
            row["red_condition"],
        )
        for row in rows
    }
    with rows_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for case in cases:
            for blue_condition, red_condition in condition_pairs:
                assignments = [
                    (
                        blue_model,
                        red_model,
                        blue_condition,
                        red_condition,
                        args.blue_history_window,
                        args.red_history_window,
                    )
                ]
                if args.swap_sides and blue_model.name != red_model.name:
                    assignments.append(
                        (
                            red_model,
                            blue_model,
                            red_condition,
                            blue_condition,
                            args.red_history_window,
                            args.blue_history_window,
                        )
                    )
                for (
                    assigned_blue,
                    assigned_red,
                    assigned_blue_condition,
                    assigned_red_condition,
                    assigned_blue_history,
                    assigned_red_history,
                ) in assignments:
                    key = (
                        case[0],
                        assigned_blue.name,
                        assigned_red.name,
                        assigned_blue_condition,
                        assigned_red_condition,
                    )
                    if key in completed:
                        continue
                    row = evaluate_crossplay(
                        assigned_blue,
                        assigned_red,
                        case,
                        blue_condition=assigned_blue_condition,
                        red_condition=assigned_red_condition,
                        blue_history_window=assigned_blue_history,
                        red_history_window=assigned_red_history,
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "completed": key,
                                "blue_return": row["metrics"]["BLUE"]["terminal_return"],
                            }
                        )
                    )
    summary = summarize(rows, manifest_sha256)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.swap_sides and blue_model.name != red_model.name:
        side_swapped = summarize_side_swapped(rows, blue_model.name)
        (args.output_dir / "side_swapped_summary.json").write_text(
            json.dumps(side_swapped, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
