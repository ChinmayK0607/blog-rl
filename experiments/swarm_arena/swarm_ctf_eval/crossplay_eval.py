from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .arena import TEAMS, WAIT, Action, Team
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
from .episode_protocol import episode_action_prompt, episode_broadcast_prompt
from .providers import OpenAICompatibleProvider

CROSSPLAY_VERSION = "arena-crossplay-v1-private-history"
CONDITIONS = ("generated", "dropped", "sender_shuffled", "delayed", "zero_budget")
Case = tuple[int, int, int]
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


def _with_history(messages: list[dict[str, str]], history: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not history:
        return messages
    body = json.loads(messages[-1]["content"])
    body["private_history"] = history
    return [*messages[:-1], {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"))}]


def _respond_agents(
    jobs: list[tuple[str, ArenaModel, list[dict[str, str]]]],
) -> dict[str, str]:
    grouped: dict[int, tuple[ArenaModel, list[tuple[str, list[dict[str, str]]]]]] = {}
    for agent_id, model, prompt in jobs:
        key = id(model)
        if key not in grouped:
            grouped[key] = (model, [])
        grouped[key][1].append((agent_id, prompt))

    def run_group(group: tuple[ArenaModel, list[tuple[str, list[dict[str, str]]]]]) -> list[tuple[str, str]]:
        model, entries = group
        responses = _respond_many(model, [prompt for _, prompt in entries], ["{}"] * len(entries))
        return [(agent_id, response) for (agent_id, _), response in zip(entries, responses, strict=True)]

    groups = list(grouped.values())
    if len(groups) == 1:
        results = [run_group(groups[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            results = list(executor.map(run_group, groups))
    return {agent_id: response for group in results for agent_id, response in group}


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
    raise ValueError(f"unknown condition: {condition}")


def evaluate_crossplay(
    blue_model: ArenaModel,
    red_model: ArenaModel,
    case: Case,
    *,
    blue_condition: str = "generated",
    red_condition: str = "generated",
    history_window: int = 3,
) -> dict[str, Any]:
    if blue_condition not in CONDITIONS or red_condition not in CONDITIONS:
        raise ValueError("unknown communication condition")
    if history_window < 0:
        raise ValueError("history window cannot be negative")

    seed, size, horizon = case
    env = ArenaEpisodeEnv(seed, size, EpisodeConfig(horizon=horizon))
    env.reset()
    models: dict[Team, ArenaModel] = {"BLUE": blue_model, "RED": red_model}
    conditions: dict[Team, str] = {"BLUE": blue_condition, "RED": red_condition}
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

    for turn in range(horizon):
        state = env._require_state()
        broadcast_jobs = []
        for team in TEAMS:
            if conditions[team] == "zero_budget":
                continue
            for index, agent_id in enumerate(_members(env, team)):
                prompt, _ = episode_broadcast_prompt(env, agent_id, seed + turn * 19 + index)
                prompt = _with_history(prompt, histories[agent_id][-history_window:] if history_window else [])
                broadcast_jobs.append((agent_id, models[team], prompt))
        raw_broadcasts = _respond_agents(broadcast_jobs) if broadcast_jobs else {}

        parsed_messages: dict[str, Broadcast] = {}
        broadcast_rows = []
        for agent_id, raw in sorted(raw_broadcasts.items()):
            team = state.agents[agent_id].team
            parsed = parse_broadcast(raw, state, agent_id)
            message = parsed.value if parsed.valid else EMPTY_BROADCAST
            assert isinstance(message, Broadcast)
            parsed_messages[agent_id] = message
            attempts[team]["broadcast"] += 1
            attempts[team]["broadcast_protocol"] += int(parsed.valid)
            broadcast_rows.append(
                {
                    "agent_id": agent_id,
                    "team": team,
                    "raw_response": raw,
                    "protocol_valid": parsed.valid,
                    "protocol_errors": list(parsed.errors),
                }
            )

        preview = _preview_accepted(env, parsed_messages)
        delivered: dict[str, Broadcast] = {}
        for team in TEAMS:
            members = _members(env, team)
            delivered.update(
                _delivered_for_team(conditions[team], members, preview, previous_accepted)
            )
        phase = env.broadcast_phase(parsed_messages, delivered_broadcasts=delivered)
        previous_accepted = dict(phase.accepted)
        for row in broadcast_rows:
            agent_id = row["agent_id"]
            team = row["team"]
            row["environment_errors"] = list(phase.errors[agent_id])
            row["message_units"] = phase.message_units[agent_id]
            attempts[team]["broadcast_grounded"] += int(not phase.errors[agent_id])

        action_jobs = []
        displayed_by_agent: dict[str, tuple[Action, ...]] = {}
        for team in TEAMS:
            for index, agent_id in enumerate(_members(env, team)):
                prompt, displayed = episode_action_prompt(
                    env,
                    agent_id,
                    permutation=seed + turn * 23 + index,
                )
                prompt = _with_history(prompt, histories[agent_id][-history_window:] if history_window else [])
                displayed_by_agent[agent_id] = displayed
                action_jobs.append((agent_id, models[team], prompt))
        raw_actions = _respond_agents(action_jobs)

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
                    "raw_response": raw,
                    "protocol_valid": parsed.valid,
                    "protocol_errors": list(parsed.errors),
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
    return {
        "crossplay_version": CROSSPLAY_VERSION,
        "episode_version": EPISODE_VERSION,
        "seed": seed,
        "size": size,
        "horizon": horizon,
        "blue_model": blue_model.name,
        "red_model": red_model.name,
        "blue_condition": blue_condition,
        "red_condition": red_condition,
        "history_window": history_window,
        "metrics": metrics,
        "turns": turns,
    }


def _mean_ci(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - radius, mean + radius]


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
        "conditions": conditions,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable two-model 4v4 Swarm Arena cross-play.")
    parser.add_argument("--blue-base-url", required=True)
    parser.add_argument("--blue-model", required=True)
    parser.add_argument("--red-base-url", required=True)
    parser.add_argument("--red-model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--split", choices=("development", "frozen"), default="development")
    parser.add_argument("--cases", type=int)
    parser.add_argument("--seed-base", type=int, default=1_000_003)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument(
        "--conditions",
        default="generated:generated,dropped:generated,sender_shuffled:generated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    blue_provider = OpenAICompatibleProvider(
        args.blue_base_url,
        args.blue_model,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=160,
        enable_thinking=False,
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
            ),
            args.red_model,
        )

    if args.split == "frozen":
        if args.cases is not None:
            parser.error("--cases cannot truncate the frozen cross-play split")
        cases = FROZEN_CROSSPLAY_CASES
    else:
        cases = development_cases(args.cases or 6, args.seed_base)
    condition_pairs = parse_conditions(args.conditions)
    manifest = {
        "version": CROSSPLAY_VERSION,
        "cases": cases,
        "conditions": condition_pairs,
        "history_window": args.history_window,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({**manifest, "sha256": manifest_sha256}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows_path = args.output_dir / "rows.jsonl"
    rows = []
    if args.resume and rows_path.is_file():
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    completed = {
        (row["seed"], row["blue_condition"], row["red_condition"])
        for row in rows
    }
    with rows_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for case in cases:
            for blue_condition, red_condition in condition_pairs:
                key = (case[0], blue_condition, red_condition)
                if key in completed:
                    continue
                row = evaluate_crossplay(
                    blue_model,
                    red_model,
                    case,
                    blue_condition=blue_condition,
                    red_condition=red_condition,
                    history_window=args.history_window,
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                rows.append(row)
                print(json.dumps({"completed": key, "blue_return": row["metrics"]["BLUE"]["terminal_return"]}))
    summary = summarize(rows, manifest_sha256)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
