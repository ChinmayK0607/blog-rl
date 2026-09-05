from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .arena import WAIT, Action
from .arena_eval import ArenaModel, _respond_many
from .arena_oracle import deterministic_policy
from .arena_protocol import Broadcast, parse_action, parse_broadcast
from .episode import (
    EMPTY_BROADCAST,
    EPISODE_VERSION,
    ArenaEpisodeEnv,
    EpisodeConfig,
    message_units,
    validate_message,
)
from .episode_protocol import EPISODE_PROMPT_VERSION, episode_action_prompt, episode_broadcast_prompt
from .episode_splits import EPISODE_EVAL_CASES, EPISODE_EVAL_MANIFEST_SHA256

CONDITIONS = ("generated", "dropped", "sender_shuffled", "delayed", "zero_budget")


def _blue_members(env: ArenaEpisodeEnv) -> list[str]:
    state = env._require_state()
    return sorted(agent.id for agent in state.agents.values() if agent.team == "BLUE")


def _preview_accepted(env: ArenaEpisodeEnv, messages: dict[str, Broadcast]) -> dict[str, Broadcast]:
    state = env._require_state()
    accepted = {}
    for agent_id in sorted(state.agents):
        message = messages.get(agent_id, EMPTY_BROADCAST)
        errors = validate_message(
            state,
            agent_id,
            message,
            max_facts=env.config.max_facts_per_message,
        )
        if not errors and message_units(message) > env.remaining_budget[agent_id]:
            errors = ("message_budget_exceeded",)
        accepted[agent_id] = EMPTY_BROADCAST if errors else message
    return accepted


def _rotate_senders(messages: dict[str, Broadcast], members: list[str]) -> dict[str, Broadcast]:
    if len(members) < 2:
        return dict(messages)
    return {agent_id: messages[members[(index + 1) % len(members)]] for index, agent_id in enumerate(members)}


def _mean_ci(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - radius, mean + radius]


def evaluate_episode(
    model: ArenaModel,
    case: tuple[int, int, int, str, str],
    condition: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    seed, size, horizon, style_before, style_after = case
    budget = 0 if condition == "zero_budget" else EpisodeConfig().message_budget_per_agent
    env = ArenaEpisodeEnv(seed, size, EpisodeConfig(horizon=horizon, message_budget_per_agent=budget))
    env.reset()
    blue = _blue_members(env)
    previous_accepted: dict[str, Broadcast] = {}
    turns = []
    broadcast_attempts = 0
    broadcast_protocol_valid = 0
    broadcast_grounded_valid = 0
    action_attempts = 0
    action_protocol_valid = 0
    action_order_matches = 0
    action_order_checks = 0
    final = None

    for turn in range(horizon):
        parsed_messages: dict[str, Broadcast] = {}
        broadcast_rows = []
        if condition != "zero_budget":
            jobs = []
            for index, agent_id in enumerate(blue):
                prompt, _ = episode_broadcast_prompt(env, agent_id, seed + turn * 11 + index)
                jobs.append((agent_id, prompt))
            responses = _respond_many(model, [prompt for _, prompt in jobs], ["{}"] * len(jobs))
            state = env._require_state()
            for (agent_id, _), raw in zip(jobs, responses, strict=True):
                parsed = parse_broadcast(raw, state, agent_id)
                message = parsed.value if parsed.valid else EMPTY_BROADCAST
                assert isinstance(message, Broadcast)
                parsed_messages[agent_id] = message
                broadcast_attempts += 1
                broadcast_protocol_valid += int(parsed.valid)
                broadcast_rows.append(
                    {
                        "agent_id": agent_id,
                        "raw_response": raw,
                        "protocol_valid": parsed.valid,
                        "protocol_errors": list(parsed.errors),
                    }
                )

        preview = _preview_accepted(env, parsed_messages)
        delivered = None
        if condition in {"dropped", "zero_budget"}:
            delivered = {}
        elif condition == "sender_shuffled":
            delivered = _rotate_senders(preview, blue)
        elif condition == "delayed":
            delivered = previous_accepted
        phase = env.broadcast_phase(parsed_messages, delivered_broadcasts=delivered)
        previous_accepted = {agent_id: phase.accepted[agent_id] for agent_id in blue}
        broadcast_grounded_valid += sum(not phase.errors[agent_id] for agent_id in blue)
        by_agent = {row["agent_id"]: row for row in broadcast_rows}
        for agent_id in blue:
            if agent_id in by_agent:
                by_agent[agent_id]["environment_errors"] = list(phase.errors[agent_id])
                by_agent[agent_id]["message_units"] = phase.message_units[agent_id]

        primary_jobs = []
        consistency_jobs = []
        for index, agent_id in enumerate(blue):
            permutation = seed + turn * 17 + index
            prompt, displayed = episode_action_prompt(env, agent_id, permutation=permutation)
            primary_jobs.append((agent_id, displayed, prompt))
            if condition == "generated":
                alternate_prompt, alternate_displayed = episode_action_prompt(
                    env,
                    agent_id,
                    permutation=permutation + 1,
                )
                consistency_jobs.append((agent_id, alternate_displayed, alternate_prompt))
        primary_responses = _respond_many(
            model,
            [prompt for _, _, prompt in primary_jobs],
            ["{}"] * len(primary_jobs),
        )
        consistency_responses = _respond_many(
            model,
            [prompt for _, _, prompt in consistency_jobs],
            ["{}"] * len(consistency_jobs),
        )
        selected = {}
        action_rows = []
        semantic_primary: dict[str, Action | None] = {}
        for (agent_id, displayed, _), raw in zip(primary_jobs, primary_responses, strict=True):
            parsed = parse_action(raw, displayed)
            action = parsed.value if parsed.valid else WAIT
            assert isinstance(action, Action)
            selected[agent_id] = action
            semantic_primary[agent_id] = action if parsed.valid else None
            action_attempts += 1
            action_protocol_valid += int(parsed.valid)
            action_rows.append(
                {
                    "agent_id": agent_id,
                    "raw_response": raw,
                    "protocol_valid": parsed.valid,
                    "protocol_errors": list(parsed.errors),
                    "selected_action": action.to_dict(),
                }
            )
        for (agent_id, displayed, _), raw in zip(consistency_jobs, consistency_responses, strict=True):
            parsed = parse_action(raw, displayed)
            alternate = parsed.value if parsed.valid else None
            action_order_checks += 1
            action_order_matches += int(alternate is not None and alternate == semantic_primary[agent_id])

        state = env._require_state()
        style = style_before if turn < horizon // 2 else style_after
        red = deterministic_policy(state, "RED", style)
        final = env.advance({**selected, **red})
        turns.append(
            {
                "turn": turn,
                "opponent_style": style,
                "broadcasts": broadcast_rows,
                "broadcast_errors": {agent_id: list(phase.errors[agent_id]) for agent_id in blue},
                "delivered_messages": sum(phase.delivered[agent_id] != EMPTY_BROADCAST for agent_id in blue),
                "shared_fact_updates": sum(phase.shared_fact_updates[agent_id] for agent_id in blue),
                "actions": action_rows,
                "duplicate_targets": final.info["duplicate_targets"]["BLUE"],
                "team_value": final.info["team_value"]["BLUE"],
            }
        )
        if final.terminated or final.truncated:
            break

    assert final is not None and (final.terminated or final.truncated)
    return {
        "episode_version": EPISODE_VERSION,
        "prompt_version": EPISODE_PROMPT_VERSION,
        "manifest_sha256": EPISODE_EVAL_MANIFEST_SHA256,
        "model": model.name,
        "seed": seed,
        "size": size,
        "horizon": horizon,
        "style_before": style_before,
        "style_after": style_after,
        "condition": condition,
        "terminal_return": final.rewards["BLUE"],
        "broadcast_protocol_rate": (broadcast_protocol_valid / broadcast_attempts if broadcast_attempts else None),
        "broadcast_grounded_rate": broadcast_grounded_valid / max(1, len(blue) * len(turns)),
        "action_protocol_rate": action_protocol_valid / max(1, action_attempts),
        "action_order_consistency": action_order_matches / max(1, action_order_checks),
        "communication_spend": final.info["communication_spend"]["BLUE"],
        "invalid_broadcasts": final.info["invalid_broadcasts"]["BLUE"],
        "invalid_actions": final.info["invalid_actions"]["BLUE"],
        "duplicate_target_turn_rate": statistics.mean(bool(row["duplicate_targets"]) for row in turns),
        "turns": turns,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("episode rows cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    condition_summary = {}
    for condition in CONDITIONS:
        group = grouped[condition]
        returns = [float(row["terminal_return"]) for row in group]
        condition_summary[condition] = {
            "episodes": len(group),
            "mean_terminal_return": statistics.mean(returns),
            "mean_terminal_return_95": _mean_ci(returns),
            "broadcast_protocol_rate": (
                statistics.mean(
                    row["broadcast_protocol_rate"] for row in group if row["broadcast_protocol_rate"] is not None
                )
                if any(row["broadcast_protocol_rate"] is not None for row in group)
                else None
            ),
            "broadcast_grounded_rate": statistics.mean(row["broadcast_grounded_rate"] for row in group),
            "action_protocol_rate": statistics.mean(row["action_protocol_rate"] for row in group),
            "mean_communication_spend": statistics.mean(row["communication_spend"] for row in group),
            "duplicate_target_turn_rate": statistics.mean(row["duplicate_target_turn_rate"] for row in group),
        }
        if condition == "generated":
            condition_summary[condition]["action_order_consistency"] = statistics.mean(
                row["action_order_consistency"] for row in group
            )

    lookup = {(row["seed"], row["condition"]): row for row in rows}
    effects = {}
    for right in CONDITIONS[1:]:
        differences = [
            float(lookup[(case[0], "generated")]["terminal_return"])
            - float(lookup[(case[0], right)]["terminal_return"])
            for case in EPISODE_EVAL_CASES
        ]
        effects[f"generated_minus_{right}"] = {
            "mean_terminal_return_difference": statistics.mean(differences),
            "mean_terminal_return_difference_95": _mean_ci(differences),
            "positive_case_rate": statistics.mean(value > 0 for value in differences),
            "zero_case_rate": statistics.mean(value == 0 for value in differences),
        }
    return {
        "episode_version": EPISODE_VERSION,
        "prompt_version": EPISODE_PROMPT_VERSION,
        "manifest_sha256": EPISODE_EVAL_MANIFEST_SHA256,
        "model": rows[0]["model"],
        "episodes_per_condition": len(EPISODE_EVAL_CASES),
        "conditions": condition_summary,
        "communication_effects": effects,
    }


def run(model: ArenaModel, output_dir: Path) -> dict[str, Any]:
    rows = [evaluate_episode(model, case, condition) for case in EPISODE_EVAL_CASES for condition in CONDITIONS]
    summary = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
