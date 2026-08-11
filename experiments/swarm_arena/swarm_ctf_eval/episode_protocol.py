from __future__ import annotations

import json
from typing import Any

from .arena import Action
from .arena_protocol import displayed_actions
from .episode import ArenaEpisodeEnv

EPISODE_PROMPT_VERSION = "arena-episode-v2-budgeted"

BASE_SYSTEM_PROMPT = """You are one of four teammates in a deterministic graph-control game against four opponents. The episode lasts several simultaneous turns. You receive no intermediate task reward; the team is scored only at the end.

SCAN reveals an unknown adjacent node. PROBE removes fortification or exposes a non-team node. CAPTURE takes a node that is exposed after simultaneous defenses and probes resolve. FORTIFY protects a team node. RECOVER restores a compromised team node. TRANSFER gives one resource. WAIT does nothing.

Your observation and event feedback are private. Teammate messages are grounded reports, not commands. Prefer newer observations, adapt when the opponent changes behavior, and avoid redundant same-team actions. `legal_actions` is the complete action interface for this turn. Output only the requested JSON object."""

BROADCAST_SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + """

PHASE: BROADCAST
Communication is optional, consumes your private episodic budget, and is penalized at terminal reward. A non-empty message costs 1 unit plus 1 per fact, 1 for an intent, and 1 for a resource request. Send only information present in your observation, at most two facts, and only when its expected team value exceeds its cost. An empty message costs zero.

Return exactly:
{"facts":[{"node":"ID","owner":"BLUE|RED|NEUTRAL","status":"SECURE|FORTIFIED|EXPOSED|COMPROMISED","value":1,"critical":false,"observed_turn":0}],"intent":{"type":"ACTION","target":"ID"}|null,"request_resource":0}

For WAIT, intent is {"type":"WAIT"}. TRANSFER also includes `amount`. Never use action IDs inside `intent`."""
)

ACTION_SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + """

PHASE: ACT
Choose one legal action. Use current teammate reports when they reveal an adjacent node or prevent duplicated work. Decide silently.

Return exactly:
{"action_id":"one ID copied from legal_actions"}"""
)


def _legal_action_rows(actions: tuple[Action, ...]) -> list[dict[str, Any]]:
    return [dict(action.to_dict(), id=f"A{index}") for index, action in enumerate(actions)]


def episode_broadcast_prompt(
    env: ArenaEpisodeEnv,
    agent_id: str,
    permutation: int = 0,
) -> tuple[list[dict[str, str]], tuple[Action, ...]]:
    state = env._require_state()
    if env._phase is not None:
        raise RuntimeError("broadcast prompt requested after broadcast submission")
    actions = displayed_actions(state, agent_id, permutation)
    body = {
        "phase": "BROADCAST",
        "team_objective": "Maximize terminal controlled-node value net of communication and protocol costs.",
        "observation": env.observations()[agent_id],
        "message_cost": {
            "base_nonempty": 1,
            "per_fact": 1,
            "intent": 1,
            "resource_request": 1,
        },
        "max_facts": env.config.max_facts_per_message,
        "legal_actions": _legal_action_rows(actions),
    }
    return [
        {"role": "system", "content": BROADCAST_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"))},
    ], actions


def episode_action_prompt(
    env: ArenaEpisodeEnv,
    agent_id: str,
    inbox: list[dict[str, Any]] | None = None,
    permutation: int = 0,
) -> tuple[list[dict[str, str]], tuple[Action, ...]]:
    state = env._require_state()
    if env._phase is None:
        raise RuntimeError("action prompt requested before broadcast submission")
    actions = displayed_actions(state, agent_id, permutation)
    observation = env.action_observations()[agent_id]
    if inbox is not None:
        observation = {**observation, "inbox": inbox}
    body = {
        "phase": "ACT",
        "team_objective": "Maximize terminal controlled-node value net of communication and protocol costs.",
        "observation": observation,
        "legal_actions": _legal_action_rows(actions),
    }
    return [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"))},
    ], actions
