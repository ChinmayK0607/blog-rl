from __future__ import annotations

import json
from typing import Any

from .arena import Action
from .arena_protocol import displayed_actions
from .episode import ArenaEpisodeEnv

EPISODE_PROMPT_VERSION = "arena-episode-v5-actionable-broadcast-priority"

BASE_SYSTEM_PROMPT = """You are one of four teammates in a deterministic graph-control game against four opponents. The episode lasts several simultaneous turns. You receive no intermediate task reward; the team is scored only at the end.

SCAN reveals an unknown adjacent node. PROBE removes fortification or exposes a non-team node. CAPTURE takes a node that is exposed after simultaneous defenses and probes resolve. FORTIFY protects a team node. RECOVER restores a compromised team node. TRANSFER gives one resource. WAIT does nothing.

Your observation and event feedback are private. Teammate messages are grounded reports, not commands. Prefer newer observations, adapt when the opponent changes behavior, and avoid redundant same-team actions. `legal_actions` is the complete action interface for this turn. Output exactly one requested JSON object, then stop. Do not append another object, commentary, or markup."""

BROADCAST_SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + """

PHASE: BROADCAST
Communication is optional and consumes your private episodic budget. A non-empty message uses 1 unit plus 1 per fact, 1 for an intent, and 1 for a resource request. The `reward_contract` field states whether this budget use also affects terminal reward. Send only information present in your observation and at most two facts. An empty message uses zero units.

Use the limited fact slots for information most likely to change a teammate's legal action. Prefer actionable EXPOSED or COMPROMISED nodes, then critical or high-value nodes, then other facts. A slightly older actionable fact can be more useful than an unrelated fresh fact. Use recency only to resolve conflicting reports about the same node.

Return exactly:
{"facts":[{"node":"ID","owner":"BLUE|RED|NEUTRAL","status":"SECURE|FORTIFIED|EXPOSED|COMPROMISED","value":1,"critical":false,"observed_turn":0}],"intent":{"type":"ACTION","target":"ID"}|null,"request_resource":0}

For WAIT, intent is {"type":"WAIT"}. TRANSFER also includes `amount`. Never use action IDs inside `intent`. If `intent` is non-null, copy exactly one object from `legal_intents`; never synthesize a type and target combination. `unknown_neighbors` contains identifiers only: they are not observations, so never put one in `facts` or infer its owner, status, value, or criticality."""
)

ACTION_SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + """

PHASE: ACT
Choose one legal action. Use current teammate reports when they reveal an adjacent node or prevent duplicated work. Decide silently.

Return exactly:
{"action_id":"one ID copied from legal_actions"}

The object must contain exactly the one key `action_id`. `type`, `target`, and
`amount` are invalid in this phase."""
)


def _legal_action_rows(actions: tuple[Action, ...]) -> list[dict[str, Any]]:
    return [dict(action.to_dict(), id=f"A{index}") for index, action in enumerate(actions)]


def _reward_contract(env: ArenaEpisodeEnv) -> dict[str, Any]:
    shaped = any(
        value != 0.0
        for value in (
            env.config.communication_cost,
            env.config.invalid_broadcast_cost,
            env.config.invalid_action_cost,
        )
    )
    return {
        "terminal_only": True,
        "objective": (
            "maximize terminal controlled-node value net of configured costs"
            if shaped
            else "maximize normalized terminal controlled-node margin change"
        ),
        "communication_has_reward_cost": env.config.communication_cost != 0.0,
        "invalid_outputs_are_rewarded": False,
    }


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
        "team_objective": _reward_contract(env)["objective"],
        "reward_contract": _reward_contract(env),
        "observation": env.observations()[agent_id],
        "message_cost": {
            "base_nonempty": 1,
            "per_fact": 1,
            "intent": 1,
            "resource_request": 1,
        },
        "max_facts": env.config.max_facts_per_message,
        "legal_actions": _legal_action_rows(actions),
        "legal_intents": [action.to_dict() for action in actions],
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
        "team_objective": _reward_contract(env)["objective"],
        "reward_contract": _reward_contract(env),
        "observation": observation,
        "legal_actions": _legal_action_rows(actions),
    }
    return [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"))},
    ], actions
