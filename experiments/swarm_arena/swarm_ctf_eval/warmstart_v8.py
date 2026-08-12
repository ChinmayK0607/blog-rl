from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .arena import Action, NodeObservation
from .arena_oracle import local_policy_action
from .arena_protocol import Broadcast, VALID_OWNERS, VALID_STATUSES, encode_action, encode_broadcast
from .arena_sft import oracle_broadcast
from .episode import EMPTY_BROADCAST, ArenaEpisodeEnv, EpisodeConfig, message_units
from .episode_protocol import EPISODE_PROMPT_VERSION, episode_action_prompt, episode_broadcast_prompt
from .warmstart_v3 import _row

DATASET_VERSION = "arena-warmstart-v8-sequential-protocol"
HISTORY_WINDOW = 3


def _with_history(messages: list[dict[str, str]], history: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not history:
        return messages
    body = json.loads(messages[-1]["content"])
    body["private_history"] = history[-HISTORY_WINDOW:]
    return [*messages[:-1], {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"))}]


def budgeted_protocol_broadcast(
    env: ArenaEpisodeEnv,
    agent_id: str,
    intent: Action,
) -> Broadcast:
    """Create one deterministic, prompt-visible message that fits the live budget."""
    state = env._require_state()
    remaining = env.remaining_budget[agent_id]
    full = oracle_broadcast(state, agent_id, intent)
    selected_intent = full.intent if full.intent is not None and remaining >= 2 else None
    selected_request = int(full.request_resource == 1 and 1 + int(selected_intent is not None) + 1 <= remaining)
    base_cost = 1 + int(selected_intent is not None) + selected_request
    facts: list[NodeObservation] = []
    for fact in full.facts[: env.config.max_facts_per_message]:
        if base_cost + len(facts) + 1 <= remaining:
            facts.append(fact)
    if not facts and selected_intent is None and not selected_request:
        # A non-empty message has a base cost of one plus at least one useful
        # component; a remaining budget below two therefore means silence.
        return EMPTY_BROADCAST
    message = Broadcast(tuple(facts), selected_intent, selected_request)
    if message_units(message) > remaining:
        raise AssertionError("budgeted broadcast exceeds the live message budget")
    return message


def generate_sequential_rows(start_seed: int, episodes: int, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(episodes):
        seed = start_seed + offset
        size = 12 if offset % 2 == 0 else 13
        horizon = 4 if offset % 2 == 0 else 6
        env = ArenaEpisodeEnv(seed, size, EpisodeConfig(horizon=horizon))
        env.reset()
        histories: dict[str, list[dict[str, Any]]] = {
            agent_id: [] for agent_id in env._require_state().agents
        }
        for turn in range(horizon):
            state = env._require_state()
            agents = sorted(state.agents)
            intents = {agent_id: local_policy_action(state, agent_id) for agent_id in agents}
            broadcasts = {
                agent_id: budgeted_protocol_broadcast(env, agent_id, intents[agent_id])
                for agent_id in agents
            }
            for index, agent_id in enumerate(agents):
                prompt, _ = episode_broadcast_prompt(env, agent_id, seed + turn * 31 + index)
                prompt = _with_history(prompt, histories[agent_id])
                target = broadcasts[agent_id]
                rows.append(
                    _row(
                        [*prompt, {"role": "assistant", "content": encode_broadcast(target)}],
                        "arena_protocol",
                        {
                            "dataset_version": DATASET_VERSION,
                            "prompt_version": EPISODE_PROMPT_VERSION,
                            "phase": "BROADCAST",
                            "split": split,
                            "seed": seed,
                            "size": size,
                            "horizon": horizon,
                            "turn": turn,
                            "agent_id": agent_id,
                            "team": state.agents[agent_id].team,
                            "history_length": min(HISTORY_WINDOW, len(histories[agent_id])),
                            "budget_before": env.remaining_budget[agent_id],
                            "target_message_units": message_units(target),
                            "target_empty": target == EMPTY_BROADCAST,
                            "label_source": "budgeted_prompt_visible_local_protocol",
                        },
                    )
                )

            phase = env.broadcast_phase(broadcasts)
            actions = {
                agent_id: local_policy_action(env._require_state(), agent_id)
                for agent_id in agents
            }
            for index, agent_id in enumerate(agents):
                prompt, displayed = episode_action_prompt(
                    env,
                    agent_id,
                    permutation=seed + turn * 37 + index,
                )
                prompt = _with_history(prompt, histories[agent_id])
                rows.append(
                    _row(
                        [*prompt, {"role": "assistant", "content": encode_action(actions[agent_id], displayed)}],
                        "arena_protocol",
                        {
                            "dataset_version": DATASET_VERSION,
                            "prompt_version": EPISODE_PROMPT_VERSION,
                            "phase": "ACT",
                            "split": split,
                            "seed": seed,
                            "size": size,
                            "horizon": horizon,
                            "turn": turn,
                            "agent_id": agent_id,
                            "team": state.agents[agent_id].team,
                            "history_length": min(HISTORY_WINDOW, len(histories[agent_id])),
                            "inbox_size": len(phase.inboxes[agent_id]),
                            "label_source": "prompt_visible_local_policy",
                        },
                    )
                )

            transition = env.advance(actions)
            for agent_id in agents:
                histories[agent_id].append(
                    {
                        "turn": turn,
                        "accepted_broadcast": phase.accepted[agent_id].to_dict(),
                        "received_broadcasts": list(phase.inboxes[agent_id]),
                        "selected_action": actions[agent_id].to_dict(),
                        "local_events": transition.observations[agent_id]["last_local_events"],
                    }
                )
            if transition.terminated or transition.truncated:
                break
    return rows


def _broadcast_validation(body: dict[str, Any], value: Any) -> tuple[bool, bool, bool]:
    if not isinstance(value, dict) or set(value) != {"facts", "intent", "request_resource"}:
        return False, False, False
    facts = value["facts"]
    request = value["request_resource"]
    if not isinstance(facts, list) or len(facts) > int(body["max_facts"]):
        return False, False, False
    if type(request) is not int or request not in (0, 1):
        return False, False, False
    known = {item["node"]: item for item in body["observation"]["known_nodes"]}
    seen: set[str] = set()
    structurally_valid = True
    supported = True
    for fact in facts:
        if not isinstance(fact, dict) or set(fact) != {
            "node", "owner", "status", "value", "critical", "observed_turn"
        }:
            structurally_valid = False
            supported = False
            break
        valid_fields = (
            isinstance(fact["node"], str)
            and fact["node"] not in seen
            and fact["owner"] in VALID_OWNERS
            and fact["status"] in VALID_STATUSES
            and type(fact["value"]) is int
            and fact["value"] in (1, 2, 3)
            and type(fact["critical"]) is bool
            and type(fact["observed_turn"]) is int
        )
        if not valid_fields:
            structurally_valid = False
            supported = False
            break
        seen.add(fact["node"])
        supported = supported and known.get(fact["node"]) == fact
    intent = value["intent"]
    intent_structural = intent is None or isinstance(intent, dict)
    structurally_valid = structurally_valid and intent_structural
    intent_legal = intent is None or intent in body["legal_intents"]
    units = 0 if value == EMPTY_BROADCAST.to_dict() else 1 + len(facts) + int(intent is not None) + request
    budget_legal = units <= int(body["observation"]["message_budget_remaining"])
    legal = structurally_valid and supported and intent_legal and budget_legal
    return structurally_valid, structurally_valid and supported, legal


def validate_sequential_response(row: dict[str, Any], raw: str) -> dict[str, bool]:
    target = json.loads(row["messages"][-1]["content"])
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"schema_valid": False, "grounded": False, "legal": False, "exact": False}
    body = json.loads(row["messages"][-2]["content"])
    if body["phase"] == "BROADCAST":
        schema_valid, grounded, legal = _broadcast_validation(body, value)
    elif body["phase"] == "ACT":
        schema_valid = isinstance(value, dict) and set(value) == {"action_id"} and isinstance(value["action_id"], str)
        grounded = schema_valid
        identifiers = {item["id"] for item in body["legal_actions"]}
        legal = schema_valid and value["action_id"] in identifiers
    else:
        raise ValueError(f"unsupported sequential phase: {body['phase']}")
    return {
        "schema_valid": schema_valid,
        "grounded": grounded,
        "legal": legal,
        "exact": schema_valid and value == target,
    }


def write_dataset(
    source_dataset: Path,
    output_dir: Path,
    *,
    train_episodes: int = 40,
    validation_episodes: int = 8,
) -> dict[str, Any]:
    from datasets import Dataset, load_dataset

    source_train = load_dataset("parquet", data_files=str(source_dataset / "train.parquet"), split="train")
    source_validation = load_dataset(
        "parquet", data_files=str(source_dataset / "validation.parquet"), split="train"
    )
    preservation_train = [dict(row) for row in source_train if row["source"] != "arena_protocol"]
    preservation_validation = [
        dict(row) for row in source_validation if row["source"] != "arena_protocol"
    ]
    train = [
        *generate_sequential_rows(7_400_000, train_episodes, "train"),
        *preservation_train,
    ]
    validation = [
        *generate_sequential_rows(7_500_000, validation_episodes, "validation"),
        *preservation_validation,
    ]
    random.Random(20260930).shuffle(train)
    random.Random(20261001).shuffle(validation)
    identifiers = [row["id"] for row in [*train, *validation]]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("duplicate v8 row ids")
    for row in [*train, *validation]:
        metadata = json.loads(row["metadata_json"])
        if metadata.get("dataset_version") == DATASET_VERSION:
            result = validate_sequential_response(row, row["messages"][-1]["content"])
            if result != {"schema_valid": True, "grounded": True, "legal": True, "exact": True}:
                raise AssertionError(f"invalid sequential target {row['id']}: {result}")
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).to_parquet(output_dir / "train.parquet")
    Dataset.from_list(validation).to_parquet(output_dir / "validation.parquet")
    sequential = [row for row in train if row["source"] == "arena_protocol"]
    metadata = [json.loads(row["metadata_json"]) for row in sequential]
    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_dataset": str(source_dataset),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_by_source": dict(sorted(Counter(row["source"] for row in train).items())),
        "sequential_protocol": {
            "episodes": train_episodes,
            "rows": len(sequential),
            "teams": dict(sorted(Counter(item["team"] for item in metadata).items())),
            "turns": dict(sorted(Counter(str(item["turn"]) for item in metadata).items())),
            "history_lengths": dict(sorted(Counter(str(item["history_length"]) for item in metadata).items())),
            "empty_broadcast_targets": sum(item.get("target_empty", False) for item in metadata),
            "depleted_budget_broadcast_targets": sum(
                item.get("phase") == "BROADCAST" and item.get("budget_before") == 0 for item in metadata
            ),
        },
        "ids_sha256": hashlib.sha256("".join(sorted(identifiers)).encode()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
