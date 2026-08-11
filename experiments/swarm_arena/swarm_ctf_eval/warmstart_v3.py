from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .arena_oracle import local_policy_action
from .arena_protocol import Broadcast, encode_action, encode_broadcast, parse_action, parse_broadcast
from .arena_sft import oracle_broadcast
from .episode import ArenaEpisodeEnv, EpisodeConfig
from .episode_protocol import EPISODE_PROMPT_VERSION, episode_action_prompt, episode_broadcast_prompt

DATASET_VERSION = "arena-warmstart-v3"


def _row(messages: list[dict[str, str]], source: str, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {"messages": messages, "source": source, "metadata": metadata},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "id": hashlib.sha256(payload.encode()).hexdigest(),
        "messages": messages,
        "source": source,
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }


def generate_arena_rows(start_seed: int, seeds: int, split: str) -> list[dict[str, Any]]:
    rows = []
    for offset in range(seeds):
        seed = start_seed + offset
        env = ArenaEpisodeEnv(seed, 12 + seed % 2, EpisodeConfig(horizon=4))
        env.reset()
        state = env._require_state()
        agents = sorted(agent.id for agent in state.agents.values() if agent.team == "BLUE")
        broadcasts = {}
        for index, agent_id in enumerate(agents):
            intent = local_policy_action(state, agent_id)
            full = oracle_broadcast(state, agent_id, intent)
            mode = (seed + index) % 4
            if mode == 0:
                message = Broadcast((), None, 0)
            elif mode == 1:
                message = Broadcast(full.facts[:2], None, full.request_resource)
            elif mode == 2:
                message = Broadcast(full.facts[:1], full.intent, full.request_resource)
            else:
                message = Broadcast(full.facts[:2], full.intent, full.request_resource)
            broadcasts[agent_id] = message
            prompt, _ = episode_broadcast_prompt(env, agent_id, seed + index)
            rows.append(
                _row(
                    [*prompt, {"role": "assistant", "content": encode_broadcast(message)}],
                    "arena_protocol",
                    {
                        "dataset_version": DATASET_VERSION,
                        "prompt_version": EPISODE_PROMPT_VERSION,
                        "phase": "BROADCAST",
                        "split": split,
                        "seed": seed,
                        "agent_id": agent_id,
                        "message_mode": mode,
                        "label_source": "grounded_local_policy",
                    },
                )
            )
        env.broadcast_phase(broadcasts)
        state = env._require_state()
        for index, agent_id in enumerate(agents):
            action = local_policy_action(state, agent_id)
            prompt, displayed = episode_action_prompt(env, agent_id, permutation=seed + index)
            rows.append(
                _row(
                    [*prompt, {"role": "assistant", "content": encode_action(action, displayed)}],
                    "arena_protocol",
                    {
                        "dataset_version": DATASET_VERSION,
                        "prompt_version": EPISODE_PROMPT_VERSION,
                        "phase": "ACT",
                        "split": split,
                        "seed": seed,
                        "agent_id": agent_id,
                        "label_source": "prompt_visible_local_policy",
                    },
                )
            )
    return rows


def generate_preservation_rows(seed: int, examples: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    system = "Follow the user's current instruction carefully and return only the requested JSON object."
    for index in range(examples):
        category = index % 4
        if category == 0:
            symbols = rng.sample(list("JKLMNPQR"), 4)
            values = rng.sample(range(10, 100), 4)
            mapping = dict(zip(symbols, values, strict=True))
            order = rng.sample(symbols, 3)
            expected = {"slot_c": mapping[order[2]], "slot_a": mapping[order[0]], "slot_b": mapping[order[1]]}
            user = f"Lookup table: {mapping}. Requested symbols: {order}. Return only " + json.dumps(
                {
                    "slot_c": "third requested value",
                    "slot_a": "first requested value",
                    "slot_b": "second requested value",
                }
            )
            kind = "binding_rehearsal"
        elif category == 1:
            values = [rng.randint(-20, 30) for _ in range(9)]
            threshold = rng.randint(-3, 12)
            kept = sorted(value for value in values if value > threshold)
            expected = {"count": len(kept), "values": kept}
            user = f"Keep values greater than {threshold} from {values}, sort ascending, and return " + json.dumps(
                {"count": "number kept", "values": ["kept integers"]}
            )
            kind = "filter_rehearsal"
        elif category == 2:
            left, right, offset = rng.randint(7, 60), rng.randint(3, 14), rng.randint(1, 20)
            expected = {"difference": left * right - offset}
            user = f'Compute ({left} multiplied by {right}) minus {offset}. Return {{"difference":integer}}.'
            kind = "arithmetic_rehearsal"
        else:
            labels = rng.sample(list("ABCDEFGH"), 4)
            measures = rng.sample(range(100, 900), 4)
            table = dict(zip(labels, measures, strict=True))
            low = min(table, key=table.get)  # type: ignore[arg-type]
            high = max(table, key=table.get)  # type: ignore[arg-type]
            expected = {"largest_label": high, "smallest_label": low}
            user = f'Measurements: {table}. Return {{"largest_label":"key","smallest_label":"key"}}.'
            kind = "selection_rehearsal"
        messages = (
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(expected, separators=(",", ":"))},
        )
        rows.append(
            _row(
                list(messages),
                "instruction_preservation",
                {"dataset_version": DATASET_VERSION, "split": split, "seed": seed, "kind": kind, "index": index},
            )
        )
    return rows


def validate_warmstart_response(row: dict[str, Any], raw: str) -> dict[str, bool]:
    metadata = json.loads(row["metadata_json"])
    target = json.loads(row["messages"][-1]["content"])
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"schema_valid": False, "grounded": False, "legal": False, "exact": False}
    if row["source"] == "instruction_preservation":
        valid = isinstance(value, dict)
        return {"schema_valid": valid, "grounded": valid, "legal": valid, "exact": valid and value == target}
    if row["source"] != "arena_protocol":
        raise ValueError(f"unsupported validation source: {row['source']}")

    seed = int(metadata["seed"])
    agent_id = str(metadata["agent_id"])
    env = ArenaEpisodeEnv(seed, 12 + seed % 2, EpisodeConfig(horizon=4))
    env.reset()
    state = env._require_state()
    if metadata["phase"] == "BROADCAST":
        parsed = parse_broadcast(raw, state, agent_id)
        grounded = parsed.valid and len(parsed.value.facts) <= env.config.max_facts_per_message
        legal = grounded and not env.broadcast_phase({agent_id: parsed.value}).errors[agent_id]
    elif metadata["phase"] == "ACT":
        broadcasts = {}
        agents = sorted(agent.id for agent in state.agents.values() if agent.team == "BLUE")
        for index, current_id in enumerate(agents):
            intent = local_policy_action(state, current_id)
            full = oracle_broadcast(state, current_id, intent)
            mode = (seed + index) % 4
            if mode == 0:
                message = Broadcast((), None, 0)
            elif mode == 1:
                message = Broadcast(full.facts[:2], None, full.request_resource)
            elif mode == 2:
                message = Broadcast(full.facts[:1], full.intent, full.request_resource)
            else:
                message = Broadcast(full.facts[:2], full.intent, full.request_resource)
            broadcasts[current_id] = message
        env.broadcast_phase(broadcasts)
        _, displayed = episode_action_prompt(env, agent_id, permutation=seed + agents.index(agent_id))
        parsed = parse_action(raw, displayed)
        grounded = parsed.valid
        legal = parsed.valid
    else:
        raise ValueError(f"unsupported arena phase: {metadata['phase']}")
    return {
        "schema_valid": parsed.valid,
        "grounded": grounded,
        "legal": legal,
        "exact": isinstance(value, dict) and value == target,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_dataset(
    replay_rows_path: Path,
    output_dir: Path,
    *,
    train_arena_seeds: int = 80,
    validation_arena_seeds: int = 12,
    train_preservation: int = 640,
    validation_preservation: int = 96,
    replay_limit: int = 1280,
) -> dict[str, Any]:
    from datasets import Dataset

    replay_rows = load_jsonl(replay_rows_path)
    replay_rows = [
        {key: value for key, value in row.items() if key != "truncated"}
        for row in replay_rows
        if not row.get("truncated", False)
    ][:replay_limit]
    if len(replay_rows) < replay_limit:
        raise ValueError(f"need {replay_limit} non-truncated replay rows, found {len(replay_rows)}")
    train = [
        *generate_arena_rows(7_000_000, train_arena_seeds, "train"),
        *generate_preservation_rows(20260901, train_preservation, "train"),
        *replay_rows,
    ]
    validation = [
        *generate_arena_rows(7_100_000, validation_arena_seeds, "validation"),
        *generate_preservation_rows(20260902, validation_preservation, "validation"),
    ]
    random.Random(20260903).shuffle(train)
    random.Random(20260904).shuffle(validation)
    identifiers = [row["id"] for row in [*train, *validation]]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("duplicate warm-start row ids")
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).to_parquet(output_dir / "train.parquet")
    Dataset.from_list(validation).to_parquet(output_dir / "validation.parquet")
    source_counts = {
        source: sum(row["source"] == source for row in train) for source in sorted({row["source"] for row in train})
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_by_source": source_counts,
        "content_sha256": hashlib.sha256("".join(sorted(identifiers)).encode()).hexdigest(),
        "replay_source": "Qwen3-4B-Instruct-2507 deterministic responses to filtered UltraChat 200k prompts",
        "replay_rows_requested": replay_limit,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
