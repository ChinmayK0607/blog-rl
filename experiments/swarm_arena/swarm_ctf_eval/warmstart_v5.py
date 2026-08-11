from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .arena_oracle import local_policy_action
from .arena_sft import oracle_broadcast
from .arena_protocol import Broadcast, encode_action, encode_broadcast
from .episode import ArenaEpisodeEnv, EpisodeConfig
from .episode_protocol import EPISODE_PROMPT_VERSION, episode_action_prompt, episode_broadcast_prompt
from .warmstart_v3 import _row, generate_preservation_rows, load_jsonl

DATASET_VERSION = "arena-warmstart-v5"


def generate_arena_rows(start_seed: int, seeds: int, split: str) -> list[dict[str, Any]]:
    """Generate prompt-identifiable protocol targets for a replay-protected warm start."""

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
            # The target is fully determined by prompt-visible observation and legal_actions.
            # Communication cost optimization remains an RL objective, not hidden SFT label noise.
            message = Broadcast(full.facts[: env.config.max_facts_per_message], full.intent, full.request_resource)
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
                        "label_source": "canonical_prompt_identifiable_local_broadcast",
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
        "broadcast_labels": "canonical prompt-identifiable local reports with legal local intents",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
