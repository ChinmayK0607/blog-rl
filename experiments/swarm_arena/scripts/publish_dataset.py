from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

METADATA_DEFAULTS = {
    "agent_id": "",
    "arena_version": "",
    "dataset_version": "",
    "generator_version": "",
    "label_source": "",
    "phase": "",
    "prompt_version": "",
    "seed": 0,
    "split": "",
    "policy_reward": 0.0,
    "solver_joint_actions_explored": 0,
    "solver_optimal_count": 0,
    "solver_reward": 0.0,
    "generator_mode": "",
    "targeted_skill": "",
}


DATASET_CARD = """---
license: apache-2.0
task_categories:
- text-generation
language:
- en
configs:
- config_name: default
  data_files:
  - split: train_broadcast
    path: train_broadcast.jsonl
  - split: train_action
    path: train_action_common.jsonl
  - split: train_action_rare
    path: train_action_rare.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
  - split: overfit
    path: overfit.jsonl
---

# Swarm Arena SFT v2

Solver-filtered warm-start data for the deterministic Swarm Arena 4v4
coordination environment. Each row contains system, user, and assistant messages
plus provenance metadata. Training broadcasts and actions are separate splits so
sampling can preserve a 60/40 phase mixture. Validation and test are never
reweighted.

The simulator, oracle, audit, frozen evaluation, and Prime-RL configs live in
<https://github.com/ChinmayK0607/blog-rl/tree/exp/swarm-arena-4b/experiments/swarm_arena>.

Optional provenance fields use typed zero/empty-string defaults so every JSONL
split has one stable Arrow schema. They do not affect prompts or targets.

Dataset content SHA-256:
`edad09bb301748621a0fab73ebf3de60d60abfd9f56c9afcc6ca02ffe12f3a80`.
"""


def normalize_row(row: dict) -> dict:
    row = dict(row)
    row["metadata"] = {**METADATA_DEFAULTS, **row["metadata"]}
    return row


def write_normalized_split(source: Path, destination: Path) -> None:
    with source.open(encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for line in input_handle:
            output_handle.write(json.dumps(normalize_row(json.loads(line)), sort_keys=True) + "\n")


def split_train(
    source: Path,
    broadcast_path: Path,
    common_action_path: Path,
    rare_action_path: Path,
) -> None:
    rare_types = {"WAIT", "SCAN", "TRANSFER"}
    with (
        source.open(encoding="utf-8") as input_handle,
        broadcast_path.open("w", encoding="utf-8") as broadcast_handle,
        common_action_path.open("w", encoding="utf-8") as common_handle,
        rare_action_path.open("w", encoding="utf-8") as rare_handle,
    ):
        for line in input_handle:
            row = normalize_row(json.loads(line))
            if row["metadata"]["phase"] == "BROADCAST":
                destination = broadcast_handle
            else:
                user = json.loads(row["messages"][-2]["content"])
                action_id = json.loads(row["messages"][-1]["content"])["action_id"]
                action = next(item for item in user["legal_actions"] if item["id"] == action_id)
                destination = rare_handle if action["type"] in rare_types else common_handle
            destination.write(json.dumps(row, sort_keys=True) + "\n")


def make_overfit(source: Path, destination: Path, count: int = 256) -> None:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    broadcasts = [row for row in rows if row["metadata"]["phase"] == "BROADCAST"][:154]
    actions = [row for row in rows if row["metadata"]["phase"] == "ACT"]
    rare_types = {"WAIT", "SCAN", "TRANSFER"}

    def is_rare(row: dict) -> bool:
        user = json.loads(row["messages"][-2]["content"])
        action_id = json.loads(row["messages"][-1]["content"])["action_id"]
        action = next(item for item in user["legal_actions"] if item["id"] == action_id)
        return action["type"] in rare_types

    rare = [row for row in actions if is_rare(row)][:24]
    common = [row for row in actions if not is_rare(row)][: count - 154 - len(rare)]
    with destination.open("w", encoding="utf-8") as handle:
        for row in broadcasts + rare + common:
            handle.write(json.dumps(normalize_row(row), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--repo-id", default="CK0607/swarm-arena-sft-v2")
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swarm-arena-hf-") as temporary:
        staging = Path(temporary)
        split_train(
            args.source / "train.jsonl",
            staging / "train_broadcast.jsonl",
            staging / "train_action_common.jsonl",
            staging / "train_action_rare.jsonl",
        )
        make_overfit(args.source / "train.jsonl", staging / "overfit.jsonl")
        for filename in ("validation.jsonl", "test.jsonl"):
            write_normalized_split(args.source / filename, staging / filename)
        for filename in ("manifest.json", "audit.json"):
            shutil.copy2(args.source / filename, staging / filename)
        (staging / "README.md").write_text(DATASET_CARD, encoding="utf-8")
        if args.experiment_root:
            shutil.copytree(
                args.experiment_root,
                staging / "code",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data", ".venv", "uv.lock"),
            )
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Publish audited Swarm Arena SFT v2",
        )
    info = HfApi(token=False).dataset_info(args.repo_id)
    if info.private:
        raise RuntimeError("published dataset repository is not anonymously public")


if __name__ == "__main__":
    main()
