from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .arena import ARENA_VERSION, Action
from .arena_generation import GENERATOR_VERSION, generate_mechanics_state, generate_state
from .arena_oracle import deterministic_policy, local_policy_action, solve_joint_action
from .arena_protocol import ARENA_PROMPT_VERSION, parse_action, parse_broadcast
from .arena_sft import DATASET_VERSION, split_for_seed
from .arena_splits import FROZEN_EVAL_SEEDS


def _action(value: dict[str, Any]) -> Action:
    return Action(value["type"], value.get("target"), value.get("amount"))


def audit_dataset(path: Path, require_split_action_coverage: bool = False) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for split in ("train", "validation", "test"):
        for line in (path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_file_split"] = split
                rows.append(row)

    failures: list[str] = []
    ids: set[str] = set()
    phases: Counter[str] = Counter()
    action_kinds: Counter[str] = Counter()
    action_kinds_by_split: dict[str, Counter[str]] = {split: Counter() for split in ("train", "validation", "test")}
    action_positions: Counter[int] = Counter()
    broadcast_facts: Counter[int] = Counter()
    seeds_by_split: dict[int, set[str]] = {}
    solver_cache = {}
    for row_number, row in enumerate(rows, 1):
        metadata = row.get("metadata", {})
        messages = row.get("messages", [])
        row_id = row.get("id")
        prefix = f"row {row_number} ({row_id})"
        if row_id in ids:
            failures.append(f"{prefix}: duplicate_id")
        ids.add(row_id)
        if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            failures.append(f"{prefix}: chat_schema")
            continue
        if metadata.get("dataset_version") != DATASET_VERSION:
            failures.append(f"{prefix}: dataset_version")
        if metadata.get("arena_version") != ARENA_VERSION:
            failures.append(f"{prefix}: arena_version")
        if metadata.get("generator_version") != GENERATOR_VERSION:
            failures.append(f"{prefix}: generator_version")
        if metadata.get("prompt_version") != ARENA_PROMPT_VERSION:
            failures.append(f"{prefix}: prompt_version")
        seed = metadata.get("seed")
        if not isinstance(seed, int):
            failures.append(f"{prefix}: seed")
            continue
        if seed in FROZEN_EVAL_SEEDS:
            failures.append(f"{prefix}: frozen_eval_leakage")
        expected_split = split_for_seed(seed)
        if metadata.get("split") != expected_split or row["_file_split"] != expected_split:
            failures.append(f"{prefix}: wrong_split")
        seeds_by_split.setdefault(seed, set()).add(expected_split)
        user_text = messages[1].get("content", "")
        if any(term in user_text for term in ("required_joint", "oracle_message", "solver_reward", "answer_key")):
            failures.append(f"{prefix}: answer_leakage")
        try:
            user = json.loads(user_text)
        except json.JSONDecodeError:
            failures.append(f"{prefix}: invalid_user_json")
            continue
        if metadata.get("generator_mode") == "targeted_mechanics":
            state = generate_mechanics_state(seed, metadata.get("targeted_skill"))
        else:
            state = generate_state(seed)
        agent_id = metadata.get("agent_id")
        phase = metadata.get("phase")
        phases[phase] += 1
        assistant = messages[2].get("content", "")
        if phase == "BROADCAST":
            parsed = parse_broadcast(assistant, state, agent_id)
            if not parsed.valid:
                failures.append(f"{prefix}: invalid_broadcast:{','.join(parsed.errors)}")
            else:
                broadcast_facts[len(parsed.value.facts)] += 1
        elif phase == "ACT":
            raw_actions = user.get("legal_actions")
            if not isinstance(raw_actions, list):
                failures.append(f"{prefix}: missing_legal_actions")
                continue
            displayed = tuple(_action(item) for item in raw_actions)
            if [item.get("id") for item in raw_actions] != [f"A{index}" for index in range(len(raw_actions))]:
                failures.append(f"{prefix}: action_ids")
            parsed = parse_action(assistant, displayed)
            if not parsed.valid:
                failures.append(f"{prefix}: invalid_action:{','.join(parsed.errors)}")
                continue
            selected = parsed.value
            action_kinds[selected.kind] += 1
            action_kinds_by_split[expected_split][selected.kind] += 1
            action_positions[displayed.index(selected)] += 1
            if seed not in solver_cache:
                solver_cache[seed] = solve_joint_action(state, "BLUE", deterministic_policy(state, "RED"))
            solution = solver_cache[seed]
            if selected not in solution.acceptable_for(agent_id):
                failures.append(f"{prefix}: not_solver_optimal")
            if selected != local_policy_action(state, agent_id):
                failures.append(f"{prefix}: not_prompt_identifiable")
            for inbox_item in user.get("inbox", []):
                intent = inbox_item.get("broadcast", {}).get("intent")
                if intent is not None and _action(intent) == selected:
                    failures.append(f"{prefix}: conflicting_declared_intent")
        else:
            failures.append(f"{prefix}: unknown_phase")

    if any(len(splits) != 1 for splits in seeds_by_split.values()):
        failures.append("seed_crosses_splits")
    content_hash = hashlib.sha256("".join(sorted(row["id"] for row in rows)).encode()).hexdigest()
    if content_hash != manifest.get("content_sha256"):
        failures.append("manifest_content_hash")
    if len(rows) != manifest.get("num_examples"):
        failures.append("manifest_example_count")
    expected_versions = {
        "dataset_version": DATASET_VERSION,
        "arena_version": ARENA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "prompt_version": ARENA_PROMPT_VERSION,
    }
    for key, expected in expected_versions.items():
        if manifest.get(key) != expected:
            failures.append(f"manifest_{key}")
    actual_splits = Counter(row["metadata"]["split"] for row in rows)
    if dict(sorted(actual_splits.items())) != manifest.get("examples_by_split"):
        failures.append("manifest_split_counts")
    if dict(sorted(phases.items())) != manifest.get("examples_by_phase"):
        failures.append("manifest_phase_counts")
    required_actions = {"WAIT", "SCAN", "PROBE", "CAPTURE", "FORTIFY", "RECOVER", "TRANSFER"}
    missing_actions = sorted(required_actions - set(action_kinds))
    if missing_actions:
        failures.append("missing_action_coverage:" + ",".join(missing_actions))
    if require_split_action_coverage:
        for split, counts in action_kinds_by_split.items():
            missing = sorted(required_actions - set(counts))
            if missing:
                failures.append(f"missing_{split}_action_coverage:" + ",".join(missing))
    if broadcast_facts[0] == 0:
        failures.append("missing_silence_coverage")
    return {
        "valid": not failures,
        "failures": failures,
        "num_examples": len(rows),
        "num_unique_seeds": len(seeds_by_split),
        "phases": dict(sorted(phases.items())),
        "action_kinds": dict(sorted(action_kinds.items())),
        "action_kinds_by_split": {
            split: dict(sorted(counts.items())) for split, counts in action_kinds_by_split.items()
        },
        "action_positions": {str(key): value for key, value in sorted(action_positions.items())},
        "broadcast_fact_counts": {str(key): value for key, value in sorted(broadcast_facts.items())},
        "verified_solver_seeds": len(solver_cache),
        "required_split_action_coverage": require_split_action_coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit an arena SFT dataset.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-split-action-coverage", action="store_true")
    args = parser.parse_args()
    report = audit_dataset(args.path, args.require_split_action_coverage)
    output = args.output or args.path / "audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
