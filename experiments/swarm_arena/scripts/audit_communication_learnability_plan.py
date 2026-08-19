from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from swarm_ctf_eval.rl_production import (
    CurriculumMix,
    CurriculumStage,
    exact_staged_curriculum_schedule,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(row: dict[str, Any]) -> CurriculumStage:
    return CurriculumStage(
        name=str(row["name"]),
        updates=int(row["updates"]),
        update_pattern=tuple(CurriculumMix(**mix) for mix in row["update_pattern"]),
        ordinary_sizes=tuple(int(value) for value in row["ordinary_sizes"]),
        ordinary_horizons=tuple(int(value) for value in row["ordinary_horizons"]),
        handoff_focus_roles=tuple(str(value) for value in row["handoff_focus_roles"]),
        handoff_cases=tuple((int(case["pair_index"]), str(case["world"])) for case in row.get("handoff_cases", [])),
        handoff_remaining_turns=(
            None if row.get("handoff_remaining_turns") is None else int(row["handoff_remaining_turns"])
        ),
    )


def audit(curriculum_path: Path, handoff_path: Path) -> dict[str, Any]:
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    stages = tuple(_stage(row) for row in curriculum["stages"])
    groups_per_update = int(curriculum["groups_per_update"])
    total_updates = int(curriculum["total_updates"])
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=groups_per_update,
        pair_offset=0,
        ordinary_seed_base=8_000_000,
        shuffle_seed=17,
    )
    replicas = int(curriculum.get("runtime", {}).get("shared_return_replicas", 4))
    baseline = str(curriculum.get("runtime", {}).get("shared_return_baseline", "leave_one_out_mean"))
    selected_pairs = sorted({row.pair_index for row in schedule if row.pair_index is not None})
    pair_rows = []
    for pair_index in selected_pairs:
        pair = handoff["pairs"][pair_index]
        critical = pair["critical"]
        audit_row = pair["matched_pair_audit"]
        worlds = {world["label"]: world["active_target"] for world in critical["worlds"]}
        pair_rows.append(
            {
                "pair_index": pair_index,
                "sender": critical["sender"],
                "receiver": critical["receiver"],
                "world_targets": worlds,
                "minimum_certified_terminal_advantage": min(
                    float(row["advantage"]) for row in critical["certificates"]
                ),
                "private_worlds_indistinguishable": bool(
                    audit_row["critical_receiver_worlds_indistinguishable_without_message"]
                ),
                "legal_actions_match": bool(audit_row["receiver_action_sets_match_across_worlds"]),
                "message_does_not_unlock_action": bool(audit_row["message_does_not_change_receiver_legal_actions"]),
            }
        )

    update_blocks = [
        schedule[index : index + groups_per_update] for index in range(0, len(schedule), groups_per_update)
    ]
    expected_cases = {(row.pair_index, row.handoff_world) for row in schedule if row.kind == "critical"}
    every_update_covers_all_cases = all(
        {(row.pair_index, row.handoff_world) for row in block if row.kind == "critical"} == expected_cases
        for block in update_blocks
    )
    counts = Counter(row.kind for row in schedule)
    valid = (
        len(schedule) == total_updates * groups_per_update
        and every_update_covers_all_cases
        and counts.get("critical", 0) > 0
        and baseline in {"leave_one_out_mean", "paired_message_drop"}
        and all(
            row["private_worlds_indistinguishable"]
            and row["legal_actions_match"]
            and row["message_does_not_unlock_action"]
            and row["minimum_certified_terminal_advantage"] > 0
            for row in pair_rows
        )
    )
    return {
        "version": "swarm-communication-learnability-plan-audit-v1",
        "status": "passed" if valid else "failed",
        "curriculum": str(curriculum_path),
        "curriculum_sha256": _sha256(curriculum_path),
        "handoff_manifest_sha256": _sha256(handoff_path),
        "updates": total_updates,
        "groups_per_update": groups_per_update,
        "group_counts": dict(sorted(counts.items())),
        "shared_return_replicas": replicas,
        "focused_receiver_samples": len(schedule) * replicas,
        "action_prompt_profile": curriculum.get("runtime", {}).get("action_prompt_profile", "full"),
        "shared_return_baseline": baseline,
        "selected_cases": sorted(expected_cases),
        "every_update_covers_all_selected_cases": every_update_covers_all_cases,
        "pairs": pair_rows,
        "reward_contract": curriculum["reward"],
        "message_reward": curriculum["message_reward"],
        "interpretation": (
            "This certifies task construction and terminal-reward opportunity, not that "
            "sampled policy batches will contain non-zero advantage. The latter remains a "
            "first-update GPU diagnostic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a communication-learnability curriculum before paid rollout.")
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.curriculum, args.handoff_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
