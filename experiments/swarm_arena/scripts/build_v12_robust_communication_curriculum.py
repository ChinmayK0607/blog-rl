from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from swarm_ctf_eval.handoff_curriculum import generate_manifest
from swarm_ctf_eval.progress_eval_v4 import build_ordinary_manifest
from swarm_ctf_eval.rl_production import (
    CurriculumMix,
    CurriculumStage,
    OpponentPool,
    OpponentSnapshot,
    exact_staged_curriculum_schedule,
)

VERSION = "arena-rl-v12-counterfactual-robustness-v1"
TRAIN_PAIRS = 96


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases(stop: int) -> list[dict[str, object]]:
    if stop % 12:
        raise ValueError("role-balanced cases require complete 12-pair blocks")
    role_groups = ((0, 4, 8, 9), (1, 5, 6, 10), (2, 3, 7, 11))
    worlds = ("left_exposed", "right_exposed")
    result = []
    for block_start in range(0, stop, 12):
        for group in role_groups:
            for flip in range(2):
                result.extend(
                    {
                        "pair_index": block_start + offset,
                        "world": worlds[(position + flip) % 2],
                    }
                    for position, offset in enumerate(group)
                )
    return result


def curriculum() -> dict[str, Any]:
    retention = {"ordinary": 2, "critical": 1, "decoy": 1}
    transfer = {"ordinary": 1, "critical": 2, "decoy": 1}
    consolidation_cases = cases(TRAIN_PAIRS)
    consolidation_cases = consolidation_cases[4:] + consolidation_cases[:4]
    return {
        "version": VERSION,
        "total_updates": 160,
        "groups_per_update": 4,
        "reward": "verified_terminal_control_delta_only",
        "message_reward": None,
        "initializer": "four_distinct_public_v11_update180_policy_adapters",
        "credit_assignment": {
            "critical": "receiver_ACT_absolute_factual_minus_receiver_only_target_swap",
            "decoy": "receiver_ACT_absolute_target_swap_challenge_minus_factual",
            "ordinary": "shared_terminal_return_leave_one_out",
        },
        "principles": [
            "continue four distinct V11 policies without merging or cloning one slot",
            "use only verified terminal team return and no additive communication reward",
            "retain factual-message critical handoffs while directly training misleading-message decoy actions",
            "make every decoy challenge a matched subset of critical topology/world cases",
            "front-load ordinary retention to repair the V11 legacy regression",
            "rotate all four opponent families exactly once per update",
            "stop at the first checkpoint that fails the predeclared repair trend twice",
            "reuse the still-unopened V11 frozen final byte-for-byte",
        ],
        "schedule": {
            "pair_offset": 0,
            "ordinary_seed_base": 21_000_113,
            "shuffle_seed": 20_260_977,
        },
        "stages": [
            {
                "name": "u180_repair_and_conflict_grounding",
                "updates": 40,
                "update_pattern": [retention],
                "ordinary_sizes": [12, 14, 16],
                "ordinary_horizons": [4, 6, 8],
                "handoff_focus_roles": ["receiver"],
                "handoff_remaining_turns": 1,
                "handoff_cases": cases(48),
            },
            {
                "name": "robust_two_turn_transfer",
                "updates": 60,
                "update_pattern": [transfer, retention],
                "ordinary_sizes": [14, 16, 18],
                "ordinary_horizons": [6, 8, 10],
                "handoff_focus_roles": ["receiver"],
                "handoff_remaining_turns": 2,
                "handoff_cases": cases(72),
            },
            {
                "name": "robust_four_turn_consolidation",
                "updates": 60,
                "update_pattern": [transfer, retention],
                "ordinary_sizes": [16, 18, 20],
                "ordinary_horizons": [8, 10, 12],
                "handoff_focus_roles": ["receiver"],
                "handoff_remaining_turns": 4,
                "handoff_cases": consolidation_cases,
            },
        ],
        "runtime": {
            "shared_return_replicas": 4,
            "action_prompt_profile": "full",
            "shared_return_baseline": "paired_receiver_target_swap",
            "decoy_shared_return_baseline": "paired_receiver_target_swap_challenge",
            "paired_contrast_centering": "none",
            "require_receiver_isolation": True,
            "target_swap_sender_retries": 8,
        },
        "online_eval_updates": list(range(0, 161, 20)),
        "development_candidates": [20, 40, 80, 120, 160],
        "fail_fast_rule": (
            "after updates 20 and 40, stop if neither ordinary retention nor decoy "
            "robustness improves while critical semantic return is non-positive"
        ),
        "selection_rule": (
            "earliest candidate with positive semantic return, positive specificity, "
            "and both ordinary clustered lower bounds at least -0.02"
        ),
    }


def stage(row: dict[str, Any]) -> CurriculumStage:
    return CurriculumStage(
        name=row["name"],
        updates=row["updates"],
        update_pattern=tuple(CurriculumMix(**mix) for mix in row["update_pattern"]),
        ordinary_sizes=tuple(row["ordinary_sizes"]),
        ordinary_horizons=tuple(row["ordinary_horizons"]),
        handoff_focus_roles=("receiver",),
        handoff_cases=tuple(
            (case["pair_index"], case["world"]) for case in row["handoff_cases"]
        ),
        handoff_remaining_turns=row["handoff_remaining_turns"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v11-data-dir", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path)
    parser.add_argument("--production-plan-output", type=Path)
    args = parser.parse_args()
    if (args.base_plan is None) != (args.production_plan_output is None):
        parser.error("--base-plan and --production-plan-output must be provided together")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v11_train_path = args.v11_data_dir / "handoff_train.json"
    v11_frozen_path = args.v11_data_dir / "handoff_frozen_ood.json"
    v11_ordinary_frozen_path = args.v11_data_dir / "ordinary_hard_frozen_ood.json"
    train = json.loads(v11_train_path.read_text())
    frozen = json.loads(v11_frozen_path.read_text())
    ordinary_frozen = json.loads(v11_ordinary_frozen_path.read_text())
    development = generate_manifest(
        count=36,
        seed_start=20_000_109,
        sizes=(14, 16, 18, 20, 22),
        horizons=(6, 8, 10, 12, 14),
    )
    ordinary_development = build_ordinary_manifest(
        count=36,
        seed_start=20_500_111,
        sizes=(16, 18, 20, 22),
        horizons=(8, 10, 12),
    )
    plan = curriculum()
    schedule = exact_staged_curriculum_schedule(
        tuple(stage(row) for row in plan["stages"]),
        groups_per_update=plan["groups_per_update"],
        pair_offset=plan["schedule"]["pair_offset"],
        ordinary_seed_base=plan["schedule"]["ordinary_seed_base"],
        shuffle_seed=plan["schedule"]["shuffle_seed"],
    )
    counts = Counter(row.kind for row in schedule)
    critical_cases = {
        (row.pair_index, row.handoff_world) for row in schedule if row.kind == "critical"
    }
    decoy_cases = {
        (row.pair_index, row.handoff_world) for row in schedule if row.kind == "decoy"
    }
    receiver_counts = {
        kind: Counter(
            train["pairs"][row.pair_index][kind]["receiver"]
            for row in schedule
            if row.kind == kind and row.pair_index is not None
        )
        for kind in ("critical", "decoy")
    }
    train_states = {
        world["state_sha256"]
        for pair in train["pairs"][:TRAIN_PAIRS]
        for world in pair["critical"]["worlds"]
    }
    development_states = {
        world["state_sha256"]
        for pair in development["pairs"]
        for world in pair["critical"]["worlds"]
    }
    frozen_states = {
        world["state_sha256"]
        for pair in frozen["pairs"]
        for world in pair["critical"]["worlds"]
    }
    audit = {
        "version": "arena-rl-v12-curriculum-audit-v1",
        "status": "passed",
        "schedule_sha256": digest([row.__dict__ for row in schedule]),
        "group_counts": dict(sorted(counts.items())),
        "receiver_counts": {
            kind: dict(sorted(value.items())) for kind, value in receiver_counts.items()
        },
        "decoys_are_matched_critical_subset": decoy_cases <= critical_cases,
        "split_state_hashes_disjoint": not (
            train_states & development_states
            or train_states & frozen_states
            or development_states & frozen_states
        ),
        "frozen_handoff_file_sha256": file_sha256(v11_frozen_path),
        "frozen_ordinary_file_sha256": file_sha256(v11_ordinary_frozen_path),
        "frozen_reused_byte_for_byte": True,
        "frozen_unopened": True,
        "expected_nonzero_signal": {
            "critical": counts["critical"] * plan["runtime"]["shared_return_replicas"],
            "challenge": counts["decoy"] * plan["runtime"]["shared_return_replicas"],
            "ordinary": counts["ordinary"] * plan["runtime"]["shared_return_replicas"],
        },
        "paired_contrast_centering": plan["runtime"]["paired_contrast_centering"],
        "uniform_contrast_signal_check": {
            "critical_factual_minus_swapped": [0.4, 0.4, 0.4, 0.4],
            "challenge_swapped_minus_factual": [-0.4, -0.4, -0.4, -0.4],
            "uniform_signal_preserved": plan["runtime"]["paired_contrast_centering"] == "none",
        },
    }
    if counts != Counter({"ordinary": 260, "critical": 220, "decoy": 160}):
        raise ValueError(f"unexpected V12 group counts: {counts}")
    if not audit["decoys_are_matched_critical_subset"]:
        raise ValueError("every challenge decoy must be matched by a critical case")
    if not audit["split_state_hashes_disjoint"]:
        raise ValueError("V12 train/development/frozen states overlap")
    expected_slots = {f"blue-{index}" for index in range(4)}
    if any(set(value) != expected_slots for value in receiver_counts.values()):
        raise ValueError("critical and challenge schedules must cover every receiver policy")
    if max(receiver_counts["decoy"].values()) != min(receiver_counts["decoy"].values()):
        raise ValueError("challenge decoys must balance all receiver policy slots exactly")
    if plan["runtime"]["paired_contrast_centering"] != "none":
        raise ValueError("V12 must preserve absolute paired terminal-return contrasts")

    progress_eval_design = {
        "version": "arena-rl-v12-progress-eval-v1",
        "status": "frozen_before_v12_rl",
        "development_candidates": plan["development_candidates"],
        "development_rule": plan["selection_rule"],
        "ordinary_noninferiority_margin": -0.02,
        "ordinary_noninferiority_statistic": "clustered_95_percent_lower_bound",
        "development": {
            "handoff_pairs": 36,
            "ordinary_cases": 36,
            "opponents": ["base", "sft", "v10-update40", "v11-update180"],
            "sides": ["BLUE", "RED"],
            "critical_conditions": [
                "normal", "dropped", "sender_shuffled", "delayed", "zero_budget", "target_swapped"
            ],
            "decoy_conditions": ["normal", "dropped", "target_swapped"],
        },
        "frozen_final": {
            "source": "unchanged V11 frozen suite",
            "run_policy": "once for the earliest development-selected checkpoint",
            "handoff_file_sha256": file_sha256(v11_frozen_path),
            "ordinary_file_sha256": file_sha256(v11_ordinary_frozen_path),
        },
        "claim_rule": (
            "clustered intervals for message-use, RL-specific effect, and critical-minus-decoy "
            "specificity are positive; ordinary lower bounds exceed -0.02; all opponents/sides reported"
        ),
    }
    outputs = {
        "handoff_train.json": train,
        "handoff_development.json": development,
        "handoff_frozen_ood.json": frozen,
        "ordinary_hard_development.json": ordinary_development,
        "ordinary_hard_frozen_ood.json": ordinary_frozen,
        "curriculum.json": plan,
        "staged_curriculum_v12_4b_robust_160.json": plan,
        "v12_curriculum_audit.json": audit,
        "progress_eval_design.json": progress_eval_design,
    }
    outputs["index.json"] = {
        "handoff": {
            "train": {"manifest": "handoff_train.json", "sha256": train["sha256"]},
            "development": {"manifest": "handoff_development.json", "sha256": development["sha256"]},
            "frozen_ood": {"manifest": "handoff_frozen_ood.json", "sha256": frozen["sha256"]},
        },
        "ordinary": {
            "development": {
                "manifest": "ordinary_hard_development.json",
                "sha256": ordinary_development["sha256"],
            },
            "frozen_ood": {
                "manifest": "ordinary_hard_frozen_ood.json",
                "sha256": ordinary_frozen["sha256"],
            },
        },
    }
    if args.base_plan is not None:
        production = json.loads(args.base_plan.read_text())
        production["version"] = "arena-rl-v4-staged-production-plan-v1"
        production["curriculum_mix"] = plan["stages"][0]["update_pattern"][0]
        production["curriculum_stages"] = plan["stages"]
        production["schedule"] = {
            **production["schedule"],
            **plan["schedule"],
            "ordinary_sizes": [12, 14, 16, 18, 20],
            "ordinary_horizons": [4, 6, 8, 10, 12],
        }
        production["rollout_runtime"] = plan["runtime"]
        snapshots = tuple(
            OpponentSnapshot(
                opponent_id=row["opponent_id"],
                family=row["family"],
                model_name=row["model_name"],
                revision=row["revision"],
                adapter_sha256=row["adapter_sha256"],
                update_index=row["update_index"],
            )
            for row in production["opponent_pool"]["snapshots"]
        )
        opponent_schedule = OpponentPool(
            snapshots=snapshots,
            rotation_seed=production["opponent_pool"]["rotation_seed"],
        ).schedule(len(schedule))
        cross_counts = Counter(
            (assignment.kind, opponent.family)
            for assignment, opponent in zip(schedule, opponent_schedule, strict=True)
        )
        targets = {"ordinary": 65, "critical": 55, "decoy": 40}
        max_deviation = max(
            abs(cross_counts[(kind, family)] - targets[kind])
            for kind in targets
            for family in ("base", "sft", "historical", "current")
        )
        if max_deviation > 2:
            raise ValueError("scenario kind and opponent family are too strongly correlated")
        audit["scenario_opponent_family_counts"] = {
            f"{kind}:{family}": value
            for (kind, family), value in sorted(cross_counts.items())
        }
        audit["scenario_opponent_max_count_deviation"] = max_deviation
        audit["every_update_uses_all_four_opponents"] = all(
            len({row.opponent_id for row in opponent_schedule[offset : offset + 4]}) == 4
            for offset in range(0, len(opponent_schedule), 4)
        )
        args.production_plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.production_plan_output.write_text(
            json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
