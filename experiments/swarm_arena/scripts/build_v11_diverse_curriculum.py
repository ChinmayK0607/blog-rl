from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from swarm_ctf_eval.handoff_curriculum import generate_manifest
from swarm_ctf_eval.progress_eval_v4 import build_ordinary_manifest
from swarm_ctf_eval.rl_production import (
    CurriculumMix,
    CurriculumStage,
    exact_staged_curriculum_schedule,
)

VERSION = "arena-rl-v11-diverse-receiver-curriculum-v1"
TRAIN_PAIR_STOP = 96
DEVELOPMENT_PAIR_STOP = 120
ONLINE_EVAL_PAIR_INDICES = (96, 99, 104, 107)
SELECTION_EVAL_PAIR_INDICES = tuple(range(96, 108))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cases(stop: int) -> list[dict[str, object]]:
    if stop % 12:
        raise ValueError("role-balanced handoff case sets require complete 12-pair blocks")
    role_groups = ((0, 4, 8, 9), (1, 5, 6, 10), (2, 3, 7, 11))
    worlds = ("left_exposed", "right_exposed")
    cases = []
    for block_start in range(0, stop, 12):
        for group in role_groups:
            for flip in range(2):
                cases.extend(
                    {
                        "pair_index": block_start + pair_offset,
                        "world": worlds[(position + flip) % 2],
                    }
                    for position, pair_offset in enumerate(group)
                )
    return cases


def _curriculum() -> dict[str, object]:
    mix = [{"ordinary": 1, "critical": 3, "decoy": 0}]
    return {
        "version": VERSION,
        "total_updates": 180,
        "groups_per_update": 4,
        "reward": "verified_terminal_control_delta_only",
        "credit_assignment": (
            "receiver_act_centered_actual_minus_receiver_only_target_swapped_terminal_return"
        ),
        "message_reward": None,
        "initializer": "pinned_sft_not_prior_rl",
        "scope": (
            "96 independent training topologies with role-size-horizon decorrelation, "
            "receiver-only semantic credit, and ordinary preservation"
        ),
        "principles": [
            "train only on certified critical handoffs and reserve matched decoys for evaluation",
            "change only the credited receiver's target fact across paired branches",
            "use verified terminal team return only and train only the receiver ACT span",
            "cover every ordered sender-receiver role at every training graph size",
            "increase remaining horizon only after terminal-proximal learnability",
            "allocate one ordinary preservation group in every update",
            "reserve independent development and frozen-final seeds",
        ],
        "stages": [
            {
                "name": "diverse_terminal_grounding",
                "updates": 32,
                "update_pattern": mix,
                "ordinary_sizes": [12, 14],
                "ordinary_horizons": [4, 5],
                "handoff_focus_roles": ["receiver"],
                "handoff_remaining_turns": 1,
                "handoff_cases": _cases(36),
            },
            {
                "name": "diverse_two_turn_transfer",
                "updates": 60,
                "update_pattern": mix,
                "ordinary_sizes": [14, 16],
                "ordinary_horizons": [6, 8],
                "handoff_focus_roles": ["receiver"],
                "handoff_remaining_turns": 2,
                "handoff_cases": _cases(72),
            },
            {
                "name": "diverse_four_turn_transfer",
                "updates": 88,
                "update_pattern": mix,
                "ordinary_sizes": [16, 18],
                "ordinary_horizons": [8, 10],
                "handoff_focus_roles": ["receiver"],
                "handoff_remaining_turns": 4,
                "handoff_cases": _cases(TRAIN_PAIR_STOP),
            },
        ],
        "runtime": {
            "shared_return_replicas": 4,
            "action_prompt_profile": "full",
            "shared_return_baseline": "paired_receiver_target_swap",
            "require_receiver_isolation": True,
            "online_evaluation_mode": "multipair",
            "online_eval_remaining_turns": 4,
        },
        "online_eval_updates": list(range(0, 181, 20)),
        "online_eval_pair_indices": list(ONLINE_EVAL_PAIR_INDICES),
        "selection_eval_updates": [60, 120, 180],
        "selection_eval_pair_indices": list(SELECTION_EVAL_PAIR_INDICES),
        "primary_success_measure": (
            "development critical normal-minus-receiver-target-swapped terminal return"
        ),
        "specificity_measure": (
            "critical-minus-matched-decoy receiver-target-swap lift"
        ),
        "checkpoint_selection": (
            "earliest checkpoint jointly improving semantic return, matched-decoy "
            "specificity, and ordinary capability on development"
        ),
        "frozen_policy": (
            "the v11 frozen manifest remains unopened until one checkpoint is selected"
        ),
    }


def _stage(row: dict[str, object]) -> CurriculumStage:
    return CurriculumStage(
        name=str(row["name"]),
        updates=int(row["updates"]),
        update_pattern=tuple(
            CurriculumMix(**mix) for mix in row["update_pattern"]  # type: ignore[arg-type]
        ),
        ordinary_sizes=tuple(int(value) for value in row["ordinary_sizes"]),  # type: ignore[arg-type]
        ordinary_horizons=tuple(int(value) for value in row["ordinary_horizons"]),  # type: ignore[arg-type]
        handoff_focus_roles=("receiver",),
        handoff_cases=tuple(
            (int(case["pair_index"]), str(case["world"]))
            for case in row["handoff_cases"]  # type: ignore[union-attr]
        ),
        handoff_remaining_turns=int(row["handoff_remaining_turns"]),
    )


def _role_coverage(manifest: dict, pair_indices: range) -> dict[str, object]:
    sizes: dict[str, set[int]] = defaultdict(set)
    horizons: dict[str, set[int]] = defaultdict(set)
    counts: Counter[str] = Counter()
    state_hashes: set[str] = set()
    for pair_index in pair_indices:
        critical = manifest["pairs"][pair_index]["critical"]
        role = f'{critical["sender"]}->{critical["receiver"]}'
        counts[role] += 1
        sizes[role].add(int(critical["size"]))
        horizons[role].add(int(critical["horizon"]))
        state_hashes.update(world["state_sha256"] for world in critical["worlds"])
    return {
        "pair_count": len(pair_indices),
        "role_counts": dict(sorted(counts.items())),
        "sizes_per_role": {role: sorted(values) for role, values in sorted(sizes.items())},
        "horizons_per_role": {
            role: sorted(values) for role, values in sorted(horizons.items())
        },
        "unique_world_state_hashes": len(state_hashes),
    }


def _evaluation_design(
    *,
    train_manifest_sha256: str,
    frozen_manifest_sha256: str,
    ordinary_development_sha256: str,
    ordinary_frozen_sha256: str,
) -> dict[str, object]:
    return {
        "version": "arena-rl-v11-progress-eval-v1",
        "status": "frozen_before_v11_rl",
        "independent_units": {
            "handoff": "two-world latent bundle",
            "ordinary": "procedural game seed",
        },
        "data_bindings": {
            "train_and_development": train_manifest_sha256,
            "frozen_handoff": frozen_manifest_sha256,
            "development_ordinary": ordinary_development_sha256,
            "frozen_ordinary": ordinary_frozen_sha256,
        },
        "online_pulse": {
            "updates": list(range(0, 181, 20)),
            "handoff_pair_indices": list(ONLINE_EVAL_PAIR_INDICES),
            "opponent": "sft",
            "conditions": ["normal", "dropped", "sender_shuffled", "target_swapped"],
            "purpose": "directional monitoring only",
        },
        "development_selection": {
            "updates": [60, 120, 180],
            "handoff_pair_indices": list(SELECTION_EVAL_PAIR_INDICES),
            "ordinary_cases": 24,
            "opponents": ["base", "sft", "v10-update40"],
            "critical_conditions": [
                "normal",
                "dropped",
                "sender_shuffled",
                "delayed",
                "zero_budget",
                "target_swapped",
            ],
            "decoy_conditions": ["normal", "dropped", "target_swapped"],
            "selection_rule": (
                "select the earliest checkpoint with positive semantic return, positive "
                "critical-minus-decoy specificity, and non-regressing ordinary return"
            ),
        },
        "frozen_final": {
            "run_policy": "once for the single development-selected checkpoint",
            "handoff_pairs": 36,
            "ordinary_cases": 36,
            "opponents": ["base", "sft", "v10-update40"],
            "sides": ["BLUE", "RED"],
            "critical_conditions": [
                "normal",
                "dropped",
                "sender_shuffled",
                "delayed",
                "zero_budget",
                "target_swapped",
            ],
            "decoy_conditions": ["normal", "dropped", "target_swapped"],
        },
        "headline_endpoints": {
            "message_use": "candidate critical normal minus receiver-only target-swapped return",
            "rl_specific": "candidate semantic effect minus frozen SFT semantic effect",
            "specificity": "critical semantic effect minus matched-decoy semantic effect",
            "receiver_behavior": "normal minus target-swapped receiver target-action rate",
            "capability": "candidate minus SFT ordinary and overall terminal return",
        },
        "claim_rule": (
            "message-use, RL-specific, and critical-minus-decoy clustered 95% intervals "
            "must be positive; ordinary capability must not regress; report every "
            "opponent and side without post-hoc exclusions"
        ),
        "diagnostics_not_claim_gates": [
            "dropped-message effect",
            "delayed-message effect",
            "sender-shuffled effect",
            "zero-budget effect",
        ],
        "mechanical_requirements": [
            "action, broadcast, and grounding validity equal 1.0",
            "constrained KL and collapse audits pass",
            "no missing, duplicated, or orphan rollout IDs",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the role/topology-diverse v11 training and evaluation package."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_and_development = generate_manifest(
        count=DEVELOPMENT_PAIR_STOP,
        seed_start=15_000_083,
        sizes=(12, 14, 16, 18, 20),
        horizons=(4, 5, 6, 8, 10, 12, 14),
    )
    frozen = generate_manifest(
        count=36,
        seed_start=17_000_099,
        sizes=(16, 18, 20, 22, 24),
        horizons=(8, 9, 10, 12, 14, 16, 18),
    )
    ordinary_development = build_ordinary_manifest(
        count=24,
        seed_start=16_500_097,
        sizes=(16, 18, 20),
        horizons=(8, 10, 12),
    )
    ordinary_frozen = build_ordinary_manifest(
        count=36,
        seed_start=17_500_103,
        sizes=(18, 20, 22, 24),
        horizons=(10, 12, 14),
    )
    curriculum = _curriculum()
    stages = tuple(_stage(row) for row in curriculum["stages"])  # type: ignore[arg-type]
    schedule = exact_staged_curriculum_schedule(
        stages,
        groups_per_update=4,
        pair_offset=0,
        ordinary_seed_base=18_000_107,
        shuffle_seed=20_260_824,
    )

    train_coverage = _role_coverage(train_and_development, range(TRAIN_PAIR_STOP))
    development_coverage = _role_coverage(
        train_and_development, range(TRAIN_PAIR_STOP, DEVELOPMENT_PAIR_STOP)
    )
    train_states = {
        world["state_sha256"]
        for pair in train_and_development["pairs"][:TRAIN_PAIR_STOP]
        for world in pair["critical"]["worlds"]
    }
    development_states = {
        world["state_sha256"]
        for pair in train_and_development["pairs"][TRAIN_PAIR_STOP:]
        for world in pair["critical"]["worlds"]
    }
    frozen_states = {
        world["state_sha256"]
        for pair in frozen["pairs"]
        for world in pair["critical"]["worlds"]
    }
    required_sizes = {12, 14, 16, 18, 20}
    role_size_passed = all(
        set(values) == required_sizes
        for values in train_coverage["sizes_per_role"].values()
    )
    role_counts = Counter(
        train_and_development["pairs"][row.pair_index]["critical"]["receiver"]
        for row in schedule
        if row.kind == "critical" and row.pair_index is not None
    )
    audit = {
        "version": "arena-rl-v11-diverse-data-audit-v1",
        "status": "passed",
        "train_and_development_manifest_sha256": train_and_development["sha256"],
        "frozen_manifest_sha256": frozen["sha256"],
        "ordinary_development_sha256": ordinary_development["sha256"],
        "ordinary_frozen_sha256": ordinary_frozen["sha256"],
        "curriculum_sha256": _digest(curriculum),
        "schedule_sha256": _digest([row.__dict__ for row in schedule]),
        "schedule_group_counts": dict(sorted(Counter(row.kind for row in schedule).items())),
        "receiver_policy_slot_counts": dict(sorted(role_counts.items())),
        "training": train_coverage,
        "development": development_coverage,
        "frozen_pair_count": frozen["pair_count"],
        "role_size_decorrelation_passed": role_size_passed,
        "split_state_hashes_disjoint": not (
            train_states & development_states
            or train_states & frozen_states
            or development_states & frozen_states
        ),
        "online_eval_pair_indices": list(ONLINE_EVAL_PAIR_INDICES),
        "selection_eval_pair_indices": list(SELECTION_EVAL_PAIR_INDICES),
        "frozen_unopened": True,
    }
    if not role_size_passed:
        raise ValueError("every ordered role must cover every training graph size")
    if not audit["split_state_hashes_disjoint"]:
        raise ValueError("v11 train/development/frozen state hashes overlap")
    if set(role_counts) != {f"blue-{index}" for index in range(4)}:
        raise ValueError("the v11 schedule must cover all receiver policy slots")
    if max(role_counts.values()) != min(role_counts.values()):
        raise ValueError("the v11 schedule must balance receiver policy slots exactly")

    outputs = {
        "handoff_train.json": train_and_development,
        "handoff_frozen_ood.json": frozen,
        "ordinary_hard_development.json": ordinary_development,
        "ordinary_hard_frozen_ood.json": ordinary_frozen,
        "staged_curriculum_v11_4b_diverse_receiver_180.json": curriculum,
        "v11_diverse_data_audit.json": audit,
        "progress_eval_design.json": _evaluation_design(
            train_manifest_sha256=train_and_development["sha256"],
            frozen_manifest_sha256=frozen["sha256"],
            ordinary_development_sha256=ordinary_development["sha256"],
            ordinary_frozen_sha256=ordinary_frozen["sha256"],
        ),
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
