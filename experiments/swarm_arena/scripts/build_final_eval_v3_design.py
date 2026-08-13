from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swarm_ctf_eval.episode_splits import (
    EPISODE_EVAL_CASES,
    EPISODE_EVAL_MANIFEST_SHA256,
)
from swarm_ctf_eval.final_eval_v3 import FINAL_EVAL_VERSION


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_design(data_dir: Path) -> dict:
    critical_manifest = data_dir / "frozen_ood.json"
    critical = json.loads(critical_manifest.read_text(encoding="utf-8"))
    return {
        "version": FINAL_EVAL_VERSION,
        "status": "frozen_before_rl",
        "independent_unit": "case_id/game seed",
        "bootstrap": {"trials": 20_000, "seed": 0, "interval": 0.95},
        "frozen_sources": {
            "ordinary_episode_v2": {
                "cases": len(EPISODE_EVAL_CASES),
                "manifest_sha256": EPISODE_EVAL_MANIFEST_SHA256,
            },
            "certified_critical_decoy_v3": {
                "pairs": critical["pair_count"],
                "manifest_body_sha256": critical["sha256"],
                "file_sha256": _file_sha256(critical_manifest),
            },
            "non_arena_regression": {
                "suites": ["swarm-regression-v1", "swarm-regression-v2"],
                "selection_gate": "no_overall_drop_gt_0.02_and_no_category_drop_gt_0.05",
            },
        },
        "runtime_bindings": {
            "candidate_rl": "required immutable model plus four adapter revisions",
            "sft_init": "required immutable warm-start model plus four initial adapters",
            "action_only_rl": "required control trained on the same games with communication disabled",
            "opponent_slots": ["base", "sft", "historical_league"],
            "opponent_artifact_ids": "must be filled before launching; aliases are rejected",
        },
        "headline_matrix": {
            "ordinary_capability": {
                "cases": 72,
                "sides": 2,
                "opponents": 3,
                "policy_variants": ["candidate_rl", "sft_init"],
                "conditions": ["normal"],
                "required_games": 864,
            },
            "action_only_control": {
                "cases": 72,
                "sides": 2,
                "opponents": 3,
                "policy_variants": ["action_only_rl"],
                "conditions": ["normal"],
                "required_games": 432,
            },
            "critical_communication": {
                "cases": 72,
                "sides": 2,
                "opponents": 3,
                "policy_variants": ["candidate_rl"],
                "conditions": ["normal", "dropped", "sender_shuffled", "delayed", "zero_budget"],
                "required_games": 2160,
            },
            "matched_decoy": {
                "cases": 72,
                "sides": 2,
                "opponents": 3,
                "policy_variants": ["candidate_rl"],
                "conditions": ["normal", "dropped"],
                "required_games": 864,
            },
        },
        "mechanism_matrix": {
            "fixed_case_subset": 12,
            "selection": "first case for each ordered sender/receiver identity",
            "adapter_assignments": "identity plus all 23 perm-<0123> non-identity permutations",
            "role_label_assignments": "identity plus all 23 perm-<0123> non-identity permutations",
            "option_orders": ["canonical", "permuted-1", "permuted-2", "permuted-3"],
            "note": "Mechanism rows are secondary and never enter headline capability confidence intervals.",
        },
        "claim_gates": {
            "capability": "RL-minus-SFT ordinary OOD 95% interval is positive",
            "communication": (
                "all four critical intervention intervals are positive, their means are positive "
                "against every opponent, both side swaps are complete, and the matched-decoy "
                "normal-minus-dropped interval includes zero"
            ),
            "specialization": "identity-minus-adapter-shuffle 95% interval is positive",
            "regression": "both frozen non-arena suites pass their existing gates",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Swarm Arena final-eval v3 design.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = build_design(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(design, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
