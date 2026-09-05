from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from swarm_ctf_eval.communication_curriculum import generate_manifest


SPLITS = {
    "train": {"count": 240, "seed_start": 3_000_003, "sizes": (12, 13)},
    "development": {"count": 48, "seed_start": 4_000_007, "sizes": (12, 13)},
    "frozen_ood": {"count": 72, "seed_start": 5_000_011, "sizes": (14, 16)},
}

CURRICULUM_SCHEDULE = {
    "version": "arena-rl-v3-curriculum-v1",
    "reward": "terminal_control_delta_only",
    "promotion_rule": "all gates must pass on the frozen development split",
    "stages": [
        {
            "stage": 1,
            "status": "ready",
            "horizon": 1,
            "mixture": {"certified_critical": 0.5, "matched_decoy": 0.5},
            "gates": {
                "protocol_validity_min": 0.995,
                "critical_normal_minus_dropped_min": 0.02,
                "decoy_normal_minus_dropped_abs_max": 0.01,
                "min_opponents_passed": 3,
            },
        },
        {
            "stage": 2,
            "status": "planned_after_rollout_integration",
            "horizon": [2, 3],
            "mixture": {
                "certified_critical": 0.25,
                "matched_decoy": 0.25,
                "ordinary_procedural": 0.5,
            },
            "gates": {
                "critical_all_intervention_effects_positive": True,
                "ordinary_return_non_regression": True,
                "no_single_opponent_failure": True,
            },
        },
        {
            "stage": 3,
            "status": "planned_after_rollout_integration",
            "horizon": [4, 6],
            "mixture": {
                "certified_critical": 0.125,
                "matched_decoy": 0.125,
                "ordinary_procedural": 0.75,
            },
            "gates": {
                "frozen_ood_capability_improves": True,
                "communication_effect_persists": True,
                "regression_suites_pass": True,
            },
        },
    ],
}


def audit(manifest: dict) -> dict:
    critical = [pair["critical"] for pair in manifest["pairs"]]
    decoys = [pair["decoy"] for pair in manifest["pairs"]]
    advantages = [float(row["minimum_advantage"]) for row in critical]
    role_counts = Counter((row["sender"], row["receiver"]) for row in critical)
    pair_audits = [pair["matched_pair_audit"] for pair in manifest["pairs"]]
    return {
        "manifest_sha256": manifest["sha256"],
        "pair_count": len(critical),
        "critical_minimum_advantage": {
            "minimum": min(advantages),
            "mean": statistics.mean(advantages),
            "maximum": max(advantages),
            "positive_rate": statistics.mean(value > 0 for value in advantages),
        },
        "decoy_zero_advantage_rate": statistics.mean(
            abs(float(row["minimum_advantage"])) <= 1e-12 for row in decoys
        ),
        "ordered_role_pairs": {
            f"{sender}->{receiver}": count
            for (sender, receiver), count in sorted(role_counts.items())
        },
        "role_balance_range": [min(role_counts.values()), max(role_counts.values())],
        "matched_pair_invariants": {
            "structural_state_identical_rate": statistics.mean(
                bool(row["structural_state_identical"]) for row in pair_audits
            ),
            "sender_observation_identical_rate": statistics.mean(
                bool(row["sender_observation_identical"]) for row in pair_audits
            ),
            "only_receiver_target_knowledge_changes_rate": statistics.mean(
                bool(row["only_receiver_target_knowledge_changes"])
                for row in pair_audits
            ),
        },
        "opponent_styles": ["balanced", "aggressive", "defensive"],
        "certificate_scope": "exact best joint action against each frozen deterministic style",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic RL v3 communication manifests.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "curriculum_schedule": "curriculum_schedule.json",
        "splits": {},
    }
    for split, config in SPLITS.items():
        manifest = generate_manifest(**config)
        report = audit(manifest)
        manifest_path = args.output_dir / f"{split}.json"
        audit_path = args.output_dir / f"{split}_audit.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        index["splits"][split] = {
            "manifest": manifest_path.name,
            "audit": audit_path.name,
            "sha256": manifest["sha256"],
            "pair_count": manifest["pair_count"],
            "sizes": manifest["sizes"],
        }
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "curriculum_schedule.json").write_text(
        json.dumps(CURRICULUM_SCHEDULE, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
