from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swarm_ctf_eval.episode_splits import EPISODE_EVAL_MANIFEST_SHA256, EPISODE_EVAL_SEEDS


def _body_digest(manifest: dict) -> str:
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _seeds(manifest: dict) -> set[int]:
    return {int(pair["critical"]["seed"]) for pair in manifest["pairs"]}


def audit(data_dir: Path) -> dict:
    manifests = {
        split: json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "development", "frozen_ood")
    }
    seed_sets = {split: _seeds(manifest) for split, manifest in manifests.items()}
    overlaps = {
        "train_development": sorted(seed_sets["train"] & seed_sets["development"]),
        "train_frozen_ood": sorted(seed_sets["train"] & seed_sets["frozen_ood"]),
        "development_frozen_ood": sorted(
            seed_sets["development"] & seed_sets["frozen_ood"]
        ),
        "rl_v3_frozen_episode_v2": sorted(
            set().union(*seed_sets.values()) & set(EPISODE_EVAL_SEEDS)
        ),
    }
    split_reports = {}
    for split, manifest in manifests.items():
        pairs = manifest["pairs"]
        invariant_rows = [pair["matched_pair_audit"] for pair in pairs]
        role_counts: dict[str, int] = {}
        for pair in pairs:
            critical = pair["critical"]
            role = f"{critical['sender']}->{critical['receiver']}"
            role_counts[role] = role_counts.get(role, 0) + 1
        split_reports[split] = {
            "body_hash_valid": _body_digest(manifest) == manifest["sha256"],
            "unique_seed_count": len(seed_sets[split]),
            "pair_count": manifest["pair_count"],
            "all_critical_advantages_positive": all(
                float(pair["critical"]["minimum_advantage"]) > 0 for pair in pairs
            ),
            "all_decoy_advantages_zero": all(
                abs(float(pair["decoy"]["minimum_advantage"])) <= 1e-12 for pair in pairs
            ),
            "all_structural_states_matched": all(
                bool(row["structural_state_identical"]) for row in invariant_rows
            ),
            "all_sender_observations_matched": all(
                bool(row["sender_observation_identical"]) for row in invariant_rows
            ),
            "only_receiver_target_knowledge_changes": all(
                bool(row["only_receiver_target_knowledge_changes"])
                for row in invariant_rows
            ),
            "ordered_role_pair_counts": dict(sorted(role_counts.items())),
            "role_balance_range": [min(role_counts.values()), max(role_counts.values())],
        }
    final_design = json.loads((data_dir / "final_eval_design.json").read_text(encoding="utf-8"))
    final_sources = final_design["frozen_sources"]
    final_design_checks = {
        "status_frozen_before_rl": final_design["status"] == "frozen_before_rl",
        "ordinary_v2_hash_matches": final_sources["ordinary_episode_v2"][
            "manifest_sha256"
        ]
        == EPISODE_EVAL_MANIFEST_SHA256,
        "certified_v3_hash_matches": final_sources["certified_critical_decoy_v3"][
            "manifest_body_sha256"
        ]
        == manifests["frozen_ood"]["sha256"],
    }
    passed = (
        all(not values for values in overlaps.values())
        and all(
            report[check]
            for report in split_reports.values()
            for check in (
                "body_hash_valid",
                "all_critical_advantages_positive",
                "all_decoy_advantages_zero",
                "all_structural_states_matched",
                "all_sender_observations_matched",
                "only_receiver_target_knowledge_changes",
            )
        )
        and all(
            report["unique_seed_count"] == report["pair_count"]
            for report in split_reports.values()
        )
        and all(final_design_checks.values())
    )
    return {
        "version": "arena-rl-v3-global-audit-v1",
        "passed": passed,
        "seed_overlaps": overlaps,
        "final_design_checks": final_design_checks,
        "splits": split_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Swarm Arena RL v3 artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
