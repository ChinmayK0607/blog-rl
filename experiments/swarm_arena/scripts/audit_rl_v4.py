from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swarm_ctf_eval.episode_splits import EPISODE_EVAL_SEEDS
from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario


def _body_digest(manifest: dict) -> str:
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _existing_v3_seeds(data_dir: Path) -> set[int]:
    return {
        int(pair["critical"]["seed"])
        for split in ("train", "development", "frozen_ood")
        for pair in json.loads(
            (data_dir.parent / "rl_v3" / f"{split}.json").read_text(encoding="utf-8")
        )["pairs"]
    }


def audit(data_dir: Path) -> dict:
    handoff = {
        split: json.loads(
            (data_dir / f"handoff_{split}.json").read_text(encoding="utf-8")
        )
        for split in ("train", "development", "frozen_ood")
    }
    ordinary = {
        split: json.loads(
            (data_dir / f"ordinary_hard_{split}.json").read_text(encoding="utf-8")
        )
        for split in ("development", "frozen_ood")
    }
    handoff_seeds = {
        split: {int(pair["critical"]["seed"]) for pair in manifest["pairs"]}
        for split, manifest in handoff.items()
    }
    ordinary_seeds = {
        split: {int(case["seed"]) for case in manifest["cases"]}
        for split, manifest in ordinary.items()
    }
    existing = set(EPISODE_EVAL_SEEDS) | _existing_v3_seeds(data_dir)
    all_new_sets = {**{f"handoff_{key}": value for key, value in handoff_seeds.items()}, **{
        f"ordinary_{key}": value for key, value in ordinary_seeds.items()
    }}
    overlap_rows = {}
    names = sorted(all_new_sets)
    for index, left in enumerate(names):
        overlap_rows[f"{left}_existing"] = sorted(all_new_sets[left] & existing)
        for right in names[index + 1 :]:
            overlap_rows[f"{left}_{right}"] = sorted(
                all_new_sets[left] & all_new_sets[right]
            )

    handoff_reports = {}
    for split, manifest in handoff.items():
        reconstruction_valid = True
        for pair in manifest["pairs"]:
            for kind in ("critical", "decoy"):
                reconstruct_manifest_scenario(pair[kind])
        audits = [pair["matched_pair_audit"] for pair in manifest["pairs"]]
        handoff_reports[split] = {
            "body_hash_valid": _body_digest(manifest) == manifest["sha256"],
            "pair_count": manifest["pair_count"],
            "unique_seed_count": len(handoff_seeds[split]),
            "reconstruction_valid": reconstruction_valid,
            "all_critical_advantages_positive": all(
                float(pair["critical"]["minimum_advantage"]) > 0
                for pair in manifest["pairs"]
            ),
            "all_decoy_advantages_zero": all(
                abs(float(pair["decoy"]["minimum_advantage"])) <= 1e-12
                for pair in manifest["pairs"]
            ),
            "all_invariants_hold": all(
                bool(value) for row in audits for value in row.values()
            ),
        }

    ordinary_reports = {
        split: {
            "body_hash_valid": _body_digest(manifest) == manifest["sha256"],
            "case_count": manifest["case_count"],
            "unique_seed_count": len(ordinary_seeds[split]),
            "sizes_match": sorted({case["size"] for case in manifest["cases"]})
            == sorted(manifest["sizes"]),
            "horizons_match": sorted(
                {case["horizon"] for case in manifest["cases"]}
            )
            == sorted(manifest["horizons"]),
        }
        for split, manifest in ordinary.items()
    }
    design = json.loads(
        (data_dir / "progress_eval_design.json").read_text(encoding="utf-8")
    )
    index = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    design_valid = (
        design["status"] == "frozen_before_v4_rl"
        and design["frozen_bindings"] == index
    )
    passed = (
        not any(overlap_rows.values())
        and design_valid
        and all(
            report[check]
            for report in handoff_reports.values()
            for check in (
                "body_hash_valid",
                "reconstruction_valid",
                "all_critical_advantages_positive",
                "all_decoy_advantages_zero",
                "all_invariants_hold",
            )
        )
        and all(
            report["pair_count"] == report["unique_seed_count"]
            for report in handoff_reports.values()
        )
        and all(
            report[check]
            for report in ordinary_reports.values()
            for check in ("body_hash_valid", "sizes_match", "horizons_match")
        )
        and all(
            report["case_count"] == report["unique_seed_count"]
            for report in ordinary_reports.values()
        )
    )
    return {
        "version": "arena-rl-v4-global-audit-v1",
        "passed": passed,
        "seed_overlaps": overlap_rows,
        "design_binding_valid": design_valid,
        "handoff": handoff_reports,
        "ordinary": ordinary_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Swarm Arena RL v4 artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
