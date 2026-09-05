from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from swarm_ctf_eval.handoff_curriculum import generate_manifest
from swarm_ctf_eval.progress_eval_v4 import build_ordinary_manifest

HANDOFF_SPLITS = {
    "train": {
        "count": 240,
        "seed_start": 10_000_019,
        "sizes": (12, 13),
        "horizons": (4, 5),
    },
    "development": {
        "count": 48,
        "seed_start": 11_000_027,
        "sizes": (14, 16),
        "horizons": (6, 8),
    },
    "frozen_ood": {
        "count": 24,
        "seed_start": 12_000_041,
        "sizes": (18, 20),
        "horizons": (8, 10),
    },
}

ORDINARY_SPLITS = {
    "development": {
        "count": 24,
        "seed_start": 13_000_057,
        "sizes": (16, 18),
        "horizons": (8, 10),
    },
    "frozen_ood": {
        "count": 24,
        "seed_start": 14_000_069,
        "sizes": (18, 20),
        "horizons": (8, 10),
    },
}

CURRICULUM = {
    "version": "arena-rl-v4-handoff-curriculum-v1",
    "reward": "terminal_control_delta_only",
    "message_reward": None,
    "stages": [
        {
            "stage": 1,
            "mixture": {
                "ordinary_procedural": 0.5,
                "handoff_critical": 0.25,
                "handoff_decoy": 0.25,
            },
            "horizon": [4, 5],
            "purpose": "learn game actions while exposing every ordered sender/receiver role",
        },
        {
            "stage": 2,
            "mixture": {
                "ordinary_procedural": 0.7,
                "handoff_critical": 0.15,
                "handoff_decoy": 0.15,
            },
            "horizon": [6, 8],
            "purpose": "preserve general play while retaining causal handoff pressure",
        },
    ],
    "promotion": (
        "development capability must not regress; critical normal-minus-dropped must be "
        "positive; matched-decoy normal-minus-dropped must remain null"
    ),
}


def _handoff_audit(manifest: dict) -> dict:
    critical = [pair["critical"] for pair in manifest["pairs"]]
    decoy = [pair["decoy"] for pair in manifest["pairs"]]
    roles = Counter((row["sender"], row["receiver"]) for row in critical)
    advantages = [float(row["minimum_advantage"]) for row in critical]
    invariant_names = sorted(manifest["pairs"][0]["matched_pair_audit"])
    return {
        "manifest_sha256": manifest["sha256"],
        "pair_count": manifest["pair_count"],
        "critical_advantage": {
            "minimum": min(advantages),
            "mean": statistics.mean(advantages),
            "maximum": max(advantages),
        },
        "decoy_zero_advantage_rate": statistics.mean(
            abs(float(row["minimum_advantage"])) <= 1e-12 for row in decoy
        ),
        "ordered_role_pairs": {
            f"{sender}->{receiver}": count
            for (sender, receiver), count in sorted(roles.items())
        },
        "role_balance_range": [min(roles.values()), max(roles.values())],
        "invariants": {
            name: statistics.mean(
                bool(pair["matched_pair_audit"][name]) for pair in manifest["pairs"]
            )
            for name in invariant_names
        },
        "certificate_scope": (
            "exact best receiver policy over two latent worlds against each frozen opponent style"
        ),
    }


def _progress_design(index: dict) -> dict:
    return {
        "version": "arena-rl-progress-eval-v4",
        "status": "frozen_before_v4_rl",
        "independent_units": {
            "ordinary": "game seed",
            "handoff": "paired two-world latent bundle",
        },
        "opponent_pool": ["base", "sft", "historical_league"],
        "conditions": ["normal", "dropped", "sender_shuffled", "delayed", "zero_budget"],
        "tiers": {
            "online_monitor": {
                "source": "development only",
                "ordinary_legacy_cases": 4,
                "ordinary_hard_cases": 4,
                "handoff_pairs": 4,
                "opponents": ["sft"],
                "conditions": ["normal", "dropped"],
                "purpose": "cheap directional signal; never a research claim",
            },
            "checkpoint_selection": {
                "source": "development only",
                "ordinary_legacy_cases": 12,
                "ordinary_hard_cases": 12,
                "handoff_pairs": 12,
                "opponents": ["base", "sft", "historical_league"],
                "conditions": ["normal", "dropped", "sender_shuffled", "delayed", "zero_budget"],
                "purpose": "select at most one checkpoint before frozen evaluation",
            },
            "frozen_final": {
                "ordinary_legacy_cases": 72,
                "ordinary_legacy_independent_seeds": 24,
                "ordinary_legacy_option_orders": [
                    "canonical",
                    "permuted-1",
                    "permuted-2",
                ],
                "ordinary_hard_cases": index["ordinary"]["frozen_ood"]["case_count"],
                "handoff_pairs": index["handoff"]["frozen_ood"]["pair_count"],
                "opponents": ["base", "sft", "historical_league"],
                "sides": ["BLUE", "RED"],
                "critical_conditions": [
                    "normal",
                    "dropped",
                    "sender_shuffled",
                    "delayed",
                    "zero_budget",
                ],
                "decoy_conditions": ["normal", "dropped"],
                "run_policy": "once for the development-selected candidate",
            },
        },
        "primary_endpoints": {
            "legacy_capability": "candidate RL minus SFT paired terminal return",
            "hard_capability": "candidate RL minus SFT paired terminal return",
            "communication": (
                "critical normal minus each message intervention, averaged inside each latent bundle"
            ),
            "decoy": "matched-decoy normal minus dropped must include zero",
        },
        "claim_gates": {
            "capability": "both legacy and hard paired 95% intervals are positive",
            "communication": (
                "all four critical intervention intervals are positive, effects are positive "
                "against every opponent, and the matched-decoy interval includes zero"
            ),
            "regression": "both existing frozen non-arena regression suites pass",
            "collapse": (
                "no always/never-speaking, repeated-target, action, excessive-KL, or single-opponent collapse"
            ),
        },
        "frozen_bindings": index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Swarm Arena RL v4 CPU manifests.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"handoff": {}, "ordinary": {}}
    for split, config in HANDOFF_SPLITS.items():
        manifest = generate_manifest(**config)
        audit = _handoff_audit(manifest)
        manifest_path = args.output_dir / f"handoff_{split}.json"
        audit_path = args.output_dir / f"handoff_{split}_audit.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        index["handoff"][split] = {
            "manifest": manifest_path.name,
            "audit": audit_path.name,
            "sha256": manifest["sha256"],
            "pair_count": manifest["pair_count"],
            "sizes": manifest["sizes"],
            "horizons": manifest["horizons"],
        }
    for split, config in ORDINARY_SPLITS.items():
        manifest = build_ordinary_manifest(**config)
        manifest_path = args.output_dir / f"ordinary_hard_{split}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        index["ordinary"][split] = {
            "manifest": manifest_path.name,
            "sha256": manifest["sha256"],
            "case_count": manifest["case_count"],
            "sizes": manifest["sizes"],
            "horizons": manifest["horizons"],
        }
    (args.output_dir / "curriculum.json").write_text(
        json.dumps(CURRICULUM, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "progress_eval_design.json").write_text(
        json.dumps(_progress_design(index), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
