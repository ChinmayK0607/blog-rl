from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from swarm_ctf_eval.async_admission import AsyncAdmissionLimits
from swarm_ctf_eval.rl_production import (
    STAGED_RL_PRODUCTION_PLAN_VERSION,
    load_production_plan,
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _production_stage(stage: dict[str, object]) -> dict[str, object]:
    result = {
        key: stage[key]
        for key in (
            "name",
            "updates",
            "update_pattern",
            "ordinary_sizes",
            "ordinary_horizons",
            "handoff_focus_roles",
            "handoff_remaining_turns",
            "handoff_trainable_turn_offsets",
        )
        if key in stage
    }
    if "handoff_cases" in stage:
        result["handoff_cases"] = [
            (
                value
                if isinstance(value, dict)
                else {"pair_index": int(value[0]), "world": str(value[1])}
            )
            for value in stage["handoff_cases"]
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind the checked-in staged curriculum to an immutable opponent/runtime plan."
    )
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    parser.add_argument("--runtime-certificate", type=Path, required=True)
    parser.add_argument("--admission-limits", type=Path, required=True)
    parser.add_argument(
        "--trainable-phase",
        action="append",
        choices=("BROADCAST", "ACT"),
        default=[],
        help="override the base plan's trainable phases while retaining its turn schedule",
    )
    parser.add_argument(
        "--trainable-turn-offset",
        action="append",
        type=int,
        default=[],
        help="override the base plan's trainable turn offsets",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base_plan.read_text(encoding="utf-8"))
    curriculum = json.loads(args.curriculum.read_text(encoding="utf-8"))
    handoff = json.loads(args.handoff_manifest.read_text(encoding="utf-8"))
    certificate = json.loads(args.runtime_certificate.read_text(encoding="utf-8"))
    admission_limits = json.loads(args.admission_limits.read_text(encoding="utf-8"))
    parsed_admission_limits = AsyncAdmissionLimits(**admission_limits)
    parsed_admission_limits.validate()
    certificate_body = {key: value for key, value in certificate.items() if key != "sha256"}
    if certificate.get("sha256") != _digest(certificate_body):
        raise ValueError("runtime certificate body hash mismatch")
    if certificate.get("version") != "swarm-runtime-certificate-v1" or certificate.get("status") != "passed":
        raise ValueError("runtime certificate is not a passed v1 certificate")
    handoff_body = {key: value for key, value in handoff.items() if key != "sha256"}
    if handoff.get("sha256") != _digest(handoff_body):
        raise ValueError("handoff training manifest body hash mismatch")
    if curriculum["groups_per_update"] != 4:
        raise ValueError("the staged curriculum requires exactly four groups per update")
    if base["groups_per_update"] != curriculum["groups_per_update"]:
        raise ValueError("base plan and curriculum disagree on groups per update")
    plan = {
        **base,
        "version": STAGED_RL_PRODUCTION_PLAN_VERSION,
        "curriculum_stages": [_production_stage(stage) for stage in curriculum["stages"]],
        "curriculum_source": {
            "path": str(args.curriculum),
            "sha256": _digest(curriculum),
        },
        "rollout_runtime": curriculum.get(
            "runtime",
            {
                "shared_return_replicas": 4,
                "action_prompt_profile": "full",
            },
        ),
        "training_data": {
            "handoff_manifest": str(args.handoff_manifest),
            "sha256": handoff["sha256"],
            "pair_count": handoff["pair_count"],
        },
        "async_admission": {
            **base["async_admission"],
            "limits": admission_limits,
            "backend": {
                "name": certificate["backend"]["name"],
                "version": certificate["backend"]["version"],
                "kernel_config_sha256": certificate["inference_config_sha256"],
                "calibration_sha256": certificate["sha256"],
            },
        },
        "runtime_certificate": {
            "path": str(args.runtime_certificate),
            "sha256": certificate["sha256"],
        },
        **(
            {"adaptive_curriculum": curriculum["adaptive_curriculum"]}
            if curriculum.get("adaptive_curriculum") is not None
            else {}
        ),
    }
    if args.trainable_phase or args.trainable_turn_offset:
        if len(set(args.trainable_phase)) != len(args.trainable_phase):
            raise ValueError("trainable phase overrides must be unique")
        if len(set(args.trainable_turn_offset)) != len(args.trainable_turn_offset):
            raise ValueError("trainable turn-offset overrides must be unique")
        if any(value < 0 for value in args.trainable_turn_offset):
            raise ValueError("trainable turn-offset overrides cannot be negative")
        plan["trainable_spans"] = {
            **base["trainable_spans"],
            **({"phases": args.trainable_phase} if args.trainable_phase else {}),
            **(
                {"turn_offsets": args.trainable_turn_offset}
                if args.trainable_turn_offset
                else {}
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded, _ = load_production_plan(args.output)
    schedule = loaded.curriculum_schedule(steps=int(curriculum["total_updates"]))
    maximum_pair_index = max(row.pair_index for row in schedule if row.pair_index is not None)
    if maximum_pair_index >= int(handoff["pair_count"]):
        raise ValueError(
            f"staged schedule requires pair {maximum_pair_index}, but manifest contains "
            f"only {handoff['pair_count']} pairs"
        )
    counts = Counter(row.kind for row in schedule)
    per_stage = {
        stage.name: Counter(row.kind for row in schedule if row.stage == stage.name) for stage in loaded.stages
    }
    audit = {
        "plan_sha256": loaded.sha256,
        "curriculum_sha256": _digest(curriculum),
        "updates": loaded.expected_updates,
        "groups": len(schedule),
        "counts": dict(counts),
        "counts_by_stage": {name: dict(rows) for name, rows in per_stage.items()},
        "maximum_pair_index": maximum_pair_index,
        "handoff_manifest_sha256": handoff["sha256"],
        "runtime_certificate_sha256": certificate["sha256"],
        "async_admission_limits_sha256": parsed_admission_limits.sha256,
        "trainable_phases": list(loaded.trainable_phases),
        "adaptive_curriculum": (
            None
            if loaded.adaptive_curriculum is None
            else loaded.adaptive_curriculum.__dict__
        ),
        "unique_ordinary_seeds": len({row.ordinary_seed for row in schedule if row.ordinary_seed is not None}),
        "schedule_sha256": _digest(
            [
                {
                    "ordinal": row.ordinal,
                    "kind": row.kind,
                    "pair_index": row.pair_index,
                    "ordinary_seed": row.ordinary_seed,
                    "stage": row.stage,
                    "ordinary_size": row.ordinary_size,
                    "ordinary_horizon": row.ordinary_horizon,
                    "handoff_focus_role": row.handoff_focus_role,
                    "handoff_world": row.handoff_world,
                    "handoff_remaining_turns": row.handoff_remaining_turns,
                    **(
                        {
                            "handoff_trainable_turn_offsets": row.handoff_trainable_turn_offsets
                        }
                        if row.handoff_trainable_turn_offsets is not None
                        else {}
                    ),
                }
                for row in schedule
            ]
        ),
        "handoff_focus_counts": dict(
            Counter(
                row.handoff_focus_role
                for row in schedule
                if row.handoff_focus_role is not None
            )
        ),
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
