from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scripts.run_final_eval_development import _prepare_output, _roster, _served_model
from swarm_ctf_eval.arena_eval import ArenaModel
from swarm_ctf_eval.crossplay_eval import FROZEN_CROSSPLAY_CASES, development_cases
from swarm_ctf_eval.final_eval_runner import FinalEvalIdentity, evaluate_final_case
from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.progress_eval_v4 import (
    COMMUNICATION_CONDITIONS,
    PROGRESS_EVAL_VERSION,
    summarize_progress_eval,
)
from swarm_ctf_eval.progress_eval_v5 import summarize_rl_specific_progress_eval

Tier = Literal["pulse", "online", "selection", "frozen"]


@dataclass(frozen=True)
class TierPlan:
    tier: Tier
    legacy_cases: int
    hard_cases: int
    handoff_pairs: int
    opponent_scope: str
    legacy_option_orders: tuple[str, ...]
    critical_conditions: tuple[str, ...]
    sides: tuple[str, ...]


TIER_PLANS = {
    "pulse": TierPlan(
        "pulse",
        1,
        1,
        1,
        "monitor",
        ("canonical",),
        ("normal", "dropped"),
        ("BLUE",),
    ),
    "online": TierPlan(
        "online",
        4,
        4,
        4,
        "monitor",
        ("canonical",),
        ("normal", "dropped"),
        ("BLUE", "RED"),
    ),
    "selection": TierPlan(
        "selection",
        12,
        12,
        12,
        "all",
        ("canonical",),
        COMMUNICATION_CONDITIONS,
        ("BLUE", "RED"),
    ),
    "frozen": TierPlan(
        "frozen",
        len(FROZEN_CROSSPLAY_CASES),
        24,
        24,
        "all",
        ("canonical", "permuted-1", "permuted-2"),
        COMMUNICATION_CONDITIONS,
        ("BLUE", "RED"),
    ),
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if _digest(body) != manifest.get("sha256"):
        raise ValueError(f"manifest body hash mismatch: {path}")
    return manifest


def _validate_frozen_confirmation(
    tier: Tier,
    design: dict[str, Any],
    confirmation: str | None,
) -> None:
    if tier != "frozen":
        if confirmation is not None:
            raise ValueError("frozen confirmation is valid only for the frozen tier")
        return
    expected = _digest(design)
    if confirmation != expected:
        raise ValueError(
            "frozen evaluation requires --frozen-confirmation " + expected
        )


def _ordinary_cases(
    tier: Tier,
    hard_manifest: dict[str, Any],
) -> tuple[tuple[str, str, tuple[int, int, int], str, str], ...]:
    plan = TIER_PLANS[tier]
    legacy = (
        FROZEN_CROSSPLAY_CASES
        if tier == "frozen"
        else development_cases(plan.legacy_cases)
    )
    rows = []
    for seed, size, horizon in legacy[: plan.legacy_cases]:
        for option_order in plan.legacy_option_orders:
            rows.append(
                (
                    f"legacy-{seed}-{option_order}",
                    f"legacy-seed-{seed}",
                    (seed, size, horizon),
                    "ordinary_legacy",
                    option_order,
                )
            )
    for case in hard_manifest["cases"][: plan.hard_cases]:
        seed = int(case["seed"])
        rows.append(
            (
                str(case["case_id"]),
                f"hard-seed-{seed}",
                (seed, int(case["size"]), int(case["horizon"])),
                "ordinary_hard",
                "canonical",
            )
        )
    return tuple(rows)


def _handoff_worlds(
    tier: Tier,
    handoff_manifest: dict[str, Any],
) -> tuple[tuple[str, str, str, Any, Any], ...]:
    rows = []
    for pair_index, pair in enumerate(
        handoff_manifest["pairs"][: TIER_PLANS[tier].handoff_pairs]
    ):
        independent_id = f"handoff-bundle-{pair_index:03d}"
        for kind in ("critical", "decoy"):
            scenario = reconstruct_manifest_scenario(pair[kind])
            for world in scenario.worlds:
                rows.append(
                    (
                        f"{independent_id}-{kind}-{world.label}",
                        independent_id,
                        f"handoff_{kind}",
                        scenario,
                        world,
                    )
                )
    return tuple(rows)


def _opponents(
    config: dict[str, Any],
    plan: TierPlan,
    base_urls: tuple[str, ...],
    api_key: str,
    monitor_opponent_id: str,
) -> dict[str, tuple[str, tuple[ArenaModel, ...]]]:
    configured = {
        str(item["id"]): (
            str(item["revision"]),
            _distributed_roster(base_urls, list(item["models"]), api_key),
        )
        for item in config["opponents"]
    }
    if len(configured) < 3:
        raise ValueError("v4 selection/frozen evaluation requires three opponent families")
    if plan.opponent_scope == "monitor":
        if monitor_opponent_id not in configured:
            raise ValueError("monitor opponent is absent from the configured opponent pool")
        return {monitor_opponent_id: configured[monitor_opponent_id]}
    return configured


def _distributed_roster(
    base_urls: tuple[str, ...], models: list[str], api_key: str
) -> tuple[ArenaModel, ...]:
    if not base_urls:
        raise ValueError("evaluation requires at least one serving URL")
    if len(base_urls) == 1:
        return _roster(base_urls[0], models, api_key)
    if len(models) != 4:
        raise ValueError("every evaluation roster must contain exactly four model IDs")
    return tuple(
        _served_model(base_urls[index % len(base_urls)], model, api_key)
        for index, model in enumerate(models)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable, tiered Swarm Arena RL progress evaluation v4."
    )
    parser.add_argument("--tier", choices=tuple(TIER_PLANS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monitor-opponent-id", default="sft")
    parser.add_argument("--frozen-confirmation")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rl-specific-communication",
        action="store_true",
        help=(
            "also evaluate the SFT initializer on critical normal/dropped cases and "
            "report RL-minus-SFT communication lift"
        ),
    )
    args = parser.parse_args()

    tier: Tier = args.tier
    plan = TIER_PLANS[tier]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    design = json.loads(
        (args.data_dir / "progress_eval_design.json").read_text(encoding="utf-8")
    )
    _validate_frozen_confirmation(tier, design, args.frozen_confirmation)
    split = "frozen_ood" if tier == "frozen" else "development"
    hard_manifest = _load_manifest(
        args.data_dir / f"ordinary_hard_{split}.json"
    )
    handoff_manifest = _load_manifest(
        args.data_dir / f"handoff_{split}.json"
    )

    configured_urls = config.get("base_urls", [config.get("base_url")])
    if not isinstance(configured_urls, list) or any(
        not isinstance(value, str) or not value for value in configured_urls
    ):
        raise ValueError("evaluation base_urls must be a non-empty string list")
    base_urls = tuple(configured_urls)
    candidate = _distributed_roster(
        base_urls, list(config["candidate"]["models"]), args.api_key
    )
    baseline = _distributed_roster(
        base_urls, list(config["baseline"]["models"]), args.api_key
    )
    opponents = _opponents(
        config,
        plan,
        base_urls,
        args.api_key,
        args.monitor_opponent_id,
    )
    ordinary = _ordinary_cases(tier, hard_manifest)
    handoffs = _handoff_worlds(tier, handoff_manifest)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "version": PROGRESS_EVAL_VERSION,
        "tier": tier,
        "source_commit": source_commit,
        "config": config,
        "config_sha256": _digest(config),
        "design_sha256": _digest(design),
        "hard_manifest_sha256": hard_manifest["sha256"],
        "handoff_manifest_sha256": handoff_manifest["sha256"],
        "opponents": {
            opponent_id: revision
            for opponent_id, (revision, _) in opponents.items()
        },
        "ordinary_case_ids": [row[0] for row in ordinary],
        "handoff_case_ids": [row[0] for row in handoffs],
        "critical_conditions": list(plan.critical_conditions),
        "decoy_conditions": ["normal", "dropped"],
        "sides": list(plan.sides),
        "generation": {"temperature": 0.0, "max_tokens": 160, "structured": True},
        "rl_specific_communication": args.rl_specific_communication,
    }
    completed = _prepare_output(args.output_dir, manifest, args.resume)
    rows_path = args.output_dir / "rows.jsonl"
    raw_path = args.output_dir / "raw.jsonl"

    def run_one(
        *,
        case_id: str,
        independent_id: str,
        suite: str,
        case: tuple[int, int, int],
        variant: str,
        revision: str,
        focal: tuple[ArenaModel, ...],
        opponent_id: str,
        opponent_revision: str,
        opponent: tuple[ArenaModel, ...],
        side: str,
        condition: str,
        option_order: str = "canonical",
        initial_state: Any | None = None,
        critical_target: str | None = None,
    ) -> None:
        evaluation_id = ":".join(
            (
                case_id,
                variant,
                revision,
                opponent_id,
                opponent_revision,
                side,
                condition,
            )
        )
        if evaluation_id in completed:
            return
        sampling_key = ":".join((case_id, opponent_id, side))
        identity = FinalEvalIdentity(
            case_id,
            suite,
            variant,
            revision,
            "identity",
            "identity",
            option_order,
            opponent_id,
            opponent_revision,
            sampling_key,
        )
        row, raw = evaluate_final_case(
            focal,
            opponent,
            case,
            identity,
            focal_side=side,  # type: ignore[arg-type]
            condition=condition,
            initial_state=initial_state,
            critical_target=critical_target,
        )
        row = {
            **row,
            "independent_id": independent_id,
            "evaluation_id": evaluation_id,
        }
        raw_record = {"evaluation_id": evaluation_id, "raw": raw}
        row["raw_sha256"] = _digest(raw_record)
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(raw_record, sort_keys=True) + "\n")
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        completed.add(evaluation_id)
        print(json.dumps({"completed": evaluation_id, "return": row["terminal_return"]}))

    variants = (
        (
            "candidate_rl",
            str(config["candidate"]["revision"]),
            candidate,
        ),
        ("sft_init", str(config["baseline"]["revision"]), baseline),
    )
    for case_id, independent_id, case, suite, option_order in ordinary:
        for opponent_id, (opponent_revision, opponent) in opponents.items():
            for variant, revision, focal in variants:
                for side in plan.sides:
                    run_one(
                        case_id=case_id,
                        independent_id=independent_id,
                        suite=suite,
                        case=case,
                        variant=variant,
                        revision=revision,
                        focal=focal,
                        opponent_id=opponent_id,
                        opponent_revision=opponent_revision,
                        opponent=opponent,
                        side=side,
                        condition="normal",
                        option_order=option_order,
                    )

    candidate_revision = str(config["candidate"]["revision"])
    for case_id, independent_id, suite, scenario, world in handoffs:
        conditions = (
            plan.critical_conditions
            if suite == "handoff_critical"
            else ("normal", "dropped")
        )
        for opponent_id, (opponent_revision, opponent) in opponents.items():
            for side in plan.sides:
                for condition in conditions:
                    run_one(
                        case_id=case_id,
                        independent_id=independent_id,
                        suite=suite,
                        case=(scenario.seed, scenario.size, scenario.horizon),
                        variant="candidate_rl",
                        revision=candidate_revision,
                        focal=candidate,
                        opponent_id=opponent_id,
                        opponent_revision=opponent_revision,
                        opponent=opponent,
                        side=side,
                        condition=condition,
                        initial_state=world.state,
                        critical_target=(
                            world.active_target
                            if suite == "handoff_critical"
                            else None
                        ),
                    )
                if args.rl_specific_communication and suite == "handoff_critical":
                    baseline_revision = str(config["baseline"]["revision"])
                    for condition in ("normal", "dropped"):
                        run_one(
                            case_id=case_id,
                            independent_id=independent_id,
                            suite=suite,
                            case=(scenario.seed, scenario.size, scenario.horizon),
                            variant="sft_init",
                            revision=baseline_revision,
                            focal=baseline,
                            opponent_id=opponent_id,
                            opponent_revision=opponent_revision,
                            opponent=opponent,
                            side=side,
                            condition=condition,
                            initial_state=world.state,
                            critical_target=world.active_target,
                        )

    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summarize = (
        summarize_rl_specific_progress_eval
        if args.rl_specific_communication
        else summarize_progress_eval
    )
    summary = summarize(
        rows,
        intervention_conditions=tuple(
            condition
            for condition in plan.critical_conditions
            if condition != "normal"
        ),
    )
    summary["tier"] = tier
    summary["scope"] = (
        "frozen final; run once"
        if tier == "frozen"
        else f"{tier} development evaluation; not a final research claim"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
