from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from swarm_ctf_eval.semantic_holdout import summarize_semantic_holdout

from scripts.run_final_eval_development import _prepare_output, _roster, _served_model

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
        6,
        6,
        6,
        "monitor",
        ("canonical",),
        ("normal", "dropped"),
        ("BLUE", "RED"),
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
        (*COMMUNICATION_CONDITIONS, "target_swapped"),
        ("BLUE", "RED"),
    ),
    "frozen": TierPlan(
        "frozen",
        len(FROZEN_CROSSPLAY_CASES),
        24,
        24,
        "all",
        ("canonical", "permuted-1", "permuted-2"),
        (*COMMUNICATION_CONDITIONS, "target_swapped"),
        ("BLUE", "RED"),
    ),
}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_cached_baseline(
    *,
    baseline_rows_path: Path,
    rows_path: Path,
    raw_path: Path,
    completed: set[str],
    baseline_revision: str,
    ordinary_case_ids: set[str],
    critical_case_ids: set[str],
    opponent_ids: set[str],
    sides: set[str],
    expected_rows: int,
) -> int:
    """Copy immutable SFT rows/raw records into a later checkpoint artifact."""
    source_raw_path = baseline_rows_path.with_name("raw.jsonl")
    if not source_raw_path.is_file():
        raise FileNotFoundError(f"cached baseline raw rows are missing: {source_raw_path}")
    source_raw = {
        str(record["evaluation_id"]): record
        for line in source_raw_path.read_text(encoding="utf-8").splitlines()
        if line
        for record in (json.loads(line),)
    }
    selected = []
    for line in baseline_rows_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("policy_variant") != "sft_init":
            continue
        suite = str(row.get("suite"))
        case_id = str(row.get("case_id"))
        condition = str(row.get("condition"))
        allowed = (
            suite in {"ordinary_legacy", "ordinary_hard"} and case_id in ordinary_case_ids and condition == "normal"
        ) or (suite == "handoff_critical" and case_id in critical_case_ids and condition in {"normal", "dropped"})
        if not allowed:
            continue
        if row.get("policy_revision") != baseline_revision:
            raise ValueError("cached baseline revision does not match the run config")
        if row.get("opponent_id") not in opponent_ids or row.get("side") not in sides:
            continue
        selected.append(row)
    if len(selected) != expected_rows:
        raise ValueError(f"cached baseline has {len(selected)} required rows; expected {expected_rows}")
    for row in selected:
        evaluation_id = str(row["evaluation_id"])
        raw_record = source_raw.get(evaluation_id)
        if raw_record is None or _digest(raw_record) != row.get("raw_sha256"):
            raise ValueError(f"cached baseline raw record mismatch: {evaluation_id}")
        if evaluation_id in completed:
            continue
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(raw_record, sort_keys=True) + "\n")
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        completed.add(evaluation_id)
    return len(selected)


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
        raise ValueError("frozen evaluation requires --frozen-confirmation " + expected)


def _ordinary_cases(
    tier: Tier,
    hard_manifest: dict[str, Any],
) -> tuple[tuple[str, str, tuple[int, int, int], str, str], ...]:
    plan = TIER_PLANS[tier]
    legacy = FROZEN_CROSSPLAY_CASES if tier == "frozen" else development_cases(plan.legacy_cases)
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
    *,
    pair_count: int | None = None,
) -> tuple[tuple[str, str, str, Any, Any], ...]:
    rows = []
    limit = TIER_PLANS[tier].handoff_pairs if pair_count is None else pair_count
    if limit > int(handoff_manifest["pair_count"]):
        raise ValueError(f"{tier} requests {limit} handoff pairs from a {handoff_manifest['pair_count']}-pair manifest")
    source_start = int(handoff_manifest.get("source_pair_start", 0))
    for pair_index, pair in enumerate(handoff_manifest["pairs"][:limit]):
        source_pair_index = source_start + pair_index
        independent_id = f"handoff-bundle-{source_pair_index:03d}"
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


def _distributed_roster(base_urls: tuple[str, ...], models: list[str], api_key: str) -> tuple[ArenaModel, ...]:
    if not base_urls:
        raise ValueError("evaluation requires at least one serving URL")
    if len(base_urls) == 1:
        return _roster(base_urls[0], models, api_key)
    if len(models) != 4:
        raise ValueError("every evaluation roster must contain exactly four model IDs")
    return tuple(_served_model(base_urls[index % len(base_urls)], model, api_key) for index, model in enumerate(models))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resumable, tiered Swarm Arena RL progress evaluation v4.")
    parser.add_argument("--tier", choices=tuple(TIER_PLANS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monitor-opponent-id", default="sft")
    parser.add_argument("--frozen-confirmation")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        help="reuse immutable sft_init rows and matching raw records from update 0",
    )
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
    design = json.loads((args.data_dir / "progress_eval_design.json").read_text(encoding="utf-8"))
    _validate_frozen_confirmation(tier, design, args.frozen_confirmation)
    split = "frozen_ood" if tier == "frozen" else "development"
    hard_manifest = _load_manifest(args.data_dir / f"ordinary_hard_{split}.json")
    handoff_manifest = _load_manifest(args.data_dir / f"handoff_{split}.json")

    configured_urls = config.get("base_urls", [config.get("base_url")])
    if not isinstance(configured_urls, list) or any(
        not isinstance(value, str) or not value for value in configured_urls
    ):
        raise ValueError("evaluation base_urls must be a non-empty string list")
    base_urls = tuple(configured_urls)
    candidate = _distributed_roster(base_urls, list(config["candidate"]["models"]), args.api_key)
    baseline = _distributed_roster(base_urls, list(config["baseline"]["models"]), args.api_key)
    opponents = _opponents(
        config,
        plan,
        base_urls,
        args.api_key,
        args.monitor_opponent_id,
    )
    ordinary = _ordinary_cases(tier, hard_manifest)
    design_tier = design["frozen_final" if tier == "frozen" else "development_selection"]
    designed_handoff_pairs = (
        int(design_tier["handoff_pairs"])
        if tier == "frozen"
        else len(design_tier["handoff_pair_indices"])
        if tier == "selection"
        else None
    )
    handoffs = _handoff_worlds(
        tier,
        handoff_manifest,
        pair_count=designed_handoff_pairs,
    )
    decoy_conditions = (
        ("normal", "dropped", "target_swapped")
        if "target_swapped" in plan.critical_conditions
        else ("normal", "dropped")
    )
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "version": PROGRESS_EVAL_VERSION,
        "tier": tier,
        "source_commit": source_commit,
        "config": config,
        "config_sha256": _digest(config),
        "design_sha256": _digest(design),
        "hard_manifest_sha256": hard_manifest["sha256"],
        "handoff_manifest_sha256": handoff_manifest["sha256"],
        "opponents": {opponent_id: revision for opponent_id, (revision, _) in opponents.items()},
        "ordinary_case_ids": [row[0] for row in ordinary],
        "handoff_case_ids": [row[0] for row in handoffs],
        "critical_conditions": list(plan.critical_conditions),
        "decoy_conditions": list(decoy_conditions),
        "sides": list(plan.sides),
        "generation": {"temperature": 0.0, "max_tokens": 160, "structured": True},
        "rl_specific_communication": args.rl_specific_communication,
        "baseline_rows_sha256": (_sha256_file(args.baseline_rows) if args.baseline_rows else None),
        "baseline_raw_sha256": (
            _sha256_file(args.baseline_rows.with_name("raw.jsonl")) if args.baseline_rows else None
        ),
    }
    completed = _prepare_output(args.output_dir, manifest, args.resume)
    rows_path = args.output_dir / "rows.jsonl"
    raw_path = args.output_dir / "raw.jsonl"
    if args.baseline_rows is not None:
        critical_case_ids = {row[0] for row in handoffs if row[2] == "handoff_critical"}
        expected_baseline_rows = (
            len(ordinary) * len(opponents) * len(plan.sides)
            + len(critical_case_ids) * len(opponents) * len(plan.sides) * 2
        )
        _import_cached_baseline(
            baseline_rows_path=args.baseline_rows,
            rows_path=rows_path,
            raw_path=raw_path,
            completed=completed,
            baseline_revision=str(config["baseline"]["revision"]),
            ordinary_case_ids={row[0] for row in ordinary},
            critical_case_ids=critical_case_ids,
            opponent_ids=set(opponents),
            sides=set(plan.sides),
            expected_rows=expected_baseline_rows,
        )

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
        handoff_sender: str | None = None,
        handoff_receiver: str | None = None,
        candidate_targets: tuple[str, str] | None = None,
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
        semantic = condition == "target_swapped"
        row, raw = evaluate_final_case(
            focal,
            opponent,
            case,
            identity,
            focal_side=side,  # type: ignore[arg-type]
            condition=condition,
            initial_state=initial_state,
            critical_target=(critical_target if suite == "handoff_critical" else None),
            target_swap_sender=handoff_sender if semantic else None,
            target_swap_targets=candidate_targets if semantic else None,
            target_swap_active_target=critical_target if semantic else None,
            target_swap_receiver=handoff_receiver if semantic else None,
        )
        row = {
            **row,
            "independent_id": independent_id,
            "evaluation_id": evaluation_id,
        }
        if (
            handoff_sender is not None
            and handoff_receiver is not None
            and critical_target is not None
        ):
            prefix = side.lower()
            sender = f"{prefix}-{handoff_sender.split('-', 1)[1]}"
            receiver = f"{prefix}-{handoff_receiver.split('-', 1)[1]}"
            sender_broadcasts = [
                broadcast
                for broadcast in raw["turns"][0]["broadcasts"]
                if broadcast["agent_id"] == sender
            ]
            if len(sender_broadcasts) != 1:
                raise ValueError(f"missing unique handoff sender broadcast: {sender}")
            accepted = sender_broadcasts[0]["accepted_message"]
            row["sender_target_fact"] = any(
                fact["node"] == critical_target for fact in accepted["facts"]
            )
            row["sender_nonempty"] = accepted != {
                "facts": [],
                "intent": None,
                "request_resource": 0,
            }
            receiver_action = next(
                action
                for action in raw["turns"][0]["actions"]
                if action["agent_id"] == receiver
            )
            row["receiver_target_action"] = (
                receiver_action["selected_action"].get("target") == critical_target
            )
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
        conditions = plan.critical_conditions if suite == "handoff_critical" else decoy_conditions
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
                        critical_target=world.active_target,
                        handoff_sender=scenario.sender,
                        handoff_receiver=scenario.receiver,
                        candidate_targets=scenario.candidate_targets,
                    )
                if args.rl_specific_communication and suite == "handoff_critical":
                    baseline_revision = str(config["baseline"]["revision"])
                    baseline_conditions = (
                        ("normal", "dropped", "target_swapped")
                        if "target_swapped" in plan.critical_conditions
                        else ("normal", "dropped")
                    )
                    for condition in baseline_conditions:
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
                            handoff_sender=scenario.sender,
                            handoff_receiver=scenario.receiver,
                            candidate_targets=scenario.candidate_targets,
                        )

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    summarize = summarize_rl_specific_progress_eval if args.rl_specific_communication else summarize_progress_eval
    standard_rows = [row for row in rows if row["condition"] != "target_swapped"]
    summary = summarize(
        standard_rows,
        intervention_conditions=tuple(
            condition
            for condition in plan.critical_conditions
            if condition not in {"normal", "target_swapped"}
        ),
    )
    if "target_swapped" in plan.critical_conditions:
        if not args.rl_specific_communication:
            raise ValueError("semantic selection/frozen evaluation requires --rl-specific-communication")
        summary["semantic"] = summarize_semantic_holdout(rows)
    summary["tier"] = tier
    summary["scope"] = (
        "frozen final; run once" if tier == "frozen" else f"{tier} development evaluation; not a final research claim"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
