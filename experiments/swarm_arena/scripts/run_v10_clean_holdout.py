from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scripts.run_final_eval_development import _prepare_output, _roster, _served_model
from scripts.run_progress_eval_v4 import _load_manifest
from swarm_ctf_eval.arena_eval import ArenaModel
from swarm_ctf_eval.crossplay_eval import FROZEN_CROSSPLAY_CASES
from swarm_ctf_eval.final_eval_runner import FinalEvalIdentity, evaluate_final_case
from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.progress_eval_v4 import COMMUNICATION_CONDITIONS
from swarm_ctf_eval.progress_eval_v5 import summarize_rl_specific_progress_eval
from swarm_ctf_eval.semantic_holdout import summarize_semantic_holdout

VERSION = "arena-rl-v10-clean-holdout-runner-v1"
CANDIDATE_CRITICAL_CONDITIONS = (*COMMUNICATION_CONDITIONS, "target_swapped")
BASELINE_CRITICAL_CONDITIONS = ("normal", "dropped", "target_swapped")
DECOY_CONDITIONS = ("normal", "dropped", "target_swapped")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl_durable(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in lock.items() if key != "sha256"}
    if _digest(body) != lock.get("sha256"):
        raise ValueError("clean holdout lock body hash mismatch")
    return lock


def _distributed_roster(base_urls: tuple[str, ...], models: list[str], api_key: str) -> tuple[ArenaModel, ...]:
    if not base_urls:
        raise ValueError("evaluation requires at least one serving URL")
    if len(models) != 4:
        raise ValueError("every evaluation roster must contain four model IDs")
    if len(base_urls) == 1:
        return _roster(base_urls[0], models, api_key)
    return tuple(_served_model(base_urls[index % len(base_urls)], model, api_key) for index, model in enumerate(models))


def _clean_ordinary_cases(
    lock: dict[str, Any], hard_manifest: dict[str, Any]
) -> tuple[tuple[str, str, tuple[int, int, int], str, str], ...]:
    exposed = lock["prior_exposure"]
    exposed_legacy_seeds = set(exposed["legacy_seeds"])
    exposed_hard_ids = set(exposed["ordinary_hard_case_ids"])
    rows: list[tuple[str, str, tuple[int, int, int], str, str]] = []
    for seed, size, horizon in FROZEN_CROSSPLAY_CASES:
        if seed in exposed_legacy_seeds:
            continue
        for option_order in ("canonical", "permuted-1", "permuted-2"):
            rows.append(
                (
                    f"legacy-{seed}-{option_order}",
                    f"legacy-seed-{seed}",
                    (seed, size, horizon),
                    "ordinary_legacy",
                    option_order,
                )
            )
    for case in hard_manifest["cases"]:
        if case["case_id"] in exposed_hard_ids:
            continue
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


def _clean_handoff_worlds(
    lock: dict[str, Any], handoff_manifest: dict[str, Any]
) -> tuple[tuple[str, str, str, Any, Any], ...]:
    exposed_indices = set(lock["prior_exposure"]["handoff_source_pair_indices"])
    rows = []
    for pair_index, pair in enumerate(handoff_manifest["pairs"]):
        if pair_index in exposed_indices:
            continue
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


def _validate_bindings(
    *,
    lock: dict[str, Any],
    config: dict[str, Any],
    data_dir: Path,
    hard_manifest: dict[str, Any],
    handoff_manifest: dict[str, Any],
    ordinary: tuple[tuple[str, str, tuple[int, int, int], str, str], ...],
    handoffs: tuple[tuple[str, str, str, Any, Any], ...],
) -> None:
    bindings = lock["bindings"]
    for name, expected in bindings["files_sha256"].items():
        if _sha256_file(data_dir / name) != expected:
            raise ValueError(f"frozen input file hash mismatch: {name}")
    manifest_bodies = {
        "ordinary_hard_frozen_ood.json": hard_manifest["sha256"],
        "handoff_frozen_ood.json": handoff_manifest["sha256"],
    }
    if manifest_bodies != bindings["manifest_body_sha256"]:
        raise ValueError("frozen manifest body hashes do not match the lock")
    if config["candidate"]["revision"] != bindings["candidate"]["policy_revision"]:
        raise ValueError("candidate policy revision does not match the lock")
    if config["candidate"].get("adapter_sha256") != bindings["candidate"]["adapter_sha256"]:
        raise ValueError("candidate adapter hashes do not match the lock")
    if config["baseline"]["revision"] != bindings["baseline"]["revision"]:
        raise ValueError("baseline revision does not match the lock")
    if config["baseline"].get("adapter_sha256") != bindings["baseline"]["adapter_sha256"]:
        raise ValueError("baseline adapter hash does not match the lock")
    configured_opponents = {str(item["id"]): str(item["revision"]) for item in config["opponents"]}
    if configured_opponents != bindings["opponents"]:
        raise ValueError("opponent revisions do not match the lock")

    counts = lock["clean_counts"]
    legacy_units = {row[1] for row in ordinary if row[3] == "ordinary_legacy"}
    hard_units = {row[1] for row in ordinary if row[3] == "ordinary_hard"}
    handoff_units = {row[1] for row in handoffs}
    actual = {
        "legacy_independent_seeds": len(legacy_units),
        "legacy_option_order_cases": sum(row[3] == "ordinary_legacy" for row in ordinary),
        "ordinary_hard_cases": len(hard_units),
        "handoff_pairs": len(handoff_units),
    }
    for name, value in actual.items():
        if value != counts[name]:
            raise ValueError(f"clean holdout count mismatch for {name}: {value} != {counts[name]}")


def _opponents(
    config: dict[str, Any], base_urls: tuple[str, ...], api_key: str
) -> dict[str, tuple[str, tuple[ArenaModel, ...]]]:
    opponents = {
        str(item["id"]): (
            str(item["revision"]),
            _distributed_roster(base_urls, list(item["models"]), api_key),
        )
        for item in config["opponents"]
    }
    if set(opponents) != {"base", "sft", "historical_league"}:
        raise ValueError("clean holdout requires base, sft, and historical_league")
    return opponents


def _expected_rows(
    ordinary: tuple[tuple[str, str, tuple[int, int, int], str, str], ...],
    handoffs: tuple[tuple[str, str, str, Any, Any], ...],
    opponent_count: int,
) -> int:
    ordinary_rows = len(ordinary) * opponent_count * 2 * 2
    critical_cases = sum(row[2] == "handoff_critical" for row in handoffs)
    decoy_cases = sum(row[2] == "handoff_decoy" for row in handoffs)
    critical_rows = (
        critical_cases * opponent_count * 2 * (len(CANDIDATE_CRITICAL_CONDITIONS) + len(BASELINE_CRITICAL_CONDITIONS))
    )
    decoy_rows = decoy_cases * opponent_count * 2 * len(DECOY_CONDITIONS)
    return ordinary_rows + critical_rows + decoy_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the once-only clean v10 semantic held-out evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    lock = _load_lock(args.lock)
    if args.confirmation != lock["sha256"]:
        raise ValueError("clean holdout confirmation does not match the frozen lock")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hard_manifest = _load_manifest(args.data_dir / "ordinary_hard_frozen_ood.json")
    handoff_manifest = _load_manifest(args.data_dir / "handoff_frozen_ood.json")
    ordinary = _clean_ordinary_cases(lock, hard_manifest)
    handoffs = _clean_handoff_worlds(lock, handoff_manifest)
    _validate_bindings(
        lock=lock,
        config=config,
        data_dir=args.data_dir,
        hard_manifest=hard_manifest,
        handoff_manifest=handoff_manifest,
        ordinary=ordinary,
        handoffs=handoffs,
    )

    configured_urls = config.get("base_urls", [config.get("base_url")])
    if not isinstance(configured_urls, list) or any(
        not isinstance(value, str) or not value for value in configured_urls
    ):
        raise ValueError("evaluation base_urls must be a non-empty string list")
    base_urls = tuple(configured_urls)
    candidate = _distributed_roster(base_urls, list(config["candidate"]["models"]), args.api_key)
    baseline = _distributed_roster(base_urls, list(config["baseline"]["models"]), args.api_key)
    opponents = _opponents(config, base_urls, args.api_key)
    expected_rows = _expected_rows(ordinary, handoffs, len(opponents))
    if expected_rows != lock["clean_counts"]["expected_rows"]:
        raise ValueError("computed row count does not match the frozen lock")
    if args.audit_only:
        print(
            json.dumps(
                {
                    "version": VERSION,
                    "status": "ready",
                    "lock_sha256": lock["sha256"],
                    "config_sha256": _digest(config),
                    "expected_rows": expected_rows,
                    "clean_legacy_units": len({row[1] for row in ordinary if row[3] == "ordinary_legacy"}),
                    "clean_hard_units": len({row[1] for row in ordinary if row[3] == "ordinary_hard"}),
                    "clean_handoff_units": len({row[1] for row in handoffs}),
                    "ordinary_case_ids_sha256": _digest([row[0] for row in ordinary]),
                    "handoff_case_ids_sha256": _digest([row[0] for row in handoffs]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "version": VERSION,
        "source_commit": source_commit,
        "lock_sha256": lock["sha256"],
        "config": config,
        "config_sha256": _digest(config),
        "expected_rows": expected_rows,
        "ordinary_case_ids": [row[0] for row in ordinary],
        "handoff_case_ids": [row[0] for row in handoffs],
        "candidate_critical_conditions": list(CANDIDATE_CRITICAL_CONDITIONS),
        "baseline_critical_conditions": list(BASELINE_CRITICAL_CONDITIONS),
        "decoy_conditions": list(DECOY_CONDITIONS),
        "sides": ["BLUE", "RED"],
        "generation": {"temperature": 0.0, "max_tokens": 160, "structured": True},
        "scope": "clean unexposed remainder; update-40 selected before this run",
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
        target: str | None = None,
        sender: str | None = None,
        receiver: str | None = None,
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
            ":".join((case_id, opponent_id, side)),
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
            critical_target=target if suite == "handoff_critical" else None,
            target_swap_sender=sender if semantic else None,
            target_swap_targets=candidate_targets if semantic else None,
            target_swap_active_target=target if semantic else None,
            target_swap_receiver=receiver if semantic else None,
        )
        row = {**row, "independent_id": independent_id, "evaluation_id": evaluation_id}
        if sender is not None and receiver is not None and target is not None:
            prefix = side.lower()
            resolved_sender = f"{prefix}-{sender.split('-', 1)[1]}"
            resolved_receiver = f"{prefix}-{receiver.split('-', 1)[1]}"
            sender_broadcast = next(
                item for item in raw["turns"][0]["broadcasts"] if item["agent_id"] == resolved_sender
            )
            receiver_action = next(item for item in raw["turns"][0]["actions"] if item["agent_id"] == resolved_receiver)
            row["sender_target_fact"] = any(
                fact["node"] == target for fact in sender_broadcast["accepted_message"]["facts"]
            )
            row["sender_nonempty"] = bool(
                sender_broadcast["accepted_message"]["facts"]
                or sender_broadcast["accepted_message"]["intent"] is not None
                or sender_broadcast["accepted_message"]["request_resource"]
            )
            row["receiver_target_action"] = receiver_action["selected_action"].get("target") == target
        raw_record = {"evaluation_id": evaluation_id, "raw": raw}
        row["raw_sha256"] = _digest(raw_record)
        # Raw-first durable ordering lets the public mirror treat every compact
        # row as proof that its corresponding full trace has reached disk.
        _append_jsonl_durable(raw_path, raw_record)
        _append_jsonl_durable(rows_path, row)
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
                for side in ("BLUE", "RED"):
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
    baseline_revision = str(config["baseline"]["revision"])
    for case_id, independent_id, suite, scenario, world in handoffs:
        candidate_conditions = CANDIDATE_CRITICAL_CONDITIONS if suite == "handoff_critical" else DECOY_CONDITIONS
        for opponent_id, (opponent_revision, opponent) in opponents.items():
            for side in ("BLUE", "RED"):
                for condition in candidate_conditions:
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
                        target=world.active_target,
                        sender=scenario.sender,
                        receiver=scenario.receiver,
                        candidate_targets=scenario.candidate_targets,
                    )
                if suite == "handoff_critical":
                    for condition in BASELINE_CRITICAL_CONDITIONS:
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
                            target=world.active_target,
                            sender=scenario.sender,
                            receiver=scenario.receiver,
                            candidate_targets=scenario.candidate_targets,
                        )

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != expected_rows:
        raise ValueError(f"completed {len(rows)} rows; expected {expected_rows}")
    standard_rows = [row for row in rows if row["condition"] != "target_swapped"]
    summary = {
        "version": VERSION,
        "scope": "clean unexposed frozen remainder; one selected checkpoint",
        "rows": len(rows),
        "lock_sha256": lock["sha256"],
        "standard": summarize_rl_specific_progress_eval(
            standard_rows,
            intervention_conditions=COMMUNICATION_CONDITIONS[1:],
        ),
        "semantic": summarize_semantic_holdout(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "COMPLETE").write_text(_digest(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
