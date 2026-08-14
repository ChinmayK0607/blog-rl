from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from swarm_ctf_eval.arena_eval import ArenaModel, OpenAIArenaModel
from swarm_ctf_eval.communication_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.crossplay_eval import development_cases
from swarm_ctf_eval.final_eval_runner import FinalEvalIdentity, evaluate_final_case
from swarm_ctf_eval.providers import OpenAICompatibleProvider
from swarm_ctf_eval.structured_protocol import protocol_response_format

CONDITIONS = ("normal", "dropped", "sender_shuffled", "delayed", "zero_budget")
VERSION = "arena-final-eval-development-v1"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _served_model(base_url: str, model: str, api_key: str) -> ArenaModel:
    provider = OpenAICompatibleProvider(
        base_url,
        model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=160,
        enable_thinking=False,
        response_format_factory=protocol_response_format,
    )
    return OpenAIArenaModel(provider, model)


def _roster(base_url: str, models: list[str], api_key: str) -> tuple[ArenaModel, ...]:
    if len(models) != 4:
        raise ValueError("every evaluation roster must contain exactly four model IDs")
    cache: dict[str, ArenaModel] = {}
    result = []
    for model in models:
        if model not in cache:
            cache[model] = _served_model(base_url, model, api_key)
        result.append(cache[model])
    return tuple(result)


def _prepare_output(output_dir: Path, manifest: dict[str, Any], resume: bool) -> set[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    rows_path = output_dir / "rows.jsonl"
    bound = json.loads(
        json.dumps(
            {**manifest, "sha256": _canonical_sha256(manifest)},
            sort_keys=True,
        )
    )
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output_dir}")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != bound:
            raise ValueError("resume manifest differs from the existing evaluation")
    elif rows_path.exists():
        raise ValueError("evaluation rows exist without a manifest")
    else:
        manifest_path.write_text(
            json.dumps(bound, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not rows_path.exists():
        return set()
    return {
        str(json.loads(line)["evaluation_id"])
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    }


def _mean_difference(
    rows: list[dict[str, Any]],
    *,
    field: str,
    left: str,
    right: str,
    filters: dict[str, str],
    match: tuple[str, ...],
) -> dict[str, Any]:
    selected = [
        row for row in rows if all(str(row[key]) == value for key, value in filters.items())
    ]
    grouped: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in selected:
        level = str(row[field])
        if level not in {left, right}:
            continue
        key = tuple(row[name] for name in match)
        bucket = grouped.setdefault(key, {})
        if level in bucket:
            raise ValueError(f"duplicate paired cell: {key}/{level}")
        bucket[level] = float(row["terminal_return"])
    complete = [values[left] - values[right] for values in grouped.values() if set(values) == {left, right}]
    if not complete:
        raise ValueError(f"no complete {left}/{right} pairs for {filters}")
    return {
        "paired_cells": len(complete),
        "mean_difference": statistics.mean(complete),
        "positive_rate": statistics.mean(value > 0 for value in complete),
        "min": min(complete),
        "max": max(complete),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    common = ("case_id", "opponent_id", "opponent_revision", "side", "sampling_key")
    capability = _mean_difference(
        rows,
        field="policy_variant",
        left="candidate_rl",
        right="sft_init",
        filters={"suite": "ordinary_ood", "condition": "normal"},
        match=common,
    )
    communication = {
        condition: _mean_difference(
            rows,
            field="condition",
            left="normal",
            right=condition,
            filters={"suite": "critical", "policy_variant": "candidate_rl"},
            match=common + ("policy_variant", "policy_revision"),
        )
        for condition in CONDITIONS[1:]
    }
    baseline_message_effect = _mean_difference(
        rows,
        field="condition",
        left="normal",
        right="dropped",
        filters={"suite": "critical", "policy_variant": "sft_init"},
        match=common + ("policy_variant", "policy_revision"),
    )
    decoy_message_effect = _mean_difference(
        rows,
        field="condition",
        left="normal",
        right="dropped",
        filters={"suite": "decoy", "policy_variant": "candidate_rl"},
        match=common + ("policy_variant", "policy_revision"),
    )
    candidate_normal = [
        row
        for row in rows
        if row["policy_variant"] == "candidate_rl" and row["condition"] == "normal"
    ]
    return {
        "version": VERSION,
        "rows": len(rows),
        "ordinary_candidate_minus_sft": capability,
        "critical_normal_minus_intervention": communication,
        "critical_sft_normal_minus_dropped": baseline_message_effect,
        "decoy_candidate_normal_minus_dropped": decoy_message_effect,
        "candidate_normal_return_by_opponent": {
            opponent: statistics.mean(
                float(row["terminal_return"])
                for row in candidate_normal
                if row["opponent_id"] == opponent
            )
            for opponent in sorted({str(row["opponent_id"]) for row in candidate_normal})
        },
        "candidate_normal_protocol": {
            field: statistics.mean(float(row[field]) for row in candidate_normal)
            for field in (
                "broadcast_protocol_rate",
                "broadcast_grounded_rate",
                "action_protocol_rate",
            )
        },
        "scope": "development-only checkpoint selection; frozen OOD remains unopened",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a compact four-policy, multi-opponent development evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ordinary-cases", type=int, default=3)
    parser.add_argument("--curriculum-pairs", type=int, default=2)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.ordinary_cases < 1 or args.curriculum_pairs < 1:
        parser.error("case counts must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_url = str(config["base_url"])
    candidate = _roster(base_url, list(config["candidate"]["models"]), args.api_key)
    baseline = _roster(base_url, list(config["baseline"]["models"]), args.api_key)
    opponents = {
        str(item["id"]): (
            str(item["revision"]),
            _roster(base_url, list(item["models"]), args.api_key),
        )
        for item in config["opponents"]
    }
    if len(opponents) < 3:
        raise ValueError("development evaluation requires at least three opponents")
    curriculum = json.loads((args.data_dir / "development.json").read_text(encoding="utf-8"))
    selected_pairs = list(curriculum["pairs"][: args.curriculum_pairs])
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "version": VERSION,
        "source_commit": source_commit,
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "development_manifest_sha256": curriculum["sha256"],
        "ordinary_cases": development_cases(args.ordinary_cases),
        "curriculum_pair_state_sha256": [
            {
                "critical": pair["critical"]["state_sha256"],
                "decoy": pair["decoy"]["state_sha256"],
            }
            for pair in selected_pairs
        ],
        "conditions": list(CONDITIONS),
        "sides": ["BLUE", "RED"],
        "generation": {"temperature": 0.0, "max_tokens": 160, "structured": True},
    }
    completed = _prepare_output(args.output_dir, manifest, args.resume)
    rows_path = args.output_dir / "rows.jsonl"
    raw_path = args.output_dir / "raw.jsonl"

    def run_one(
        *,
        case_id: str,
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
        initial_state: Any | None = None,
        target: str | None = None,
    ) -> None:
        evaluation_id = ":".join(
            (case_id, variant, revision, opponent_id, opponent_revision, side, condition)
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
            "canonical",
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
            critical_target=target,
        )
        raw_record = {"evaluation_id": evaluation_id, "raw": raw}
        row = {**row, "evaluation_id": evaluation_id, "raw_sha256": _canonical_sha256(raw_record)}
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
    for seed, size, horizon in development_cases(args.ordinary_cases):
        case_id = f"ordinary-{seed}"
        for opponent_id, (opponent_revision, opponent) in opponents.items():
            for variant, revision, focal in variants:
                for side in ("BLUE", "RED"):
                    run_one(
                        case_id=case_id,
                        suite="ordinary_ood",
                        case=(seed, size, horizon),
                        variant=variant,
                        revision=revision,
                        focal=focal,
                        opponent_id=opponent_id,
                        opponent_revision=opponent_revision,
                        opponent=opponent,
                        side=side,
                        condition="normal",
                    )

    for pair_index, pair in enumerate(selected_pairs):
        for kind in ("critical", "decoy"):
            scenario = reconstruct_manifest_scenario(pair[kind])
            conditions = CONDITIONS if kind == "critical" else ("normal", "dropped")
            kind_variants = variants if kind == "critical" else variants[:1]
            for opponent_id, (opponent_revision, opponent) in opponents.items():
                for variant, revision, focal in kind_variants:
                    selected_conditions = conditions if variant == "candidate_rl" else ("normal", "dropped")
                    for side in ("BLUE", "RED"):
                        for condition in selected_conditions:
                            run_one(
                                case_id=f"development-pair-{pair_index}-{kind}",
                                suite=kind,
                                case=(scenario.seed, scenario.size, 2),
                                variant=variant,
                                revision=revision,
                                focal=focal,
                                opponent_id=opponent_id,
                                opponent_revision=opponent_revision,
                                opponent=opponent,
                                side=side,
                                condition=condition,
                                initial_state=scenario.state,
                                target=scenario.target if kind == "critical" else None,
                            )

    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = _summary(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
