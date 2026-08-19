from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from swarm_ctf_eval.arena_eval import ArenaModel, OpenAIArenaModel
from swarm_ctf_eval.final_eval_runner import FinalEvalIdentity, evaluate_final_case
from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.providers import OpenAICompatibleProvider
from swarm_ctf_eval.structured_protocol import protocol_response_format

VERSION = "pair7-communication-overfit-eval-v1"
MULTIPAIR_VERSION = "multipair-communication-learnability-eval-v2"
CONDITIONS = ("normal", "dropped", "sender_shuffled")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _model(base_url: str, name: str, *, temperature: float, seed: int) -> ArenaModel:
    provider = OpenAICompatibleProvider(
        base_url.rstrip("/") + "/v1",
        name,
        temperature=temperature,
        max_tokens=160,
        enable_thinking=False,
        seed=seed,
        response_format_factory=protocol_response_format,
    )
    return OpenAIArenaModel(provider, name)


def _roster(
    base_urls: tuple[str, ...], names: tuple[str, ...], *, temperature: float, seed: int
) -> tuple[ArenaModel, ...]:
    if len(names) != 4:
        raise ValueError("a roster requires four model aliases")
    return tuple(
        _model(
            base_urls[index % len(base_urls)],
            name,
            temperature=temperature,
            seed=(seed + 1_000_003 * index) % (2**32),
        )
        for index, name in enumerate(names)
    )


def _turn_zero(raw: dict[str, Any], section: str, agent_id: str) -> dict[str, Any]:
    return next(row for row in raw["turns"][0][section] if row["agent_id"] == agent_id)


def _mean(rows: list[dict[str, Any]], field: str, **filters: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if all(str(row[key]) == value for key, value in filters.items())
    ]
    return statistics.fmean(values) if values else 0.0


def _summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    critical_generated = _mean(rows, "terminal_return", kind="critical", condition="normal")
    critical_dropped = _mean(rows, "terminal_return", kind="critical", condition="dropped")
    critical_shuffled = _mean(
        rows, "terminal_return", kind="critical", condition="sender_shuffled"
    )
    decoy_generated = _mean(rows, "terminal_return", kind="decoy", condition="normal")
    decoy_dropped = _mean(rows, "terminal_return", kind="decoy", condition="dropped")
    return {
        "critical": {
            "normal_return": critical_generated,
            "normal_minus_dropped_return": critical_generated - critical_dropped,
            "normal_minus_shuffled_return": critical_generated - critical_shuffled,
            "normal_receiver_target_action_rate": _mean(
                rows, "receiver_target_action", kind="critical", condition="normal"
            ),
            "dropped_receiver_target_action_rate": _mean(
                rows, "receiver_target_action", kind="critical", condition="dropped"
            ),
            "normal_sender_target_fact_rate": _mean(
                rows, "sender_target_fact", kind="critical", condition="normal"
            ),
        },
        "specificity": {
            "critical_minus_decoy_normal_dropped_lift": (
                critical_generated - critical_dropped
            )
            - (decoy_generated - decoy_dropped),
        },
        "protocol": {
            "broadcast_valid_rate": _mean(rows, "broadcast_valid"),
            "broadcast_grounded_rate": _mean(rows, "broadcast_grounded"),
            "action_valid_rate": _mean(rows, "action_valid"),
        },
    }


def summarize(
    rows: list[dict[str, Any]],
    pair_indices: tuple[int, ...] = (7,),
) -> dict[str, Any]:
    version = VERSION if pair_indices == (7,) else MULTIPAIR_VERSION
    return {
        "version": version,
        "rows": len(rows),
        "pair_indices": list(pair_indices),
        **_summary_metrics(rows),
        "by_pair": {
            str(pair_index): _summary_metrics(
                [row for row in rows if row.get("pair_index", 7) == pair_index]
            )
            for pair_index in pair_indices
        },
        "interpretation": (
            "This measures training-pair message-conditioned learning only. "
            "Held-out development remains the generalization measure."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast matched intervention eval for the repeated pair-7 "
            "communication curriculum."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--focal-model", action="append", required=True)
    parser.add_argument("--opponent-model", action="append", required=True)
    parser.add_argument("--focal-revision", required=True)
    parser.add_argument("--opponent-revision", required=True)
    parser.add_argument("--pair-index", type=int, action="append", default=[])
    parser.add_argument("--remaining-turns", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument(
        "--receiver-action-prompt-profile",
        choices=("full", "focused_handoff_compact"),
        default="full",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.focal_model) != 4 or len(args.opponent_model) != 4:
        parser.error("exactly four focal and four opponent model aliases are required")
    if not args.base_url or args.remaining_turns < 1 or args.repetitions < 1:
        parser.error("base URL, remaining turns, and repetitions must be positive")
    pair_indices = tuple(args.pair_index or (7,))
    if len(pair_indices) != len(set(pair_indices)) or any(value < 0 for value in pair_indices):
        parser.error("pair indices must be unique and non-negative")

    training = json.loads(args.manifest.read_text(encoding="utf-8"))
    if max(pair_indices) >= len(training["pairs"]):
        parser.error("pair index exceeds the training manifest")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    body = {
        "version": VERSION if pair_indices == (7,) else MULTIPAIR_VERSION,
        "source_commit": source_commit,
        "training_manifest_sha256": training["sha256"],
        "pair_indices": list(pair_indices),
        "focal_models": args.focal_model,
        "opponent_models": args.opponent_model,
        "focal_revision": args.focal_revision,
        "opponent_revision": args.opponent_revision,
        "conditions": list(CONDITIONS),
        "remaining_turns": args.remaining_turns,
        "repetitions": args.repetitions,
        "temperature": args.temperature,
        "receiver_action_prompt_profile": args.receiver_action_prompt_profile,
        "claim_boundary": "training-pair learnability, not held-out communication generalization",
    }
    bound = {**body, "sha256": _digest(body)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    rows_path = args.output_dir / "rows.jsonl"
    if manifest_path.is_file():
        if not args.resume or json.loads(manifest_path.read_text()) != bound:
            raise ValueError("refusing to overwrite or resume a different pair-7 eval")
    elif rows_path.exists():
        raise ValueError("rows exist without an evaluation manifest")
    else:
        manifest_path.write_text(json.dumps(bound, indent=2, sort_keys=True) + "\n")
    completed = {
        json.loads(line)["evaluation_id"]
        for line in rows_path.read_text().splitlines()
        if line
    } if rows_path.is_file() else set()

    base_urls = tuple(args.base_url)
    started = time.time()
    for pair_index in pair_indices:
        pair = training["pairs"][pair_index]
        for kind in ("critical", "decoy"):
            scenario = reconstruct_manifest_scenario(pair[kind])
            for world in scenario.worlds:
                for condition in CONDITIONS:
                    for repeat in range(args.repetitions):
                        evaluation_id = (
                            f"pair-{pair_index}:{kind}:{world.label}:"
                            f"{condition}:{repeat}"
                        )
                        if evaluation_id in completed:
                            continue
                        seed = (scenario.seed * 1_000_003 + repeat * 97 + 11) % (2**32)
                        focal = _roster(
                            base_urls, tuple(args.focal_model), temperature=args.temperature, seed=seed
                        )
                        opponent = _roster(
                            base_urls,
                            tuple(args.opponent_model),
                            temperature=args.temperature,
                            seed=(seed + 1_000_000_007) % (2**32),
                        )
                        identity = FinalEvalIdentity(
                            evaluation_id,
                            f"handoff_{kind}",
                            "candidate_rl",
                            args.focal_revision,
                            "identity",
                            "identity",
                            "canonical",
                            "sft",
                            args.opponent_revision,
                            evaluation_id,
                        )
                        row, raw = evaluate_final_case(
                            focal,
                            opponent,
                            (scenario.seed, scenario.size, world.state.turn + args.remaining_turns),
                            identity,
                            focal_side="BLUE",
                            condition=condition,
                            initial_state=world.state,
                            critical_target=world.active_target if kind == "critical" else None,
                            action_prompt_profiles={
                                scenario.receiver: args.receiver_action_prompt_profile
                            },
                        )
                        broadcast = _turn_zero(raw, "broadcasts", scenario.sender)
                        action = _turn_zero(raw, "actions", scenario.receiver)
                        result = {
                            "evaluation_id": evaluation_id,
                            "pair_index": pair_index,
                            "kind": kind,
                            "world": world.label,
                            "condition": condition,
                            "repeat": repeat,
                            "sampling_seed": seed,
                            "target": world.active_target,
                            "receiver_action_prompt_profile": (
                                args.receiver_action_prompt_profile
                            ),
                            "terminal_return": row["terminal_return"],
                            "sender_target_fact": any(
                                fact["node"] == world.active_target
                                for fact in broadcast["accepted_message"]["facts"]
                            ),
                            "receiver_target_action": (
                                action["selected_action"].get("target") == world.active_target
                            ),
                            "broadcast_valid": row["broadcast_protocol_rate"],
                            "broadcast_grounded": row["broadcast_grounded_rate"],
                            "action_valid": row["action_protocol_rate"],
                        }
                        with rows_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(result, sort_keys=True) + "\n")
                        completed.add(evaluation_id)

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    summary = {
        **summarize(rows, pair_indices),
        "receiver_action_prompt_profile": args.receiver_action_prompt_profile,
        "temperature": args.temperature,
        "manifest_sha256": bound["sha256"],
        "wall_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
