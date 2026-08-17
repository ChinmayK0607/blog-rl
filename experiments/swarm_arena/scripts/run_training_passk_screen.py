from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from swarm_ctf_eval.arena import NodeObservation, observation_for
from swarm_ctf_eval.arena_eval import ArenaModel, OpenAIArenaModel
from swarm_ctf_eval.arena_protocol import Broadcast
from swarm_ctf_eval.final_eval_runner import FinalEvalIdentity, evaluate_final_case
from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.passk_screen import PASSK_SCREEN_VERSION, summarize_passk
from swarm_ctf_eval.providers import OpenAICompatibleProvider
from swarm_ctf_eval.structured_protocol import protocol_response_format


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


def _model(
    base_url: str,
    model: str,
    *,
    temperature: float,
    seed: int,
) -> ArenaModel:
    provider = OpenAICompatibleProvider(
        base_url,
        model,
        temperature=temperature,
        max_tokens=160,
        enable_thinking=False,
        seed=seed,
        response_format_factory=protocol_response_format,
    )
    return OpenAIArenaModel(provider, model)


def _roster(model: ArenaModel) -> tuple[ArenaModel, ...]:
    return (model, model, model, model)


def _reference_broadcast(state: Any, sender: str, target: str) -> Broadcast:
    known = {
        row["node"]: row for row in observation_for(state, sender)["known_nodes"]
    }
    row = known.get(target)
    if row is None:
        raise ValueError(f"certified sender {sender} does not observe target {target}")
    fact = NodeObservation(
        row["node"],
        row["owner"],
        row["status"],
        row["value"],
        row["critical"],
        row["observed_turn"],
    )
    return Broadcast((fact,), None, 0)


def _stratified_pair_indices(pairs: list[dict[str, Any]], count: int) -> list[int]:
    by_role: dict[tuple[str, str], list[int]] = {}
    for index, pair in enumerate(pairs):
        critical = pair["critical"]
        by_role.setdefault((critical["sender"], critical["receiver"]), []).append(index)
    roles = sorted(by_role)
    if len(roles) != 12:
        raise ValueError(f"expected 12 ordered sender/receiver roles; got {len(roles)}")
    selected = []
    round_index = 0
    while len(selected) < count:
        progressed = False
        for role in roles:
            candidates = by_role[role]
            if round_index < len(candidates) and len(selected) < count:
                selected.append(candidates[round_index])
                progressed = True
        if not progressed:
            raise ValueError(f"cannot select {count} role-stratified pairs")
        round_index += 1
    return selected


def _target_fact(raw: dict[str, Any], sender: str, target: str) -> bool:
    row = next(
        item for item in raw["turns"][0]["broadcasts"] if item["agent_id"] == sender
    )
    return any(fact["node"] == target for fact in row["accepted_message"]["facts"])


def _receiver_target_action(raw: dict[str, Any], receiver: str, target: str) -> bool:
    row = next(
        item for item in raw["turns"][0]["actions"] if item["agent_id"] == receiver
    )
    return row["selected_action"].get("target") == target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen training-only handoff scenarios with stochastic pass@k rollouts."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="sft")
    parser.add_argument("--pair-count", type=int, default=12)
    parser.add_argument("--generated-k", type=int, default=8)
    parser.add_argument("--control-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.pair_count < 12 or args.generated_k < 4 or args.control_k < 2:
        raise ValueError("screen requires at least 12 pairs, generated K=4, and control K=2")

    manifest = _load_manifest(args.manifest)
    pairs = manifest["pairs"]
    pair_indices = _stratified_pair_indices(pairs, args.pair_count)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run_manifest = {
        "version": PASSK_SCREEN_VERSION,
        "source_commit": source_commit,
        "training_manifest_sha256": manifest["sha256"],
        "pair_indices": pair_indices,
        "model": args.model,
        "temperature": args.temperature,
        "generated_k": args.generated_k,
        "control_k": args.control_k,
        "conditions": ["generated", "dropped", "reference"],
        "scope": "training split only; frozen development and OOD evaluation unopened",
        "success": "certified active target owned by focal team at terminal state",
    }
    bound_manifest = {**run_manifest, "sha256": _digest(run_manifest)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    rows_path = args.output_dir / "rows.jsonl"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output_dir}")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != bound_manifest:
            raise ValueError("resume manifest differs from the existing screen")
    else:
        manifest_path.write_text(
            json.dumps(bound_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    completed = {
        json.loads(line)["evaluation_id"]
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    } if rows_path.exists() else set()

    started = time.time()
    games = len(completed)
    for pair_index in pair_indices:
        pair = pairs[pair_index]
        for kind in ("critical", "decoy"):
            scenario = reconstruct_manifest_scenario(pair[kind])
            for world in scenario.worlds:
                case = (scenario.seed, scenario.size, scenario.horizon)
                reference = _reference_broadcast(
                    world.state, scenario.sender, world.active_target
                )
                for condition in ("generated", "dropped", "reference"):
                    repetitions = (
                        args.generated_k
                        if condition == "generated" and kind == "critical"
                        else args.control_k
                    )
                    for repeat in range(repetitions):
                        evaluation_id = (
                            f"pair-{pair_index}:{kind}:{world.label}:{condition}:{repeat}"
                        )
                        if evaluation_id in completed:
                            continue
                        sampling_seed = (
                            scenario.seed * 1_000_003
                            + repeat * 97
                            + 11
                        ) % (2**32)
                        focal_model = _model(
                            args.base_url,
                            args.model,
                            temperature=args.temperature,
                            seed=sampling_seed,
                        )
                        opponent_model = _model(
                            args.base_url,
                            args.model,
                            temperature=args.temperature,
                            seed=(sampling_seed + 1_000_000_007) % (2**32),
                        )
                        identity = FinalEvalIdentity(
                            evaluation_id,
                            f"handoff_{kind}",
                            "sft_screen",
                            args.model,
                            "identity",
                            "identity",
                            "canonical",
                            "sft_model_opponent",
                            args.model,
                            evaluation_id,
                        )
                        row, raw = evaluate_final_case(
                            _roster(focal_model),
                            _roster(opponent_model),
                            case,
                            identity,
                            focal_side="BLUE",
                            condition="dropped" if condition == "dropped" else "normal",
                            initial_state=world.state,
                            critical_target=world.active_target if kind == "critical" else None,
                            turn_zero_broadcast_overrides=(
                                {scenario.sender: reference} if condition == "reference" else None
                            ),
                        )
                        final_owner = raw["turns"][-1]["post_state"]["nodes"][
                            world.active_target
                        ]["owner"]
                        first_owner = raw["turns"][0]["post_state"]["nodes"][
                            world.active_target
                        ]["owner"]
                        compact = {
                            "evaluation_id": evaluation_id,
                            "pair_index": pair_index,
                            "kind": kind,
                            "world": world.label,
                            "sender": scenario.sender,
                            "receiver": scenario.receiver,
                            "target": world.active_target,
                            "condition": condition,
                            "repeat": repeat,
                            "sampling_seed": sampling_seed,
                            "terminal_return": row["terminal_return"],
                            "target_captured": final_owner == "BLUE",
                            "target_captured_turn_zero": first_owner == "BLUE",
                            "sender_target_fact": _target_fact(
                                raw, scenario.sender, world.active_target
                            ),
                            "receiver_target_action": _receiver_target_action(
                                raw, scenario.receiver, world.active_target
                            ),
                            "protocol_valid": all(
                                row[name] == 1.0
                                for name in (
                                    "broadcast_protocol_rate",
                                    "broadcast_grounded_rate",
                                    "action_protocol_rate",
                                )
                            ),
                            "requests": raw["inference"]["requests"],
                            "completion_tokens": raw["inference"]["completion_tokens"],
                        }
                        with rows_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(compact, sort_keys=True) + "\n")
                        completed.add(evaluation_id)
                        games += 1
                        if games % 12 == 0:
                            elapsed = time.time() - started
                            print(
                                json.dumps(
                                    {
                                        "games": games,
                                        "total": sum(
                                            args.generated_k + 2 * args.control_k
                                            if kind_ == "critical"
                                            else 3 * args.control_k
                                            for _ in pair_indices
                                            for kind_ in ("critical", "decoy")
                                            for _ in range(2)
                                        ),
                                        "elapsed_seconds": elapsed,
                                    }
                                ),
                                flush=True,
                            )

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    summary = summarize_passk(rows)
    summary["manifest_sha256"] = bound_manifest["sha256"]
    summary["wall_seconds"] = time.time() - started
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
