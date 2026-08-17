from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from swarm_ctf_eval.arena_eval import ArenaModel, OpenAIArenaModel
from swarm_ctf_eval.final_eval_runner import FinalEvalIdentity, evaluate_final_case
from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.passk_screen import (
    TERMINAL_PROXIMAL_SCREEN_VERSION,
    summarize_terminal_proximal,
)
from swarm_ctf_eval.providers import OpenAICompatibleProvider
from swarm_ctf_eval.structured_protocol import protocol_response_format


def _digest_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _model(base_url: str, model: str, *, temperature: float, seed: int) -> ArenaModel:
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


def _target_fact(raw: dict[str, Any], sender: str, target: str) -> bool:
    row = next(item for item in raw["turns"][0]["broadcasts"] if item["agent_id"] == sender)
    return any(fact["node"] == target for fact in row["accepted_message"]["facts"])


def _receiver_target_action(raw: dict[str, Any], receiver: str, target: str) -> bool:
    row = next(item for item in raw["turns"][0]["actions"] if item["agent_id"] == receiver)
    return row["selected_action"].get("target") == target


def _find_world(scenario: Any, label: str) -> Any:
    return next(world for world in scenario.worlds if world.label == label)


def _run_case(
    focal_model: ArenaModel,
    opponent_model: ArenaModel,
    *,
    evaluation_id: str,
    kind: str,
    scenario: Any,
    world: Any,
    condition: str,
    horizon: int,
) -> dict[str, Any]:
    identity = FinalEvalIdentity(
        evaluation_id,
        f"handoff_{kind}",
        "terminal_proximal_screen",
        "sft",
        "identity",
        "identity",
        "canonical",
        "sft_model_opponent",
        "sft",
        evaluation_id,
    )
    row, raw = evaluate_final_case(
        _roster(focal_model),
        _roster(opponent_model),
        (scenario.seed, scenario.size, horizon),
        identity,
        focal_side="BLUE",
        condition=condition,
        initial_state=world.state,
        critical_target=world.active_target if kind == "critical" else None,
    )
    return {
        "terminal_return": row["terminal_return"],
        "target_captured": (
            raw["turns"][-1]["post_state"]["nodes"][world.active_target]["owner"] == "BLUE"
        ),
        "target_captured_turn_zero": (
            raw["turns"][0]["post_state"]["nodes"][world.active_target]["owner"] == "BLUE"
        ),
        "sender_target_fact": _target_fact(raw, scenario.sender, world.active_target),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare selected training handoffs at a terminal-proximal horizon."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-analysis", type=Path, required=True)
    parser.add_argument("--baseline-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="sft")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.horizon < 2 or args.repetitions != 4:
        raise ValueError("the matched screen requires horizon >=2 and exactly four repetitions")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = json.loads(args.selection_analysis.read_text(encoding="utf-8"))
    selected = analysis["selection"]["bands"]["primary_receiver_band"]
    cases = [(int(row["pair_index"]), str(row["world"])) for row in selected]
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run_body = {
        "version": TERMINAL_PROXIMAL_SCREEN_VERSION,
        "source_commit": source_commit,
        "training_manifest_sha256": manifest["sha256"],
        "selection_analysis_file_sha256": _digest_bytes(args.selection_analysis),
        "baseline_rows_file_sha256": _digest_bytes(args.baseline_rows),
        "selected_cases": [
            {"pair_index": pair_index, "world": world} for pair_index, world in cases
        ],
        "model": args.model,
        "temperature": args.temperature,
        "horizon": args.horizon,
        "repetitions": args.repetitions,
        "conditions": ["generated", "dropped"],
        "scope": "training split only; development, selection, and frozen OOD tiers unopened",
    }
    run_manifest = {
        **run_body,
        "sha256": hashlib.sha256(
            json.dumps(run_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    rows_path = args.output_dir / "rows.jsonl"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != run_manifest:
            raise ValueError("refusing to overwrite or resume a different screen")
    else:
        manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    completed = {
        row["evaluation_id"] for row in _load_rows(rows_path)
    } if rows_path.exists() else set()
    started = time.time()
    total = len(cases) * 2 * 2 * args.repetitions
    for pair_index, world_label in cases:
        pair = manifest["pairs"][pair_index]
        for kind in ("critical", "decoy"):
            scenario = reconstruct_manifest_scenario(pair[kind])
            world = _find_world(scenario, world_label)
            for condition in ("generated", "dropped"):
                for repeat in range(args.repetitions):
                    evaluation_id = (
                        f"h{args.horizon}:pair-{pair_index}:{kind}:{world_label}:{condition}:{repeat}"
                    )
                    if evaluation_id in completed:
                        continue
                    sampling_seed = (scenario.seed * 1_000_003 + repeat * 97 + 11) % (2**32)
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
                    result = _run_case(
                        focal_model,
                        opponent_model,
                        evaluation_id=evaluation_id,
                        kind=kind,
                        scenario=scenario,
                        world=world,
                        condition="dropped" if condition == "dropped" else "normal",
                        horizon=args.horizon,
                    )
                    compact = {
                        "evaluation_id": evaluation_id,
                        "pair_index": pair_index,
                        "kind": kind,
                        "world": world_label,
                        "sender": scenario.sender,
                        "receiver": scenario.receiver,
                        "target": world.active_target,
                        "condition": condition,
                        "repeat": repeat,
                        "sampling_seed": sampling_seed,
                        "horizon": args.horizon,
                        **result,
                    }
                    with rows_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(compact, sort_keys=True) + "\n")
                    completed.add(evaluation_id)
                    if len(completed) % 16 == 0:
                        print(
                            json.dumps(
                                {
                                    "games": len(completed),
                                    "total": total,
                                    "elapsed_seconds": time.time() - started,
                                }
                            ),
                            flush=True,
                        )

    rows = _load_rows(rows_path)
    summary = summarize_terminal_proximal(rows, _load_rows(args.baseline_rows))
    summary["manifest_sha256"] = run_manifest["sha256"]
    summary["wall_seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
