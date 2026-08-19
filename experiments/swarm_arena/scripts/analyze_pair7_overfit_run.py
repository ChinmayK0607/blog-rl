from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import msgspec
from transformers import AutoTokenizer

from prime_rl.transport.types import TrainingBatch


def _action_record(tokenizer: Any, sample: Any) -> dict[str, Any]:
    prompt = tokenizer.decode(sample.prompt_ids, skip_special_tokens=True)
    user_payload = prompt.rsplit("\nuser\n", 1)[1].split("\nassistant\n", 1)[0]
    request = json.loads(user_payload)
    response = json.loads(tokenizer.decode(sample.completion_ids, skip_special_tokens=True))
    selected = next(action for action in request["legal_actions"] if action["id"] == response["action_id"])
    return {
        "selected_type": selected["type"],
        "selected_target": selected.get("target"),
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows) if rows else 0.0


def _bootstrap_interval(values: list[float], *, trials: int = 20_000) -> list[float]:
    generator = random.Random(0)
    means = sorted(statistics.fmean(values[generator.randrange(len(values))] for _ in values) for _ in range(trials))
    return [means[int(0.025 * (trials - 1))], means[int(0.975 * (trials - 1))]]


def _paired_values(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    field: str,
    intervention: str,
) -> list[float]:
    cells: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if row["kind"] != kind or row["condition"] not in {"normal", intervention}:
            continue
        key = (row["world"], int(row["repeat"]))
        cells.setdefault(key, {})[row["condition"]] = float(row[field])
    if not cells or any(set(cell) != {"normal", intervention} for cell in cells.values()):
        raise ValueError(f"incomplete {kind} normal/{intervention} pairs")
    return [cells[key]["normal"] - cells[key][intervention] for key in sorted(cells)]


def _evaluation_summary(directory: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines() if line]
    effects = {}
    for kind in ("critical", "decoy"):
        effects[kind] = {}
        for intervention in ("dropped", "sender_shuffled"):
            effects[kind][f"normal_minus_{intervention}"] = {}
            for field in ("terminal_return", "receiver_target_action"):
                values = _paired_values(
                    rows,
                    kind=kind,
                    field=field,
                    intervention=intervention,
                )
                effects[kind][f"normal_minus_{intervention}"][field] = {
                    "mean": statistics.fmean(values),
                    "mean_95": _bootstrap_interval(values),
                    "paired_cells": len(values),
                }
    critical = _paired_values(rows, kind="critical", field="terminal_return", intervention="dropped")
    decoy = _paired_values(rows, kind="decoy", field="terminal_return", intervention="dropped")
    specificity = [left - right for left, right in zip(critical, decoy, strict=True)]
    return {
        "rows": len(rows),
        "effects": effects,
        "critical_minus_decoy_normal_dropped_return": {
            "mean": statistics.fmean(specificity),
            "mean_95": _bootstrap_interval(specificity),
            "paired_cells": len(specificity),
        },
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = {}
    for kind in ("critical", "decoy"):
        selected = [row for row in rows if row["kind"] == kind]
        target = [row for row in selected if row["target_action"]]
        other = [row for row in selected if not row["target_action"]]
        worlds = {}
        for world in ("left_exposed", "right_exposed"):
            world_rows = [row for row in selected if row["world"] == world]
            worlds[world] = {
                "samples": len(world_rows),
                "mean_return": _mean(world_rows, "reward"),
                "target_action_rate": _mean(world_rows, "target_action"),
                "target_capture_rate": _mean(world_rows, "target_capture"),
                "selected_targets": dict(sorted(Counter(str(row["selected_target"]) for row in world_rows).items())),
            }
        by_kind[kind] = {
            "samples": len(selected),
            "mean_return": _mean(selected, "reward"),
            "mean_advantage": _mean(selected, "advantage"),
            "positive_advantage_rate": _mean(selected, "positive_advantage"),
            "zero_advantage_rate": _mean(selected, "zero_advantage"),
            "target_action_rate": _mean(selected, "target_action"),
            "target_capture_rate": _mean(selected, "target_capture"),
            "target_action_mean_return": _mean(target, "reward"),
            "other_action_mean_return": _mean(other, "reward"),
            "target_action_mean_advantage": _mean(target, "advantage"),
            "other_action_mean_advantage": _mean(other, "advantage"),
            "target_action_positive_advantage_rate": _mean(target, "positive_advantage"),
            "other_action_positive_advantage_rate": _mean(other, "positive_advantage"),
            "action_types": dict(sorted(Counter(row["selected_type"] for row in selected).items())),
            "worlds": worlds,
        }
    return by_kind


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the receiver decisions in a pair-7 overfit run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-dir", action="append", type=Path, default=[])
    args = parser.parse_args()

    progress = json.loads((args.run_dir / "live_rl_progress.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)
    rows = []
    for update in progress:
        batch_path = args.run_dir / "run_blue_1" / "rollouts" / f"step_{update['step']}" / "train_rollouts.bin"
        batch = decoder.decode(batch_path.read_bytes())
        expected = sum(len(group["replicas"]) for group in update["groups"])
        if len(batch.examples) != expected:
            raise ValueError(f"step {update['step']} has {len(batch.examples)} examples, expected {expected}")
        offset = 0
        for group in update["groups"]:
            scenario = group["scenario"]
            for replica, sample in zip(
                group["replicas"],
                batch.examples[offset : offset + len(group["replicas"])],
                strict=True,
            ):
                action = _action_record(tokenizer, sample)
                target_action = action["selected_target"] == scenario["target"]
                advantage = float(replica["advantages"]["blue-1"])
                rows.append(
                    {
                        "step": int(update["step"]),
                        "kind": scenario["kind"],
                        "world": scenario["world"],
                        "target": scenario["target"],
                        **action,
                        "target_action": target_action,
                        "target_capture": target_action and action["selected_type"] == "CAPTURE",
                        "reward": float(replica["return"]),
                        "advantage": advantage,
                        "positive_advantage": advantage > 0,
                        "zero_advantage": advantage == 0,
                    }
                )
            offset += len(group["replicas"])

    windows = []
    for start in range(0, len(progress), 10):
        selected = [row for row in rows if start <= row["step"] < start + 10]
        windows.append(
            {
                "updates": [start, start + 9],
                "by_kind": _summarize(selected),
            }
        )
    report = {
        "version": "pair7-overfit-training-diagnosis-v1",
        "run_dir": str(args.run_dir),
        "updates": len(progress),
        "samples": len(rows),
        "overall": _summarize(rows),
        "windows": windows,
        "evaluations": {directory.name: _evaluation_summary(directory) for directory in args.evaluation_dir},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
