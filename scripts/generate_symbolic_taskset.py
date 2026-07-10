#!/usr/bin/env python3
"""Generate deterministic symbolic tool-calling task files.

This is intentionally independent of pass@k curation.  It creates a clean,
harder distribution for follow-on experiments when the easy/medium mixed
taskset saturates.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from symbolic_tool_calling_v1.generator import generate_task
from symbolic_tool_calling_v1.models import BenchmarkTask
from symbolic_tool_calling_v1.validation import validate_task


def _write_jsonl(path: Path, tasks: list[BenchmarkTask]) -> None:
    path.write_text("".join(task.model_dump_json() + "\n" for task in tasks))


def _summary(tasks: list[BenchmarkTask]) -> dict:
    return {
        "num_tasks": len(tasks),
        "by_horizon": dict(Counter(task.horizon_bucket for task in tasks)),
        "by_verbosity": dict(Counter(task.verbosity_setting for task in tasks)),
        "by_imbalance": dict(Counter(task.imbalance_setting for task in tasks)),
        "by_depth": dict(Counter(str(task.dependency_depth) for task in tasks)),
        "by_optimal_plan_length": dict(Counter(str(task.optimal_plan_length) for task in tasks)),
        "distractor_ratio": {
            "min": min(task.distractor_ratio for task in tasks),
            "max": max(task.distractor_ratio for task in tasks),
        },
        "recovery_cost": {
            "min": min(task.recovery_cost for task in tasks),
            "max": max(task.recovery_cost for task in tasks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-tasks", type=int, default=600)
    parser.add_argument("--val-tasks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=92000)
    parser.add_argument(
        "--horizon", choices=("short", "medium", "long", "xlong", "xxlong"), default="long"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of horizon buckets to interleave "
            "(e.g. 'long,xlong,xxlong'). Overrides --horizon when set; tasks are "
            "round-robined across buckets for a mixed-length distribution."
        ),
    )
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--distractor-ratio", type=float, default=1.0)
    parser.add_argument("--recovery-cost", type=int, default=4)
    parser.add_argument("--verbosity", choices=("low", "high"), default="high")
    parser.add_argument("--imbalance", choices=("low", "high"), default="high")
    args = parser.parse_args()

    if args.val_tasks <= 0 or args.val_tasks >= args.num_tasks:
        raise ValueError("--val-tasks must be between 1 and num_tasks-1")

    horizons = (
        [h.strip() for h in args.horizons.split(",") if h.strip()] if args.horizons else [args.horizon]
    )

    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    seed = args.seed
    while len(tasks) < args.num_tasks:
        horizon = horizons[len(tasks) % len(horizons)]
        task = generate_task(
            seed,
            horizon_bucket=horizon,
            branching_factor=args.branching_factor,
            distractor_ratio=args.distractor_ratio,
            recovery_cost=args.recovery_cost,
            verbosity_setting=args.verbosity,
            imbalance_setting=args.imbalance,
        )
        seed += 1
        if task.task_id in seen:
            continue
        validate_task(task)
        seen.add(task.task_id)
        tasks.append(task)

    rng = random.Random(args.seed)
    shuffled = tasks[:]
    rng.shuffle(shuffled)
    val = shuffled[: args.val_tasks]
    train = shuffled[args.val_tasks :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "tasks.jsonl", tasks)
    _write_jsonl(args.output_dir / "tasks_train.jsonl", train)
    _write_jsonl(args.output_dir / "tasks_val.jsonl", val)

    config = {
        "num_tasks": args.num_tasks,
        "val_tasks": args.val_tasks,
        "seed": args.seed,
        "horizon": args.horizon,
        "horizons": horizons,
        "branching_factor": args.branching_factor,
        "distractor_ratio": args.distractor_ratio,
        "recovery_cost": args.recovery_cost,
        "verbosity": args.verbosity,
        "imbalance": args.imbalance,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    summary = {"all": _summary(tasks), "train": _summary(train), "val": _summary(val)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
