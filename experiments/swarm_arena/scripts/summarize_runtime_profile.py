from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path


def summarize_profile(
    progress: list[dict],
    *,
    trainer_gpus: int,
    inference_gpus: int,
    minimum_updates: int,
) -> dict[str, object]:
    rows = [
        row["timing"]
        for row in progress
        if isinstance(row, dict) and isinstance(row.get("timing"), dict)
    ]
    if len(rows) < minimum_updates:
        raise ValueError(
            f"runtime profile requires at least {minimum_updates} timed updates; got {len(rows)}"
        )
    rows = rows[:minimum_updates]
    required = {
        "rollout_generation_seconds",
        "batch_prepare_seconds",
        "trainer_update_seconds",
        "total_seconds",
    }
    if any(set(row) < required for row in rows):
        raise ValueError("runtime profile row is missing a required phase timing")
    if any(
        not math.isfinite(float(row[key])) or float(row[key]) < 0
        for row in rows
        for key in required
    ):
        raise ValueError("runtime profile phase timings must be finite and non-negative")
    medians = {
        key: statistics.median(float(row[key]) for row in rows)
        for key in sorted(required)
    }
    total = medians["total_seconds"]
    if total <= 0:
        raise ValueError("runtime profile total duration must be positive")
    rollout_fraction = medians["rollout_generation_seconds"] / total
    trainer_fraction = medians["trainer_update_seconds"] / total
    if rollout_fraction >= 0.60:
        recommendation = "favor_inference"
    elif trainer_fraction >= 0.35:
        recommendation = "favor_trainer"
    else:
        recommendation = "keep_balanced"
    return {
        "version": "swarm-runtime-profile-v1",
        "scope": "operational_timings_only_no_reward_or_gate_inputs",
        "window": f"first_{minimum_updates}_durable_timed_updates",
        "timed_updates": len(rows),
        "topology": {
            "trainer_gpus": trainer_gpus,
            "inference_gpus": inference_gpus,
        },
        "median_seconds": medians,
        "fractions": {
            "rollout_generation": rollout_fraction,
            "trainer_update": trainer_fraction,
        },
        "recommendation": recommendation,
        "rule": {
            "favor_inference_if_rollout_fraction_at_least": 0.60,
            "favor_trainer_if_trainer_fraction_at_least": 0.35,
            "otherwise": "keep_balanced",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize topology pressure from reward-blind controller phase timings."
    )
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--trainer-gpus", type=int, required=True)
    parser.add_argument("--inference-gpus", type=int, required=True)
    parser.add_argument("--minimum-updates", type=int, default=3)
    parser.add_argument("--wait-timeout", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trainer_gpus < 1 or args.inference_gpus < 1:
        parser.error("trainer and inference GPU counts must be positive")
    if args.minimum_updates < 1:
        parser.error("minimum updates must be positive")
    if args.wait_timeout < 0 or args.poll_seconds <= 0:
        parser.error("wait timeout must be non-negative and poll interval positive")
    deadline = time.monotonic() + args.wait_timeout
    progress: object = None
    while True:
        try:
            progress = json.loads(args.progress.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            progress = None
        timed = (
            sum(
                isinstance(row, dict) and isinstance(row.get("timing"), dict)
                for row in progress
            )
            if isinstance(progress, list)
            else 0
        )
        if timed >= args.minimum_updates or time.monotonic() >= deadline:
            break
        time.sleep(args.poll_seconds)
    if not isinstance(progress, list):
        raise ValueError("controller progress must be a list")
    result = summarize_profile(
        progress,
        trainer_gpus=args.trainer_gpus,
        inference_gpus=args.inference_gpus,
        minimum_updates=args.minimum_updates,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite runtime profile: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
