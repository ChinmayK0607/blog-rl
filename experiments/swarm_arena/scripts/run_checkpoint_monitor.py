from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from swarm_ctf_eval.checkpoint_monitor import (
    due_checkpoint_steps,
    load_checkpoint_monitor_plan,
    run_checkpoint_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe export, online evaluation, regression, collapse, and publication monitor."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("poll interval must be positive")

    plan = load_checkpoint_monitor_plan(args.plan)
    while True:
        if args.progress.exists():
            progress = json.loads(args.progress.read_text(encoding="utf-8"))
            for checkpoint_step in due_checkpoint_steps(progress, plan.every_updates):
                run_checkpoint_tasks(
                    plan,
                    checkpoint_step=checkpoint_step,
                    state_path=args.state,
                )
        if args.once:
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
