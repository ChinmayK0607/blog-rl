"""Reject infeasible schedules before starting trainer/controller processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from swarm_ctf_eval.evaluation_contract import staged_evaluation_budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--interval", type=int, required=True)
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--deadline-epoch", type=int)
    time_group.add_argument("--available-seconds", type=float)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--topology", required=True, help="trainer IDs/inference IDs, e.g. 0/1,2,3")
    parser.add_argument("--gpu-model", required=True, help="exact GPU model used by the operational measurements")
    parser.add_argument("--final-sync-seconds", type=float, default=2700)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if profile["version"] != "staged-operational-profile-v1":
        raise ValueError("unsupported operational profile")
    for name, path in (("inference", args.inference_config), ("trainer", args.trainer_config)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != profile[name + "_config_sha256"]:
            raise ValueError(f"operational profile {name} configuration mismatch")
    if profile["topology"] != args.topology or profile["game_concurrency"] != 1:
        raise ValueError("operational profile does not match the sequential evaluation topology")
    if profile["gpu_model"] != args.gpu_model:
        raise ValueError("operational profile GPU model mismatch")
    if profile["checkpoint_seconds"] < 600:
        raise ValueError("checkpoint reserve must cover the 600-second retained-checkpoint timeout")
    # Rates must come from declared operational measurements, not reward-based selection.
    if not profile.get("evidence"):
        raise ValueError("operational measurements need source evidence references")
    report = staged_evaluation_budget(
        updates=args.expected_updates,
        interval=args.interval,
        games_per_minute=profile["games_per_minute"],
        update_seconds=profile["update_seconds"],
        available_seconds=(args.deadline_epoch - time.time() if args.deadline_epoch is not None else args.available_seconds),
        setup_seconds=profile["remaining_setup_seconds"],
        final_sync_seconds=args.final_sync_seconds,
        safety_factor=profile["safety_factor"],
        checkpoint_seconds=profile["checkpoint_seconds"],
    )
    report["profile_sha256"] = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    report["profile"] = profile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["fits"]:
        raise SystemExit("schedule exceeds the remaining budget; do not launch or shorten frozen gates")


if __name__ == "__main__":
    main()
