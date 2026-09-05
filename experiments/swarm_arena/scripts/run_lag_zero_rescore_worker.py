from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from swarm_ctf_eval.async_admission import PolicySnapshot
from swarm_ctf_eval.async_rescore import ASYNC_RESCORE_PROTOCOL_VERSION
from swarm_ctf_eval.prime_rl_bridge import RolloutDecision
from swarm_ctf_eval.safety_supervisor import canonical_sha256


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rescore_decision_id(row: dict[str, object]) -> str:
    return RolloutDecision(**row).decision_id


def process_request(
    request_path: Path,
    *,
    response_dir: Path,
    snapshot_manifest: Path,
    plan_sha256: str,
) -> Path:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_sha256 = request.pop("request_sha256")
    if canonical_sha256(request) != request_sha256:
        raise ValueError(f"rescore request hash mismatch: {request_path}")
    if request["version"] != ASYNC_RESCORE_PROTOCOL_VERSION:
        raise ValueError("rescore request uses an unknown protocol")
    if request["production_plan_sha256"] != plan_sha256:
        raise ValueError("rescore request uses a different production plan")

    manifest = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    if (
        manifest["version"] != ASYNC_RESCORE_PROTOCOL_VERSION
        or manifest["production_plan_sha256"] != plan_sha256
    ):
        raise ValueError("current snapshot manifest is stale or incompatible")
    current_trainable = {
        row["policy_id"]: PolicySnapshot(**row) for row in manifest["snapshots"]
    }
    behavior = tuple(PolicySnapshot(**row) for row in request["behavior_snapshots"])
    current = []
    for snapshot in behavior:
        snapshot.validate()
        if snapshot.trainable:
            live = current_trainable.get(snapshot.policy_id)
            if live is None:
                raise ValueError(f"missing current trainable snapshot: {snapshot.policy_id}")
            if live != snapshot:
                raise ValueError(
                    "lag-zero rescorer refuses a stale behavior snapshot: "
                    f"{snapshot.policy_id}"
                )
            current.append(live)
        else:
            current.append(snapshot)

    response = {
        "version": ASYNC_RESCORE_PROTOCOL_VERSION,
        "mode": "same-backend-lag-zero",
        "rollout_id": request["rollout_id"],
        "request_sha256": request_sha256,
        "current_snapshots": [
            {
                "policy_id": snapshot.policy_id,
                "revision": snapshot.revision,
                "adapter_sha256": snapshot.adapter_sha256,
                "update_index": snapshot.update_index,
                "trainable": snapshot.trainable,
            }
            for snapshot in current
        ],
        "current_policy_logprobs": {
            _rescore_decision_id(row): row["rollout_logprobs"]
            for row in request["decisions"]
        },
    }
    response_path = response_dir / request_path.name
    if response_path.exists():
        raise FileExistsError(f"refusing to overwrite rescore response: {response_path}")
    _atomic_json(response_path, response)
    return response_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serve same-backend current-policy scores only while behavior lag is exactly zero."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--production-plan-sha256", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("poll interval must be positive")

    request_dir = args.root / "requests"
    response_dir = args.root / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    processed = {path.name for path in response_dir.glob("*.json")}
    while True:
        for request_path in sorted(request_dir.glob("*.json")):
            if request_path.name in processed:
                continue
            process_request(
                request_path,
                response_dir=response_dir,
                snapshot_manifest=args.snapshot_manifest,
                plan_sha256=args.production_plan_sha256,
            )
            processed.add(request_path.name)
            if args.once:
                return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
