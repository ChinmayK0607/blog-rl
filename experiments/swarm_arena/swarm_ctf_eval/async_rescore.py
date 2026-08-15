from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

from .async_admission import PolicySnapshot
from .prime_rl_bridge import RolloutDecision
from .safety_supervisor import canonical_sha256

ASYNC_RESCORE_PROTOCOL_VERSION = "arena-current-policy-rescore-v1"


def _request_payload(
    rollout_id: str,
    plan_sha256: str,
    behavior_snapshots: tuple[PolicySnapshot, ...],
    decisions: tuple[RolloutDecision, ...],
) -> dict[str, object]:
    return {
        "version": ASYNC_RESCORE_PROTOCOL_VERSION,
        "rollout_id": rollout_id,
        "production_plan_sha256": plan_sha256,
        "behavior_snapshots": [asdict(snapshot) for snapshot in behavior_snapshots],
        "decisions": [asdict(decision) for decision in decisions],
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class FilesystemCurrentPolicyRescorer:
    """Exchange immutable rescore requests with a GPU scoring worker."""

    def __init__(self, root: Path, *, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("rescore timeout must be positive")
        self.root = root
        self.timeout = timeout

    async def rescore(
        self,
        *,
        rollout_id: str,
        plan_sha256: str,
        behavior_snapshots: tuple[PolicySnapshot, ...],
        decisions: tuple[RolloutDecision, ...],
    ) -> tuple[tuple[PolicySnapshot, ...], dict[str, tuple[float, ...]]]:
        if not decisions:
            raise ValueError("current-policy rescore requires decisions")
        payload = _request_payload(
            rollout_id,
            plan_sha256,
            behavior_snapshots,
            decisions,
        )
        request_sha256 = canonical_sha256(payload)
        request = {
            **payload,
            "request_sha256": request_sha256,
        }
        inbox = self.root / "requests" / f"{rollout_id}.json"
        response_path = self.root / "responses" / f"{rollout_id}.json"
        if inbox.exists() or response_path.exists():
            raise FileExistsError(f"refusing to reuse async rescore ID: {rollout_id}")
        _atomic_json(inbox, request)

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline and not response_path.exists():
            await asyncio.sleep(0.2)
        if not response_path.exists():
            raise TimeoutError(f"current-policy rescore timed out: {rollout_id}")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("version") != ASYNC_RESCORE_PROTOCOL_VERSION:
            raise ValueError("rescore response uses an unknown protocol")
        if response.get("rollout_id") != rollout_id:
            raise ValueError("rescore response has the wrong rollout ID")
        if response.get("request_sha256") != request_sha256:
            raise ValueError("rescore response does not bind the immutable request")

        snapshots = tuple(PolicySnapshot(**row) for row in response["current_snapshots"])
        for snapshot in snapshots:
            snapshot.validate()
        raw_logprobs = response["current_policy_logprobs"]
        expected = {decision.decision_id: len(decision.completion_ids) for decision in decisions}
        if set(raw_logprobs) != set(expected):
            raise ValueError("rescore response does not cover the exact selected decisions")
        logprobs = {
            decision_id: tuple(float(value) for value in values)
            for decision_id, values in raw_logprobs.items()
        }
        for decision_id, values in logprobs.items():
            if len(values) != expected[decision_id] or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(f"invalid rescore log-prob row: {decision_id}")
        return snapshots, logprobs


def write_current_snapshot_manifest(
    path: Path,
    *,
    plan_sha256: str,
    snapshots: tuple[PolicySnapshot, ...],
) -> None:
    if not snapshots or any(not snapshot.trainable for snapshot in snapshots):
        raise ValueError("current snapshot manifest contains a missing or frozen policy")
    for snapshot in snapshots:
        snapshot.validate()
    _atomic_json(
        path,
        {
            "version": ASYNC_RESCORE_PROTOCOL_VERSION,
            "production_plan_sha256": plan_sha256,
            "snapshots": [asdict(snapshot) for snapshot in snapshots],
        },
    )
