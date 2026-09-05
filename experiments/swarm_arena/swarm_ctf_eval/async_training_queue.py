from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .async_admission import (
    AsyncAdmissionLimits,
    AsyncAdmissionResult,
    AsyncRolloutHeader,
    PolicySnapshot,
    admit_async_rollout,
)
from .prime_multi_run_router import merge_routed_batch_groups
from .prime_rl_bridge import RolloutDecision
from .safety_supervisor import append_hash_chained_record

ASYNC_QUEUE_VERSION = "arena-atomic-async-training-queue-v1"


@dataclass(frozen=True)
class AdmittedAsyncGroup:
    rollout_id: str
    trainer_step: int
    batches: dict[str, Any]
    admission: AsyncAdmissionResult


class AtomicAsyncTrainingQueue:
    """Keep complete four-policy groups together from rescore through routing."""

    def __init__(
        self,
        *,
        capacity: int,
        audit_path: Path,
        allowed_backend_calibrations: frozenset[tuple[str, str, str, str]],
        allowed_constraint_sha256s: frozenset[str],
        limits: AsyncAdmissionLimits,
    ) -> None:
        if capacity < 1:
            raise ValueError("async queue capacity must be positive")
        self.capacity = capacity
        self.audit_path = audit_path
        self.allowed_backend_calibrations = allowed_backend_calibrations
        self.allowed_constraint_sha256s = allowed_constraint_sha256s
        self.limits = limits
        self._groups: deque[AdmittedAsyncGroup] = deque()
        self._seen_rollout_ids: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._groups)

    def admit(
        self,
        *,
        header: AsyncRolloutHeader,
        decisions: tuple[RolloutDecision, ...],
        trainable_decision_ids: frozenset[str],
        trainable_branch: str = "actual",
        current_snapshots: tuple[PolicySnapshot, ...],
        current_policy_logprobs: dict[str, tuple[float, ...]],
        routed_batches: dict[str, Any],
        trainer_step: int,
    ) -> AsyncAdmissionResult:
        if trainer_step < 0:
            raise ValueError("async trainer step cannot be negative")
        if header.rollout_id in self._seen_rollout_ids:
            raise ValueError(f"duplicate async rollout ID: {header.rollout_id}")
        if len(self._groups) >= self.capacity:
            raise BufferError("async queue capacity reached before admission")
        if sum(snapshot.trainable for snapshot in header.policy_snapshots) != 4:
            raise ValueError("async Swarm group requires four trainable policy snapshots")
        if len(routed_batches) != 4 or set(routed_batches) != {
            f"run_blue_{index}" for index in range(4)
        }:
            raise ValueError("async group must contain all four isolated policy batches")
        if any(batch.step != trainer_step for batch in routed_batches.values()):
            raise ValueError("async routed batches disagree on trainer step")

        result = admit_async_rollout(
            header,
            decisions,
            current_snapshots,
            current_policy_logprobs,
            trainable_decision_ids=trainable_decision_ids,
            trainable_branch=trainable_branch,
            allowed_backend_calibrations=self.allowed_backend_calibrations,
            allowed_constraint_sha256s=self.allowed_constraint_sha256s,
            limits=self.limits,
        )
        self._seen_rollout_ids.add(header.rollout_id)
        append_hash_chained_record(
            self.audit_path,
            {
                "version": ASYNC_QUEUE_VERSION,
                "rollout_header": asdict(header),
                "trainer_step": trainer_step,
                "trainable_decision_ids": sorted(trainable_decision_ids),
                "trainable_branch": trainable_branch,
                "accepted": result.accepted,
                "reasons": result.reasons,
                "metrics": result.metrics,
            },
        )
        if not result.accepted:
            return result
        self._groups.append(
            AdmittedAsyncGroup(
                rollout_id=header.rollout_id,
                trainer_step=trainer_step,
                batches=routed_batches,
                admission=result,
            )
        )
        return result

    def pop_logical_update(self, *, groups: int, trainer_step: int) -> dict[str, Any]:
        if groups < 1:
            raise ValueError("logical update requires at least one group")
        if len(self._groups) < groups:
            raise LookupError("not enough admitted groups for a logical update")
        selected = tuple(self._groups[index] for index in range(groups))
        if any(group.trainer_step != trainer_step for group in selected):
            raise ValueError("queue head mixes trainer steps; refusing partial reorder")
        for _ in range(groups):
            self._groups.popleft()
        return merge_routed_batch_groups(
            tuple(group.batches for group in selected),
            step=trainer_step,
        )
