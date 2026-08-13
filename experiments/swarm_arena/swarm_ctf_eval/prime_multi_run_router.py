from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .safety_supervisor import Approval, verify_approval_signature

ROUTER_VERSION = "arena-prime-multi-run-router-v1"


@dataclass(frozen=True)
class PolicyRunRoute:
    policy_id: str
    run_id: str

    def validate(self) -> None:
        if not self.policy_id or not self.run_id.startswith("run_"):
            raise ValueError("policy routes require a policy ID and a fixed run_* directory")
        if "/" in self.run_id or self.run_id in {"run_", "run_.."}:
            raise ValueError("run IDs cannot contain paths")


@dataclass(frozen=True)
class OwnedAgentSamples:
    game_id: str
    agent_id: str
    policy_id: str
    decision_ids: tuple[str, ...]
    samples: tuple[Any, ...]


def route_approved_samples(
    approval: Approval,
    owned_samples: tuple[OwnedAgentSamples, ...],
    routes: tuple[PolicyRunRoute, ...],
    *,
    step: int,
    signing_key: bytes,
    trainer_parity_gate_sha256: str | None = None,
) -> dict[str, Any]:
    """Convert one signed approval into four isolated Prime-RL run batches."""
    from prime_rl.transport import TrainingBatch

    verify_approval_signature(approval, signing_key)
    if approval.parity_mode == "trainer_pre_step":
        if (
            trainer_parity_gate_sha256 is None
            or trainer_parity_gate_sha256 != approval.trainer_parity_gate_sha256
        ):
            raise ValueError(
                "deferred parity approval does not match the active trainer pre-step gate"
            )
    if step < 0:
        raise ValueError("trainer step cannot be negative")
    for route in routes:
        route.validate()
    route_by_policy = {route.policy_id: route.run_id for route in routes}
    if len(route_by_policy) != 4 or len(set(route_by_policy.values())) != 4:
        raise ValueError("four policies must map one-to-one onto four run directories")

    envelope_by_agent = {row.agent_id: row for row in approval.envelopes}
    sample_by_agent = {row.agent_id: row for row in owned_samples}
    if len(envelope_by_agent) != 4 or set(sample_by_agent) != set(envelope_by_agent):
        raise ValueError("owned samples must cover exactly the four approved agents")
    if {row.policy_id for row in approval.envelopes} != set(route_by_policy):
        raise ValueError("approved policies do not match the immutable run routing table")

    result = {}
    seen_decisions = set()
    for agent_id, envelope in sorted(envelope_by_agent.items()):
        owned = sample_by_agent[agent_id]
        if (owned.game_id, owned.policy_id) != (approval.game_id, envelope.policy_id):
            raise ValueError(f"sample ownership mismatch for {agent_id}")
        if owned.decision_ids != envelope.decision_ids:
            raise ValueError(f"decision-span mismatch for {agent_id}")
        if seen_decisions & set(owned.decision_ids):
            raise ValueError("a decision span was assigned to multiple policies")
        seen_decisions.update(owned.decision_ids)
        if not owned.samples:
            raise ValueError(f"no Prime-RL samples for approved agent {agent_id}")
        completion_tokens = sum(
            sum(bool(value) for value in sample.completion_mask)
            for sample in owned.samples
        )
        if completion_tokens != envelope.completion_tokens:
            raise ValueError(
                f"trainable completion-token count mismatch for {agent_id}: "
                f"expected {envelope.completion_tokens}, got {completion_tokens}"
            )

        examples = []
        for source in owned.samples:
            if source.advantage is not None or source.reward is not None:
                raise ValueError("untrusted rollout samples cannot pre-populate reward or advantage")
            if source.training_mode != "rl":
                raise ValueError("Swarm Arena approval can route only RL samples")
            sample = copy.deepcopy(source)
            sample.advantage = envelope.advantage
            sample.reward = approval.replay_return
            examples.append(sample)
        run_id = route_by_policy[envelope.policy_id]
        result[run_id] = TrainingBatch(examples=examples, step=step)
    return result


def merge_routed_batch_groups(
    groups: tuple[dict[str, Any], ...],
    *,
    step: int,
) -> dict[str, Any]:
    """Merge independently approved games into one atomic batch per policy run."""
    from prime_rl.transport import TrainingBatch

    if not groups:
        raise ValueError("at least one approved group is required")
    expected = set(groups[0])
    if len(expected) != 4 or any(set(group) != expected for group in groups):
        raise ValueError("every approved group must cover the same four policy runs")
    merged = {}
    for run_id in sorted(expected):
        examples = []
        for group in groups:
            batch = group[run_id]
            if batch.step != step:
                raise ValueError("cannot merge approved groups from different steps")
            examples.extend(batch.examples)
        merged[run_id] = TrainingBatch(examples=examples, step=step)
    return merged


async def send_approved_batches(
    trainer_output_dir: Path,
    batches: dict[str, Any],
) -> None:
    from prime_rl.transport.filesystem import FileSystemTrainingBatchSender

    expected = {path.name for path in trainer_output_dir.glob("run_*") if path.is_dir()}
    if not set(batches) <= expected:
        raise ValueError("trainer run directories must exist before approved queue admission")
    await asyncio.gather(
        *(
            FileSystemTrainingBatchSender(trainer_output_dir / run_id).send(batch)
            for run_id, batch in sorted(batches.items())
        )
    )
