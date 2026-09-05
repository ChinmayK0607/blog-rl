from __future__ import annotations

from pathlib import Path
from typing import Any

from .safety_supervisor import verify_hash_chain

SHARED_RETURN_PARITY_PROBE_VERSION = "arena-shared-return-parity-probe-v1"
SHARED_RETURN_DISTRIBUTION_PROBE_VERSION = "arena-shared-return-parity-probe-v2"


def build_shared_return_parity_probe(evidence_path: Path) -> dict[str, Any]:
    """Extract only run-lock-selected trainable spans from verified evidence."""
    records = verify_hash_chain(evidence_path)
    if not records:
        raise ValueError("shared-return evidence chain is empty")
    samples = []
    evidence_records = []
    seen_decisions: set[str] = set()
    for record in records:
        payload = record["payload"]
        spec = payload.get("spec")
        initial_state = payload.get("initial_state")
        replicas = payload.get("replicas")
        if not isinstance(spec, dict) or not isinstance(initial_state, dict):
            raise ValueError("evidence record lacks a shared-return spec or initial state")
        if not isinstance(replicas, list) or len(replicas) != int(spec["replicas"]):
            raise ValueError("evidence record has an invalid shared-return replica set")
        phases = set(spec["trainable_phases"])
        offsets = spec["trainable_turn_offsets"]
        turns = (
            None
            if offsets is None
            else {
                int(initial_state["turn"]) + int(offset)
                for offset in offsets
            }
        )
        evidence_records.append(
            {
                "record_sha256": record["record_sha256"],
                "group_id": payload["group_id"],
                "initial_state_sha256": payload["initial_state_sha256"],
            }
        )
        for replica in sorted(replicas, key=lambda row: int(row["replica_index"])):
            selected = [
                decision
                for decision in replica["decisions"]
                if decision["team"] == "BLUE"
                and decision["phase"] in phases
                and (turns is None or int(decision["turn"]) in turns)
            ]
            selected.sort(key=lambda row: int(row["trajectory_index"]))
            if not selected:
                raise ValueError("shared-return replica has no trainable parity spans")
            for decision in selected:
                decision_id = (
                    f"{decision['game_id']}:actual:{decision['agent_id']}:"
                    f"{decision['turn']}:{decision['phase']}"
                )
                if decision_id in seen_decisions:
                    raise ValueError(f"duplicate parity decision: {decision_id}")
                seen_decisions.add(decision_id)
                policy_id = str(decision["policy_id"])
                prefix = "blue-policy-"
                if not policy_id.startswith(prefix):
                    raise ValueError(f"trainable parity span has an unknown policy: {policy_id}")
                policy_slot = int(policy_id.removeprefix(prefix))
                if policy_slot not in range(4):
                    raise ValueError(f"trainable parity span has an invalid slot: {policy_id}")
                completion_ids = list(decision["completion_ids"])
                allowed_rows = [list(row) for row in decision["allowed_token_ids"]]
                rollout_logprobs = list(decision["rollout_logprobs"])
                if not (
                    len(completion_ids) == len(allowed_rows) == len(rollout_logprobs)
                ):
                    raise ValueError(f"incomplete parity token rows: {decision_id}")
                if any(
                    token not in allowed
                    for token, allowed in zip(
                        completion_ids, allowed_rows, strict=True
                    )
                ):
                    raise ValueError(f"parity completion violates its constraint: {decision_id}")
                sample = {
                    "decision_id": decision_id,
                    "game_id": decision["game_id"],
                    "replica_index": replica["replica_index"],
                    "agent_id": decision["agent_id"],
                    "policy_id": policy_id,
                    "policy_slot": policy_slot,
                    "phase": decision["phase"],
                    "turn": decision["turn"],
                    "prompt_ids": list(decision["prompt_ids"]),
                    "completion_ids": completion_ids,
                    "completion_logprobs": rollout_logprobs,
                    "allowed_token_ids": allowed_rows,
                }
                serving_rows = decision.get("serving_allowed_logprobs")
                if serving_rows:
                    if len(serving_rows) != len(completion_ids):
                        raise ValueError(
                            f"serving distribution rows are incomplete: {decision_id}"
                        )
                    sample["serving_allowed_logprobs"] = serving_rows
                samples.append(sample)
    has_distributions = ["serving_allowed_logprobs" in row for row in samples]
    if any(has_distributions) and not all(has_distributions):
        raise ValueError("parity probe mixes decisions with and without serving distributions")
    return {
        "version": (
            SHARED_RETURN_DISTRIBUTION_PROBE_VERSION
            if all(has_distributions)
            else SHARED_RETURN_PARITY_PROBE_VERSION
        ),
        "source_evidence": evidence_records,
        "samples": samples,
    }
