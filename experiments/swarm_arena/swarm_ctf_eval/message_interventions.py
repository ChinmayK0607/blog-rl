from __future__ import annotations

from dataclasses import replace

from .arena_protocol import Broadcast


def target_swapped_broadcast(
    broadcast: Broadcast,
    *,
    candidate_targets: tuple[str, str],
    active_target: str,
) -> Broadcast:
    """Swap only the certified candidate identities in a grounded message."""
    if len(set(candidate_targets)) != 2 or active_target not in candidate_targets:
        raise ValueError("target-swap intervention requires two targets and one active target")
    candidate_facts = [fact for fact in broadcast.facts if fact.node in candidate_targets]
    if not candidate_facts or all(fact.node != active_target for fact in candidate_facts):
        raise ValueError("target-swap sender must broadcast the active candidate fact")
    other_target = next(target for target in candidate_targets if target != active_target)
    target_mapping = {active_target: other_target, other_target: active_target}
    facts = tuple(
        replace(fact, node=target_mapping[fact.node])
        if fact.node in target_mapping
        else fact
        for fact in broadcast.facts
    )
    intent = broadcast.intent
    if intent is not None and intent.target in candidate_targets:
        intent = replace(intent, target=target_mapping[intent.target])
    return Broadcast(facts, intent, broadcast.request_resource)
