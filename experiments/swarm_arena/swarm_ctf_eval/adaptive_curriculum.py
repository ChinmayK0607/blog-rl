from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from typing import Any, Iterable, Mapping, Sequence

from .rl_production import AdaptiveCurriculumConfig, ScenarioAssignment

ANALYSIS_VERSION = "arena-rl-training-frontier-analysis-v1"
SELECTION_VERSION = "arena-rl-adaptive-stage-selection-v1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def handoff_case_key(kind: str, pair_index: int, world: str) -> str:
    if kind not in {"critical", "decoy"}:
        raise ValueError(f"unknown handoff kind: {kind}")
    return f"{kind}:{pair_index}:{world}"


def _classify(*, positive: int, total: int, config: AdaptiveCurriculumConfig) -> str:
    if total < config.minimum_replicas:
        return "frontier"
    rate = positive / total
    if rate >= config.mastered_pass_rate:
        return "mastered"
    if rate <= config.stalled_pass_rate:
        return "stalled"
    return "frontier"


def summarize_training_progress(
    progress: Sequence[Mapping[str, Any]],
    *,
    config: AdaptiveCurriculumConfig,
    step_start: int | None = None,
    step_end: int | None = None,
) -> dict[str, Any]:
    """Summarize training-only rollout signal without consulting eval data.

    One curriculum group is the independent scheduling unit. Replica outcomes
    are retained for pass-rate diagnostics but are not reported as independent
    map observations.
    """
    config.validate()
    handoff: dict[str, dict[str, Any]] = {}
    ordinary: dict[str, dict[str, Any]] = {}
    observed_steps: list[int] = []
    for update in progress:
        step = int(update["step"])
        if step_start is not None and step < step_start:
            continue
        if step_end is not None and step >= step_end:
            continue
        observed_steps.append(step)
        for group in update.get("groups", []):
            scenario = group["scenario"]
            replicas = list(group.get("replicas", []))
            kind = scenario.get("kind")
            if kind in {"critical", "decoy"}:
                pair_index = int(scenario["pair_index"])
                world = str(scenario["world"])
                key = handoff_case_key(str(kind), pair_index, world)
                field = "semantic_effect" if kind == "critical" else "challenge_effect"
                effects = [float(replica[field]) for replica in replicas]
                focused = str(scenario["focused_agent"])
                advantages = [
                    float(replica.get("advantages", {}).get(focused, 0.0))
                    for replica in replicas
                ]
                row = handoff.setdefault(
                    key,
                    {
                        "kind": str(kind),
                        "pair_index": pair_index,
                        "world": world,
                        "receiver": str(scenario["receiver"]),
                        "groups": 0,
                        "replicas": 0,
                        "positive": 0,
                        "zero": 0,
                        "negative": 0,
                        "effect_sum": 0.0,
                        "nonzero_advantage": 0,
                        "target_action": 0,
                    },
                )
                row["groups"] += 1
                row["replicas"] += len(effects)
                row["positive"] += sum(value > config.positive_epsilon for value in effects)
                row["zero"] += sum(abs(value) <= config.positive_epsilon for value in effects)
                row["negative"] += sum(value < -config.positive_epsilon for value in effects)
                row["effect_sum"] += sum(effects)
                row["nonzero_advantage"] += sum(
                    abs(value) > config.positive_epsilon for value in advantages
                )
                if kind == "critical":
                    active_target = scenario.get("active_target")
                    row["target_action"] += sum(
                        replica.get("focused_action", {}).get("target") == active_target
                        for replica in replicas
                    )
                continue

            if scenario.get("source") != "ordinary":
                raise ValueError("training progress contains an unknown scenario kind")
            size = int(scenario["size"])
            horizon = int(scenario["scheduled_horizon"])
            key = f"ordinary:{size}:{horizon}"
            returns = [float(replica["return"]) for replica in replicas]
            focused = str(scenario["focused_agent"])
            advantages = [
                float(replica.get("advantages", {}).get(focused, 0.0))
                for replica in replicas
            ]
            row = ordinary.setdefault(
                key,
                {
                    "size": size,
                    "horizon": horizon,
                    "groups": 0,
                    "replicas": 0,
                    "positive": 0,
                    "zero": 0,
                    "negative": 0,
                    "return_sum": 0.0,
                    "nonzero_advantage": 0,
                },
            )
            row["groups"] += 1
            row["replicas"] += len(returns)
            row["positive"] += sum(value > config.positive_epsilon for value in returns)
            row["zero"] += sum(abs(value) <= config.positive_epsilon for value in returns)
            row["negative"] += sum(value < -config.positive_epsilon for value in returns)
            row["return_sum"] += sum(returns)
            row["nonzero_advantage"] += sum(
                abs(value) > config.positive_epsilon for value in advantages
            )

    for row in handoff.values():
        replicas = int(row["replicas"])
        row["pass_rate"] = row["positive"] / replicas
        row["mean_effect"] = row.pop("effect_sum") / replicas
        row["nonzero_advantage_rate"] = row["nonzero_advantage"] / replicas
        row["target_action_rate"] = (
            row.pop("target_action") / replicas if row["kind"] == "critical" else None
        )
        row["classification"] = _classify(
            positive=int(row["positive"]), total=replicas, config=config
        )
    for row in ordinary.values():
        replicas = int(row["replicas"])
        row["positive_return_rate"] = row["positive"] / replicas
        row["mean_return"] = row.pop("return_sum") / replicas
        row["nonzero_advantage_rate"] = row["nonzero_advantage"] / replicas
        row["classification"] = _classify(
            positive=int(row["positive"]), total=replicas, config=config
        )

    body = {
        "version": ANALYSIS_VERSION,
        "scope": "training_rollouts_only_no_development_or_frozen_data",
        "step_range": (
            None
            if not observed_steps
            else {"start_inclusive": min(observed_steps), "end_exclusive": max(observed_steps) + 1}
        ),
        "updates": len(set(observed_steps)),
        "config": asdict(config),
        "handoff_cases": dict(sorted(handoff.items())),
        "ordinary_buckets": dict(sorted(ordinary.items())),
        "classification_counts": {
            "handoff": dict(
                sorted(Counter(row["classification"] for row in handoff.values()).items())
            ),
            "ordinary": dict(
                sorted(Counter(row["classification"] for row in ordinary.values()).items())
            ),
        },
    }
    return {**body, "sha256": canonical_sha256(body)}


def _category_sequence(slots: int, config: AdaptiveCurriculumConfig) -> list[str]:
    mastered = round(slots * config.mastered_anchor_fraction)
    stalled = round(slots * config.stalled_anchor_fraction)
    frontier = slots - mastered - stalled
    if frontier < 0:
        raise ValueError("adaptive allocation overfilled the requested slots")
    remaining = {"frontier": frontier, "mastered": mastered, "stalled": stalled}
    sequence: list[str] = []
    previous = None
    while sum(remaining.values()):
        available = [name for name, count in remaining.items() if count]
        candidates = [name for name in available if name != previous] or available
        selected = sorted(candidates, key=lambda name: (-remaining[name], name))[0]
        sequence.append(selected)
        remaining[selected] -= 1
        previous = selected
    return sequence


def _stable_order(values: Iterable[tuple[int, str]], *, seed: str) -> list[tuple[int, str]]:
    return sorted(
        set(values),
        key=lambda value: hashlib.sha256(f"{seed}:{value[0]}:{value[1]}".encode()).hexdigest(),
    )


def select_handoff_cases(
    *,
    kind: str,
    receiver_sequence: Sequence[str],
    pool_by_receiver: Mapping[str, Sequence[tuple[int, str]]],
    analysis: Mapping[str, Any],
    config: AdaptiveCurriculumConfig,
    selection_namespace: str,
) -> tuple[tuple[int, str], ...]:
    """Select a balanced frontier-heavy sequence with small fixed anchors."""
    config.validate()
    categories = _category_sequence(len(receiver_sequence), config)
    cursors: Counter[tuple[str, str]] = Counter()
    used: dict[str, set[tuple[int, str]]] = defaultdict(set)
    selected: list[tuple[int, str]] = []
    stats = analysis.get("handoff_cases", {})
    for slot, (receiver, requested_category) in enumerate(
        zip(receiver_sequence, categories, strict=True)
    ):
        pool = _stable_order(
            pool_by_receiver[receiver],
            seed=f"{config.selection_seed}:{selection_namespace}:{receiver}",
        )
        by_category: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for case in pool:
            key = handoff_case_key(kind, case[0], case[1])
            category = stats.get(key, {}).get("classification", "frontier")
            by_category[str(category)].append(case)
        fallback = [requested_category, "frontier", "mastered", "stalled"]
        candidates = [
            case
            for case in by_category.get(requested_category, [])
            if case not in used[receiver]
        ]
        category = requested_category
        if not candidates and by_category.get(requested_category):
            candidates = by_category[requested_category]
        if not candidates:
            for candidate_category in fallback[1:]:
                fresh = [
                    case
                    for case in by_category.get(candidate_category, [])
                    if case not in used[receiver]
                ]
                if fresh:
                    candidates = fresh
                    category = candidate_category
                    break
        if not candidates:
            for candidate_category in fallback[1:]:
                if by_category.get(candidate_category):
                    candidates = by_category[candidate_category]
                    category = candidate_category
                    break
        if not candidates:
            raise ValueError(f"no adaptive {kind} cases exist for {receiver}")
        cursor_key = (receiver, category)
        case = candidates[cursors[cursor_key] % len(candidates)]
        cursors[cursor_key] += 1
        used[receiver].add(case)
        selected.append(case)
    return tuple(selected)


def adapt_stage_assignments(
    schedule: Sequence[ScenarioAssignment],
    *,
    stage_name: str,
    analysis: Mapping[str, Any],
    receiver_by_case: Mapping[tuple[int, str], str],
    config: AdaptiveCurriculumConfig,
    candidate_pool_by_receiver: Mapping[str, Sequence[tuple[int, str]]] | None = None,
) -> tuple[tuple[ScenarioAssignment, ...], dict[str, Any]]:
    """Replace one future stage's handoff cases while preserving its exact shape."""
    config.validate()
    result = list(schedule)
    stage_indices = [index for index, row in enumerate(schedule) if row.stage == stage_name]
    if not stage_indices:
        raise ValueError(f"adaptive stage does not exist: {stage_name}")
    if stage_indices != list(range(min(stage_indices), max(stage_indices) + 1)):
        raise ValueError("adaptive stage assignments must be contiguous")
    pool_by_receiver: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if candidate_pool_by_receiver is not None:
        for receiver, cases in candidate_pool_by_receiver.items():
            pool_by_receiver[receiver].extend(cases)
    else:
        for row in schedule:
            if row.pair_index is None or row.handoff_world is None:
                continue
            case = (row.pair_index, row.handoff_world)
            receiver = receiver_by_case[case]
            if case not in pool_by_receiver[receiver]:
                pool_by_receiver[receiver].append(case)

    matched_rows: list[tuple[int, int]] = []
    extra_critical_rows: list[int] = []
    for offset in range(min(stage_indices), max(stage_indices) + 1, 4):
        block_indices = list(range(offset, offset + 4))
        critical = [index for index in block_indices if schedule[index].kind == "critical"]
        decoy = [index for index in block_indices if schedule[index].kind == "decoy"]
        unmatched = set(critical)
        for decoy_index in decoy:
            decoy_case = (
                schedule[decoy_index].pair_index,
                schedule[decoy_index].handoff_world,
            )
            critical_index = next(
                index
                for index in critical
                if (
                    schedule[index].pair_index,
                    schedule[index].handoff_world,
                )
                == decoy_case
            )
            matched_rows.append((critical_index, decoy_index))
            unmatched.remove(critical_index)
        extra_critical_rows.extend(sorted(unmatched))

    decoy_receivers = [
        receiver_by_case[(schedule[index].pair_index, schedule[index].handoff_world)]
        for index, _ in matched_rows
    ]
    decoy_cases = select_handoff_cases(
        kind="decoy",
        receiver_sequence=decoy_receivers,
        pool_by_receiver=pool_by_receiver,
        analysis=analysis,
        config=config,
        selection_namespace=f"{stage_name}:decoy",
    )
    for (critical_index, decoy_index), case in zip(matched_rows, decoy_cases, strict=True):
        result[critical_index] = replace(
            result[critical_index], pair_index=case[0], handoff_world=case[1]
        )
        result[decoy_index] = replace(
            result[decoy_index], pair_index=case[0], handoff_world=case[1]
        )

    critical_receivers = [
        receiver_by_case[(schedule[index].pair_index, schedule[index].handoff_world)]
        for index in extra_critical_rows
    ]
    critical_cases = select_handoff_cases(
        kind="critical",
        receiver_sequence=critical_receivers,
        pool_by_receiver=pool_by_receiver,
        analysis=analysis,
        config=config,
        selection_namespace=f"{stage_name}:critical",
    )
    for index, case in zip(extra_critical_rows, critical_cases, strict=True):
        result[index] = replace(result[index], pair_index=case[0], handoff_world=case[1])

    selected_rows = [asdict(result[index]) for index in stage_indices]
    body = {
        "version": SELECTION_VERSION,
        "stage": stage_name,
        "analysis_sha256": analysis["sha256"],
        "config": asdict(config),
        "selected_assignments": selected_rows,
        "category_counts": {
            "critical": dict(
                sorted(
                    Counter(
                        analysis.get("handoff_cases", {})
                        .get(handoff_case_key("critical", case[0], case[1]), {})
                        .get("classification", "frontier")
                        for case in critical_cases
                    ).items()
                )
            ),
            "decoy": dict(
                sorted(
                    Counter(
                        analysis.get("handoff_cases", {})
                        .get(handoff_case_key("decoy", case[0], case[1]), {})
                        .get("classification", "frontier")
                        for case in decoy_cases
                    ).items()
                )
            ),
        },
        "ordinary_schedule_changed": False,
        "frozen_or_development_data_used": False,
    }
    body = json.loads(json.dumps(body, sort_keys=True))
    return tuple(result), {**body, "sha256": canonical_sha256(body)}
