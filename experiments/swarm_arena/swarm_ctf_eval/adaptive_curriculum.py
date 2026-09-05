from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from typing import Any, Iterable, Mapping, Sequence

from .rl_production import (
    AdaptiveCurriculumConfig,
    OpponentSnapshot,
    OrdinaryCase,
    ScenarioAssignment,
)

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
    ordinary_cases: dict[str, dict[str, Any]] = {}
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
            case_id = scenario.get("ordinary_case_id")
            if case_id is not None:
                case = ordinary_cases.setdefault(
                    str(case_id),
                    {
                        "case_id": str(case_id),
                        "focused_agent": focused,
                        "opponent_family": str(scenario["opponent"]["family"]),
                        "seed": int(scenario["seed"]),
                        "size": size,
                        "horizon": horizon,
                        "groups": 0,
                        "replicas": 0,
                        "return_min": min(returns),
                        "return_max": max(returns),
                        "nonzero_advantage": 0,
                    },
                )
                if case["focused_agent"] != focused or case["opponent_family"] != str(
                    scenario["opponent"]["family"]
                ):
                    raise ValueError("ordinary case identity changed policy/opponent binding")
                case["groups"] += 1
                case["replicas"] += len(returns)
                case["return_min"] = min(float(case["return_min"]), *returns)
                case["return_max"] = max(float(case["return_max"]), *returns)
                case["nonzero_advantage"] += sum(
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
    for row in ordinary_cases.values():
        replicas = int(row["replicas"])
        row["nonzero_advantage_rate"] = row.pop("nonzero_advantage") / replicas
        if replicas < config.minimum_replicas:
            row["classification"] = "frontier"
        elif (
            float(row["return_max"]) - float(row["return_min"])
            > config.positive_epsilon
            and row["nonzero_advantage_rate"] > 0
        ):
            row["classification"] = "frontier"
        elif float(row["return_min"]) > config.positive_epsilon:
            row["classification"] = "mastered"
        else:
            row["classification"] = "stalled"

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
        "ordinary_cases": dict(sorted(ordinary_cases.items())),
        "classification_counts": {
            "handoff": dict(
                sorted(Counter(row["classification"] for row in handoff.values()).items())
            ),
            "ordinary": dict(
                sorted(Counter(row["classification"] for row in ordinary.values()).items())
            ),
            "ordinary_cases": dict(
                sorted(
                    Counter(
                        row["classification"] for row in ordinary_cases.values()
                    ).items()
                )
            ),
        },
    }
    return {**body, "sha256": canonical_sha256(body)}


def select_ordinary_stage_cases(
    schedule: Sequence[ScenarioAssignment],
    *,
    stage_name: str,
    opponent_schedule: Sequence[OpponentSnapshot],
    pool: Sequence[OrdinaryCase],
    analysis: Mapping[str, Any],
    config: AdaptiveCurriculumConfig,
    selection_namespace: str,
) -> tuple[dict[int, OrdinaryCase], dict[str, Any]]:
    """Select a frontier-heavy ordinary case for each exact future stage slot."""
    config.validate()
    ordinary = [
        row for row in schedule if row.stage == stage_name and row.kind == "ordinary"
    ]
    if not ordinary:
        raise ValueError(f"adaptive ordinary stage has no ordinary slots: {stage_name}")
    if len(opponent_schedule) != len(schedule):
        raise ValueError("ordinary selector requires an opponent for every schedule slot")
    categories = _category_sequence(len(ordinary), config)
    stats = analysis.get("ordinary_cases", {})
    by_key: dict[tuple[str, str], list[OrdinaryCase]] = defaultdict(list)
    for case in pool:
        case.validate()
        by_key[(case.focused_agent, case.opponent_family)].append(case)
    policy_modes = _policy_modes(config)
    routed_frontier_slots: Counter[str] = Counter()
    has_frontier: dict[str, bool] = {}
    for policy in (f"blue-{index}" for index in range(4)):
        policy_cases = [case for case in pool if case.focused_agent == policy]
        has_frontier[policy] = any(
            stats.get(case.case_id, {}).get(
                "classification", case.initial_classification
            )
            == "frontier"
            for case in policy_cases
        )
    for assignment, requested in zip(ordinary, categories, strict=True):
        if requested == "frontier":
            routed_frontier_slots[f"blue-{assignment.ordinal % 4}"] += 1
    route_sequences = {
        policy: _ordinary_route_sequence(
            slots,
            mode=policy_modes.get(policy, "consolidate"),
            has_frontier=has_frontier[policy],
            config=config,
        )
        for policy, slots in routed_frontier_slots.items()
    }
    route_cursors: Counter[str] = Counter()
    selected: dict[int, OrdinaryCase] = {}
    cursors: Counter[tuple[str, str, str]] = Counter()
    used: dict[tuple[str, str], set[str]] = defaultdict(set)
    category_counts: Counter[str] = Counter()
    requested_counts: Counter[str] = Counter()
    routed_counts: Counter[str] = Counter()
    for assignment, requested in zip(ordinary, categories, strict=True):
        policy = f"blue-{assignment.ordinal % 4}"
        family = opponent_schedule[assignment.ordinal].family
        key = (policy, family)
        cases = sorted(
            by_key.get(key, []),
            key=lambda case: hashlib.sha256(
                f"{config.selection_seed}:{selection_namespace}:{case.case_id}".encode()
            ).hexdigest(),
        )
        if not cases:
            raise ValueError(f"ordinary case pool lacks {policy}/{family}")
        by_category: dict[str, list[OrdinaryCase]] = defaultdict(list)
        for case in cases:
            category = str(
                stats.get(case.case_id, {}).get(
                    "classification", case.initial_classification
                )
            )
            by_category[category].append(case)
        requested_counts[requested] += 1
        routed = requested
        if requested == "frontier":
            routed = route_sequences[policy][route_cursors[policy]]
            route_cursors[policy] += 1
        routed_counts[f"{policy}/{routed}"] += 1
        candidates: list[OrdinaryCase] = []
        selected_category = routed
        preference = _ordinary_category_preference(
            requested=requested,
            routed=routed,
            mode=policy_modes.get(policy, "consolidate"),
        )
        for category in preference:
            candidates = [
                case for case in by_category.get(category, []) if case.case_id not in used[key]
            ]
            if not candidates and by_category.get(category):
                candidates = by_category[category]
            if candidates:
                selected_category = category
                break
        if not candidates:
            raise ValueError(f"ordinary case pool has no selectable {policy}/{family} case")
        cursor_key = (policy, family, selected_category)
        case = candidates[cursors[cursor_key] % len(candidates)]
        cursors[cursor_key] += 1
        used[key].add(case.case_id)
        selected[assignment.ordinal] = case
        category_counts[selected_category] += 1
    body = {
        "version": "arena-rl-adaptive-ordinary-selection-v1",
        "stage": stage_name,
        "analysis_sha256": analysis.get("sha256"),
        "config": asdict(config),
        "selected_cases": {
            str(ordinal): asdict(case) for ordinal, case in sorted(selected.items())
        },
        "requested_category_counts": dict(sorted(requested_counts.items())),
        "routed_category_counts": dict(sorted(routed_counts.items())),
        "selected_category_counts": dict(sorted(category_counts.items())),
        "policy_modes": dict(sorted(policy_modes.items())),
        "frozen_or_development_data_used": False,
    }
    body = json.loads(json.dumps(body, sort_keys=True))
    return selected, {**body, "sha256": canonical_sha256(body)}


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


def _policy_modes(config: AdaptiveCurriculumConfig) -> dict[str, str]:
    return {
        policy: mode
        for policy, mode in (value.split(":") for value in config.policy_modes)
    }


def _ordinary_route_sequence(
    slots: int,
    *,
    mode: str,
    has_frontier: bool,
    config: AdaptiveCurriculumConfig,
) -> list[str]:
    if slots < 0:
        raise ValueError("ordinary routed slot count cannot be negative")
    if mode == "consolidate":
        frontier_fraction = 1.0
    elif mode == "expand":
        frontier_fraction = config.expand_frontier_fraction
    elif mode == "discover":
        frontier_fraction = (
            config.discovery_frontier_fraction if has_frontier else 0.0
        )
    else:
        raise ValueError(f"unknown adaptive ordinary policy mode: {mode}")
    frontier = round(slots * frontier_fraction)
    remaining = {"frontier": frontier, "unseen": slots - frontier}
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


def _ordinary_category_preference(
    *, requested: str, routed: str, mode: str
) -> tuple[str, ...]:
    if requested != "frontier":
        order = (requested, "frontier", "unseen", "mastered", "stalled")
    elif mode == "discover":
        order = (routed, "unseen", "frontier", "stalled", "mastered")
    else:
        order = (routed, "frontier", "unseen", "mastered", "stalled")
    return tuple(dict.fromkeys(order))


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
