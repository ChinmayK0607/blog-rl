"""Prospective pulse identities and resource admission; never rewrite old rows."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .progress_eval_v4 import _bootstrap

EVALUATION_IDENTITY_VERSION = "actual-initializer-v1"


def required_independent_units(
    *, bundle_sd: float, worthwhile_effect: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Normal-approximation planning only; units are bundles, not episode rows."""
    if not all(math.isfinite(value) for value in (bundle_sd, worthwhile_effect, power, alpha)):
        raise ValueError("power inputs must be finite")
    if bundle_sd <= 0 or worthwhile_effect <= 0 or not 0.5 < power < 1 or not 0 < alpha < 0.5:
        raise ValueError("invalid variance, effect, power, or alpha")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return max(2, math.ceil((z * bundle_sd / worthwhile_effect) ** 2))


def branch_return_summary(rows: list[dict]) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["suite"] not in {"handoff_critical", "handoff_decoy"}:
            continue
        name = "/".join((row["policy_variant"], row["suite"], row["condition"]))
        groups[name][row["independent_id"]].append(float(row["terminal_return"]))
    result = {}
    for name, units in sorted(groups.items()):
        values = [statistics.mean(group) for _, group in sorted(units.items())]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite branch return")
        result[name] = {
            "independent_units": len(values),
            "mean": statistics.mean(values),
            "mean_95": _bootstrap(values),
        }
    return result


def pulse_config(*, base_urls: list[str], ready: dict, snapshots: dict, baseline_revision: str) -> dict:
    sft = snapshots["sft"]
    if baseline_revision != sft.revision:
        raise ValueError("SFT baseline revision must match the immutable SFT opponent snapshot")
    if set(ready["policy_adapter_sha256"]) != {f"blue-{i}" for i in range(4)}:
        raise ValueError("pulse requires all four ready policy adapter hashes")
    return {
        "evaluation_identity_version": EVALUATION_IDENTITY_VERSION,
        "purpose": "actual_initializer" if ready["step"] == 0 else "trained_checkpoint",
        "base_urls": [url.rstrip("/") + "/v1" for url in base_urls],
        "candidate": {
            "revision": ready["policy_revision"],
            "models": [f"blue-{i}" for i in range(4)],
            "adapter_sha256": ready["policy_adapter_sha256"],
        },
        "baseline": {
            "revision": sft.revision,
            "models": [sft.model_name] * 4,
            "adapter_sha256": {sft.model_name: sft.adapter_sha256},
        },
        "opponents": [
            {
                "id": identifier,
                "revision": snapshots[family].revision,
                "models": [snapshots[family].model_name] * 4,
                "adapter_sha256": {snapshots[family].model_name: snapshots[family].adapter_sha256},
            }
            for identifier, family in (("base", "base"), ("sft", "sft"), ("historical_league", "historical"))
        ],
    }


def verify_served_adapters(config: dict, registries: dict[str, dict]) -> dict:
    """Hash registered adapter roots on the same host as the serving processes."""
    expected = {}
    for arm in (config["candidate"], config["baseline"], *config["opponents"]):
        for alias, digest in arm["adapter_sha256"].items():
            if digest is None:  # Base weights are pinned by the runtime certificate.
                continue
            if alias in expected and expected[alias] != digest:
                raise ValueError(f"conflicting adapter identity for {alias}")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid adapter hash for {alias}")
            expected[alias] = digest
    if set(registries) != set(config["base_urls"]):
        raise ValueError("registry evidence must cover every configured server")
    evidence = {}
    for url, registry in registries.items():
        evidence[url] = {}
        for alias, digest in expected.items():
            matches = [row for row in registry["data"] if row.get("id") == alias]
            if len(matches) != 1 or not matches[0].get("root"):
                raise ValueError(f"missing or ambiguous registered alias: {alias}")
            root = Path(matches[0]["root"]).resolve(strict=True)
            weight = root / "adapter_model.safetensors"
            with weight.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"served adapter hash mismatch for {alias} on {url}")
            evidence[url][alias] = {"root": str(root), "adapter_sha256": actual}
    return evidence


def initializer_improvement(current: list[dict], initial: list[dict]) -> dict:
    """Matched checkpoint-minus-initializer effects, bootstrapped by bundle."""
    fields = ("suite", "case_id", "independent_id", "opponent_id", "opponent_revision", "side", "condition")

    def index(rows):
        indexed = {}
        for row in rows:
            if row["policy_variant"] != "candidate_rl":
                continue
            key = tuple(row[field] for field in fields)
            if key in indexed:
                raise ValueError("duplicate initializer comparison cell")
            value = float(row["terminal_return"])
            if not math.isfinite(value):
                raise ValueError("nonfinite initializer comparison return")
            indexed[key] = value
        return indexed

    left, right = index(current), index(initial)
    if not left or left.keys() != right.keys():
        raise ValueError("initializer comparison requires exactly matched cells")
    groups = defaultdict(lambda: defaultdict(list))
    for key in sorted(left):
        suite, _, unit, _, _, _, condition = key
        groups[f"{suite}/{condition}"][unit].append((left[key], right[key]))
    endpoints = {}
    for name, units in groups.items():
        paired = [
            (statistics.mean(a for a, _ in cells), statistics.mean(b for _, b in cells))
            for _, cells in sorted(units.items())
        ]
        effects = [a - b for a, b in paired]
        endpoints[name] = {
            "independent_units": len(effects),
            "checkpoint_mean": statistics.mean(a for a, _ in paired),
            "initializer_mean": statistics.mean(b for _, b in paired),
            "mean_difference": statistics.mean(effects),
            "mean_difference_95": _bootstrap(effects),
        }
    communication = {}
    for suite in ("handoff_critical", "handoff_decoy"):
        normal = groups.get(f"{suite}/normal")
        if normal is None:
            continue
        for intervention in ("dropped", "target_swapped"):
            altered = groups.get(f"{suite}/{intervention}")
            if altered is None:
                continue
            if normal.keys() != altered.keys():
                raise ValueError("communication contrast requires matched independent bundles")
            effects = [
                statistics.mean(a - b for a, b in normal[unit]) - statistics.mean(a - b for a, b in altered[unit])
                for unit in sorted(normal)
            ]
            communication[f"{suite}/normal_minus_{intervention}"] = {
                "independent_units": len(effects),
                "mean_difference": statistics.mean(effects),
                "mean_difference_95": _bootstrap(effects),
            }
    return {
        "definition": "checkpoint minus actual step-zero initializer; not checkpoint minus SFT",
        "scope": "paired development diagnostics, not a confirmatory claim or a replacement gate",
        "paired_cells": len(left),
        "return_changes": endpoints,
        "communication_effect_changes": communication,
    }


def staged_evaluation_budget(
    *,
    updates: int,
    interval: int,
    games_per_minute: float,
    update_seconds: float,
    available_seconds: float,
    setup_seconds: float = 0,
    final_sync_seconds: float = 2700,
    safety_factor: float = 1.25,
    checkpoint_seconds: float = 600,
) -> dict:
    values = (
        games_per_minute,
        update_seconds,
        available_seconds,
        setup_seconds,
        final_sync_seconds,
        safety_factor,
        checkpoint_seconds,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("budget inputs must be finite")
    if updates < 1 or interval < 1 or updates % interval:
        raise ValueError("positive pulse interval must divide updates")
    if games_per_minute <= 0 or update_seconds <= 0 or available_seconds <= 0:
        raise ValueError("throughput, update duration, and available time must be positive")
    if min(setup_seconds, final_sync_seconds, checkpoint_seconds) < 0 or safety_factor < 1:
        raise ValueError("reserves must be nonnegative and safety factor at least one")
    fresh_games = 192 + 120 * (updates // interval)
    evaluation_seconds = fresh_games * 60 / games_per_minute * safety_factor
    training_seconds = updates * update_seconds * safety_factor
    checkpoint_total = (updates // interval + 1) * checkpoint_seconds
    required = setup_seconds + evaluation_seconds + training_seconds + checkpoint_total + final_sync_seconds
    barrier = math.ceil(192 * 60 / games_per_minute * safety_factor + checkpoint_seconds)
    return {
        "version": "staged-evaluation-budget-v1",
        "fresh_games": fresh_games,
        "cached_sft_rows_per_later_pulse": 72,
        "evaluation_seconds": evaluation_seconds,
        "training_seconds": training_seconds,
        "checkpoint_seconds": checkpoint_total,
        "setup_seconds": setup_seconds,
        "final_sync_seconds": final_sync_seconds,
        "safety_factor": safety_factor,
        "required_seconds": required,
        "available_seconds": available_seconds,
        "fits": required <= available_seconds,
        "checkpoint_barrier_timeout_seconds": barrier,
        "pulse_wait_timeout_seconds": math.ceil(required - setup_seconds - final_sync_seconds),
    }
