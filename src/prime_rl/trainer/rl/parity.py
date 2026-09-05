from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def rollout_parity_failures(
    metrics: dict[str, float], config: Any
) -> dict[str, tuple[float, float]]:
    """Return every enabled parity threshold exceeded by one logical batch."""
    thresholds = {
        "mean_logprob_error": config.max_mean_logprob_error,
        "p99_logprob_error": config.max_p99_logprob_error,
        "max_probability_error": config.max_probability_error,
        "p99_probability_error": config.max_p99_probability_error,
        "probability_tail_fraction": config.max_probability_tail_fraction,
        "mean_mismatch_kl": config.max_mean_mismatch_kl,
        "max_mismatch_kl": config.max_mismatch_kl,
    }
    return {
        name: (metrics[name], threshold)
        for name, threshold in thresholds.items()
        if threshold is not None and metrics[name] > threshold
    }


def rollout_parity_quarantine_disposition(
    *,
    logical_update: int,
    prior_window_count: int,
    window_size: int,
    window_limit: int,
) -> tuple[int, int, bool]:
    """Return ``(window, next_count, allowed)`` for a failed logical update."""
    if logical_update < 1:
        raise ValueError("logical update must be positive")
    if prior_window_count < 0:
        raise ValueError("prior quarantine count cannot be negative")
    if window_size < 1 or window_limit < 1 or window_limit > window_size:
        raise ValueError("invalid parity quarantine window")
    window_index = (logical_update - 1) // window_size
    next_count = prior_window_count + 1
    return window_index, next_count, next_count <= window_limit


def load_rollout_parity_quarantine_counts(
    path: Path, *, window_size: int
) -> dict[int, int]:
    """Restore bounded quarantine counts without making a restart permissive."""
    if not path.is_file():
        return {}
    updates: set[int] = set()
    counts: dict[int, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("version") != "prime-rl-parity-quarantine-v1":
            raise ValueError(f"unknown parity quarantine record at line {line_number}")
        if row.get("action") == "abort_quarantine_limit_exceeded":
            raise RuntimeError("cannot resume a run after its parity quarantine limit failed")
        if row.get("action") != "quarantine_logical_update":
            raise ValueError(f"unknown parity quarantine action at line {line_number}")
        if row.get("replacement_batch_sampled") is not False:
            raise ValueError("parity quarantine record permits replacement sampling")
        if row.get("optimizer_step_applied") is not False:
            raise ValueError("parity quarantine record falsely claims an optimizer step")
        logical_update = int(row["logical_update"])
        if logical_update < 1 or logical_update in updates:
            raise ValueError("parity quarantine updates must be positive and unique")
        updates.add(logical_update)
        window_index = (logical_update - 1) // window_size
        if row.get("window_index") != window_index or row.get("window_size") != window_size:
            raise ValueError("parity quarantine record uses a different window contract")
        counts[window_index] = counts.get(window_index, 0) + 1
        if row.get("window_quarantine_count") != counts[window_index]:
            raise ValueError("parity quarantine count is not append-only and contiguous")
    return counts
