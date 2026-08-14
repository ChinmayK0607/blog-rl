from __future__ import annotations

from scripts.measure_constrained_policy_kl import _summary


def test_policy_kl_summary_preserves_tail_metrics() -> None:
    rows = [
        {
            "allowed_token_count": 2,
            "candidate_to_baseline_kl": value,
            "baseline_to_candidate_kl": value / 2,
            "total_variation": value / 4,
        }
        for value in (0.01, 0.02, 0.03, 0.50)
    ]
    report = _summary(rows)
    assert report["tokens"] == 4
    assert report["branching_tokens"] == 4
    assert report["candidate_to_baseline_kl"]["mean"] == 0.14
    assert report["candidate_to_baseline_kl"]["p99"] == 0.50
    assert report["baseline_to_candidate_kl"]["max"] == 0.25
