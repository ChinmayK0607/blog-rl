"""Exactly enumerate a baseline-independence counterexample; no model training."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from swarm_ctf_eval.safety_supervisor import paired_terminal_contrast_advantages


def audit(p: float = 0.7, q: float = 0.4, *, coupled: bool = True, centering: str = "none") -> dict:
    """Bernoulli factual action; fixed counterfactual policy, reward equals action."""
    if not 0 < p < 1 or not 0 < q < 1:
        raise ValueError("Bernoulli probabilities must be strictly between zero and one")
    if coupled:
        boundaries = sorted({0.0, p, q, 1.0})
        cells = [
            (right - left, float((left + right) / 2 < p), float((left + right) / 2 < q))
            for left, right in zip(boundaries, boundaries[1:])
        ]
    else:
        cells = [(pa * pb, a, b) for a, pa in ((0.0, 1 - p), (1.0, p)) for b, pb in ((0.0, 1 - q), (1.0, q))]
    expected = 0.0
    for first, second in itertools.product(cells, repeat=2):
        weight = first[0] * second[0]
        factual = (first[1], second[1])
        counterfactual = (first[2], second[2])
        advantages = paired_terminal_contrast_advantages(factual, counterfactual, centering=centering)
        expected += weight * sum((a - p) * advantage for a, advantage in zip(factual, advantages, strict=True)) / 2
    exact_gradient = p * (1 - p)  # d E[R_factual - R_fixed_counterfactual] / d factual logit.
    return {
        "version": "paired-credit-enumerable-audit-v1",
        "factual_probability": p,
        "fixed_counterfactual_probability": q,
        "shared_randomness": coupled,
        "centering": centering,
        "expected_implemented_score_update": expected,
        "exact_objective_logit_gradient": exact_gradient,
        "bias": expected - exact_gradient,
        "scope": "counterexample to automatic unbiased-baseline claims; not a diagnosis of the full multi-turn estimator",
        "production_estimator_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        audit(coupled=coupled, centering=centering)
        for coupled in (False, True)
        for centering in ("none", "replica_mean")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
