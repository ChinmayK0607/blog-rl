from __future__ import annotations

import copy

import pytest

from swarm_ctf_eval.arena_compare import compare
from swarm_ctf_eval.arena_eval import OracleArenaModel, evaluate_case, summarize
from swarm_ctf_eval.arena_generation import generate_state
from swarm_ctf_eval.arena_oracle import deterministic_policy
from swarm_ctf_eval.arena_sft import oracle_broadcast


def oracle_row(seed: int) -> dict:
    state = generate_state(seed)
    reference = deterministic_policy(state, "BLUE")
    shuffled = {agent: oracle_broadcast(state, agent, action) for agent, action in reference.items()}
    return evaluate_case(OracleArenaModel(), seed, 12, "balanced", shuffled)


def test_paired_comparison_requires_identical_cases() -> None:
    row = oracle_row(101)
    with pytest.raises(ValueError, match="identical frozen cases"):
        compare([row], [], summarize([row]), summarize([row]), trials=10)


def test_paired_comparison_reports_zero_effect_for_identical_runs() -> None:
    rows = [oracle_row(101), oracle_row(102)]
    summary = summarize(rows)
    result = compare(rows, copy.deepcopy(rows), summary, copy.deepcopy(summary), trials=100)
    assert result["primary_endpoint"]["mean_difference"] == 0.0
    assert result["primary_endpoint"]["randomization_p_two_sided"] == 1.0
    assert not result["claim_gates"]["coordination_improvement_supported"]
