from __future__ import annotations

import json

import pytest
from scripts.run_final_eval_development import _prepare_output, _summary
from swarm_ctf_eval.progress_eval_v4 import summarize_progress_eval


def _row(
    *,
    case_id: str,
    suite: str,
    opponent: str,
    variant: str,
    revision: str,
    condition: str,
    terminal_return: float,
) -> dict:
    return {
        "case_id": case_id,
        "suite": suite,
        "opponent_id": opponent,
        "opponent_revision": f"{opponent}-revision",
        "side": "BLUE",
        "sampling_key": f"{case_id}:{opponent}:BLUE",
        "policy_variant": variant,
        "policy_revision": revision,
        "condition": condition,
        "terminal_return": terminal_return,
        "broadcast_protocol_rate": 1.0,
        "broadcast_grounded_rate": 1.0,
        "action_protocol_rate": 1.0,
    }


def test_development_summary_pairs_capability_communication_and_decoys() -> None:
    rows = []
    for opponent in ("base", "sft", "historical"):
        rows.extend(
            (
                _row(
                    case_id="ordinary-1",
                    suite="ordinary_ood",
                    opponent=opponent,
                    variant="candidate_rl",
                    revision="candidate",
                    condition="normal",
                    terminal_return=0.3,
                ),
                _row(
                    case_id="ordinary-1",
                    suite="ordinary_ood",
                    opponent=opponent,
                    variant="sft_init",
                    revision="sft",
                    condition="normal",
                    terminal_return=0.1,
                ),
            )
        )
        for condition in ("normal", "dropped", "sender_shuffled", "delayed", "zero_budget"):
            rows.append(
                _row(
                    case_id="critical-1",
                    suite="critical",
                    opponent=opponent,
                    variant="candidate_rl",
                    revision="candidate",
                    condition=condition,
                    terminal_return=0.4 if condition == "normal" else 0.3,
                )
            )
        for condition, value in (("normal", 0.2), ("dropped", 0.15)):
            rows.append(
                _row(
                    case_id="critical-1",
                    suite="critical",
                    opponent=opponent,
                    variant="sft_init",
                    revision="sft",
                    condition=condition,
                    terminal_return=value,
                )
            )
        for condition in ("normal", "dropped"):
            rows.append(
                _row(
                    case_id="decoy-1",
                    suite="decoy",
                    opponent=opponent,
                    variant="candidate_rl",
                    revision="candidate",
                    condition=condition,
                    terminal_return=0.1,
                )
            )

    summary = _summary(rows)
    assert summary["ordinary_candidate_minus_sft"]["mean_difference"] == pytest.approx(0.2)
    assert summary["critical_normal_minus_intervention"]["dropped"]["mean_difference"] == pytest.approx(
        0.1
    )
    assert summary["critical_sft_normal_minus_dropped"]["mean_difference"] == pytest.approx(0.05)
    assert summary["decoy_candidate_normal_minus_dropped"]["mean_difference"] == 0.0
    assert set(summary["candidate_normal_return_by_opponent"]) == {"base", "sft", "historical"}


def test_development_manifest_resume_canonicalizes_tuples(tmp_path) -> None:
    manifest = {"cases": ((1, 12, 4),), "version": "test"}
    assert _prepare_output(tmp_path, manifest, False) == set()
    (tmp_path / "rows.jsonl").write_text(
        json.dumps({"evaluation_id": "already-done"}) + "\n",
        encoding="utf-8",
    )
    assert _prepare_output(tmp_path, manifest, True) == {"already-done"}


def test_progress_eval_separates_hard_capability_handoffs_and_decoys() -> None:
    rows = []
    common = {
        "opponent_id": "sft",
        "opponent_revision": "sft-revision",
        "side": "BLUE",
        "sampling_key": "sample",
        "messages_nonempty": 1,
        "broadcast_protocol_rate": 1.0,
        "broadcast_grounded_rate": 1.0,
        "action_protocol_rate": 1.0,
    }
    for suite in ("ordinary_legacy", "ordinary_hard"):
        for variant, value in (("candidate_rl", 0.3), ("sft_init", 0.1)):
            rows.append(
                {
                    **common,
                    "independent_id": f"{suite}-seed",
                    "case_id": f"{suite}-case",
                    "suite": suite,
                    "policy_variant": variant,
                    "policy_revision": variant,
                    "condition": "normal",
                    "terminal_return": value,
                }
            )
    for condition in ("normal", "dropped", "sender_shuffled", "delayed", "zero_budget"):
        rows.append(
            {
                **common,
                "independent_id": "handoff-pair",
                "case_id": "handoff-left",
                "suite": "handoff_critical",
                "policy_variant": "candidate_rl",
                "policy_revision": "candidate",
                "condition": condition,
                "terminal_return": 0.4 if condition == "normal" else 0.2,
            }
        )
    for condition in ("normal", "dropped"):
        rows.append(
            {
                **common,
                "independent_id": "decoy-pair",
                "case_id": "decoy-left",
                "suite": "handoff_decoy",
                "policy_variant": "candidate_rl",
                "policy_revision": "candidate",
                "condition": condition,
                "terminal_return": 0.1,
            }
        )
    summary = summarize_progress_eval(rows)
    assert summary["capability_rl_minus_sft"]["ordinary_hard"][
        "mean_difference"
    ] == pytest.approx(0.2)
    assert summary["communication_effects"]["normal_minus_dropped"][
        "mean_difference"
    ] == pytest.approx(0.2)
    assert summary["matched_decoy_normal_minus_dropped"]["mean_difference"] == 0
    assert summary["claim_checks"]["communication_claim_passed"]

    online_rows = [
        row
        for row in rows
        if row["condition"] in {"normal", "dropped"}
    ]
    online = summarize_progress_eval(
        online_rows,
        intervention_conditions=("dropped",),
    )
    assert set(online["communication_effects"]) == {"normal_minus_dropped"}
    assert not online["claim_checks"]["full_intervention_matrix"]
    assert not online["claim_checks"]["communication_claim_passed"]
