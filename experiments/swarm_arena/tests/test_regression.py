from __future__ import annotations

import json

from swarm_ctf_eval.regression import (
    FROZEN_REGRESSION_CASES,
    REGRESSION_MANIFEST_SHA256,
    summarize_regression_rows,
    validate_response,
)
from swarm_ctf_eval.regression_compare import compare
from swarm_ctf_eval.regression_v2 import (
    FROZEN_REGRESSION_V2_CASES,
    REGRESSION_V2_MANIFEST_SHA256,
    validate_v2_response,
)
from swarm_ctf_eval.episode import ArenaEpisodeEnv
from swarm_ctf_eval.episode_protocol import episode_broadcast_prompt
from swarm_ctf_eval.warm_start_selection import select_warm_start
from swarm_ctf_eval.warm_start_selection_v3 import select_warm_start_v3
from swarm_ctf_eval.warmstart_v3 import (
    generate_arena_rows,
    generate_preservation_rows,
    validate_warmstart_response,
)
from swarm_ctf_eval.warmstart_v5 import generate_arena_rows as generate_v5_arena_rows
from swarm_ctf_eval.warmstart_v8 import generate_sequential_rows, validate_sequential_response


def test_frozen_regression_manifest_is_balanced_and_unique() -> None:
    assert len(FROZEN_REGRESSION_CASES) == 256
    assert len({row.id for row in FROZEN_REGRESSION_CASES}) == 256
    counts = {}
    for row in FROZEN_REGRESSION_CASES:
        counts[row.category] = counts.get(row.category, 0) + 1
    assert set(counts.values()) == {64}
    assert len(REGRESSION_MANIFEST_SHA256) == 64


def test_response_validation_detects_arena_leakage() -> None:
    case = FROZEN_REGRESSION_CASES[0]
    assert validate_response(case, '{"action_id":"A0"}') == {
        "valid_json": True,
        "exact": False,
        "arena_leakage": True,
    }
    assert validate_response(case, "not json")["valid_json"] is False
    assert validate_response(case, '{"answer":{"facts":[]}}')["arena_leakage"] is True


def test_summary_and_comparison_are_paired() -> None:
    rows = [
        {
            "id": case.id,
            "category": case.category,
            "valid_json": True,
            "exact": True,
            "arena_leakage": False,
        }
        for case in FROZEN_REGRESSION_CASES
    ]
    summary = summarize_regression_rows(rows)
    assert summary["exact"] == 1.0
    result = compare(rows, [dict(row) for row in rows])
    assert result["gates"]["passed"]


def test_comparison_rejects_category_regression() -> None:
    base = []
    adapter = []
    for case in FROZEN_REGRESSION_CASES:
        row = {
            "id": case.id,
            "category": case.category,
            "valid_json": True,
            "exact": True,
            "arena_leakage": False,
        }
        base.append(row)
        adapter.append(dict(row, exact=case.category != "arithmetic"))
    result = compare(base, adapter)
    assert not result["gates"]["every_category_drop_within_0_05"]
    assert not result["gates"]["passed"]


def test_warm_start_selection_prefers_safe_checkpoint(tmp_path) -> None:
    base_rows = []
    for case in FROZEN_REGRESSION_CASES:
        base_rows.append(
            {
                "id": case.id,
                "category": case.category,
                "valid_json": True,
                "exact": True,
                "arena_leakage": False,
            }
        )
    base_path = tmp_path / "base.jsonl"
    base_path.write_text("".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8")
    validation_root = tmp_path / "validation"
    regression_root = tmp_path / "regression"
    for step, score, safe in ((40, 0.7, True), (80, 0.9, False)):
        validation_dir = validation_root / f"step_{step}" / "validation"
        validation_dir.mkdir(parents=True)
        validation_dir.joinpath("summary.json").write_text(
            json.dumps(
                {
                    "schema_valid": 1.0,
                    "selection_score": score,
                    "broadcast": {"supported": 1.0, "exact": score},
                    "act": {"legal": 1.0, "exact": score},
                }
            ),
            encoding="utf-8",
        )
        regression_dir = regression_root / f"step_{step}"
        regression_dir.mkdir(parents=True)
        adapter_rows = [dict(row) for row in base_rows]
        if not safe:
            for row in adapter_rows:
                if row["category"] == "instruction_binding":
                    row["exact"] = False
        regression_dir.joinpath("rows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in adapter_rows),
            encoding="utf-8",
        )
    result = select_warm_start(validation_root, regression_root, base_path)
    assert result["decision"] == "adapter"
    assert result["selected_step"] == 40


def test_regression_v2_is_frozen_balanced_and_strict() -> None:
    assert len(FROZEN_REGRESSION_V2_CASES) == 256
    assert len({case.id for case in FROZEN_REGRESSION_V2_CASES}) == 256
    counts = {}
    for case in FROZEN_REGRESSION_V2_CASES:
        counts[case.category] = counts.get(case.category, 0) + 1
        assert validate_v2_response(case, json.dumps(case.expected))["exact"]
    assert set(counts.values()) == {64}
    assert len(REGRESSION_V2_MANIFEST_SHA256) == 64


def test_warmstart_v3_balances_phases_and_preservation() -> None:
    arena = generate_arena_rows(7_200_000, 2, "test")
    assert len(arena) == 16
    assert sum(json.loads(row["metadata_json"])["phase"] == "BROADCAST" for row in arena) == 8
    assert sum(json.loads(row["metadata_json"])["phase"] == "ACT" for row in arena) == 8
    preservation = generate_preservation_rows(20260920, 12, "test")
    assert len(preservation) == 12
    assert len({row["id"] for row in preservation}) == 12
    for row in [*arena, *preservation]:
        result = validate_warmstart_response(row, row["messages"][-1]["content"])
        assert result == {"schema_valid": True, "grounded": True, "legal": True, "exact": True}
    broadcast = next(
        row
        for row in arena
        if json.loads(row["metadata_json"])["phase"] == "BROADCAST"
        and json.loads(row["messages"][-1]["content"])["facts"]
    )
    unsupported = json.loads(broadcast["messages"][-1]["content"])
    unsupported["facts"][0]["owner"] = "RED" if unsupported["facts"][0]["owner"] != "RED" else "BLUE"
    result = validate_warmstart_response(broadcast, json.dumps(unsupported))
    assert result["schema_valid"]
    assert not result["grounded"]
    action = next(row for row in arena if json.loads(row["metadata_json"])["phase"] == "ACT")
    result = validate_warmstart_response(action, '{"action_id":"A999"}')
    assert result["schema_valid"]
    assert not result["legal"]


def test_warmstart_v5_broadcast_labels_are_prompt_identifiable() -> None:
    arena = generate_v5_arena_rows(7_300_000, 2, "test")
    broadcasts = [row for row in arena if json.loads(row["metadata_json"])["phase"] == "BROADCAST"]
    assert len(broadcasts) == 8
    for row in broadcasts:
        metadata = json.loads(row["metadata_json"])
        assert metadata["label_source"] == "canonical_prompt_identifiable_local_broadcast"
        assert "message_mode" not in metadata
        assert validate_warmstart_response(row, row["messages"][-1]["content"]) == {
            "schema_valid": True,
            "grounded": True,
            "legal": True,
            "exact": True,
        }


def test_episode_protocol_exposes_exact_legal_intent_shapes() -> None:
    env = ArenaEpisodeEnv(seed=73, size=13)
    env.reset()
    messages, actions = episode_broadcast_prompt(env, "blue-0")
    body = json.loads(messages[-1]["content"])
    assert body["legal_intents"] == [action.to_dict() for action in actions]
    assert "copy exactly one object from `legal_intents`" in messages[0]["content"]
    assert "`unknown_neighbors` contains identifiers only" in messages[0]["content"]


def test_warm_start_v3_requires_both_regression_suites(tmp_path) -> None:
    base_rows = [
        {"id": f"case-{index}", "category": f"category-{index}", "exact": True, "arena_leakage": False}
        for index in range(4)
    ]
    base_v1 = tmp_path / "base_v1.jsonl"
    base_v2 = tmp_path / "base_v2.jsonl"
    payload = "".join(json.dumps(row) + "\n" for row in base_rows)
    base_v1.write_text(payload, encoding="utf-8")
    base_v2.write_text(payload, encoding="utf-8")
    validation_root = tmp_path / "validation"
    regression_v1_root = tmp_path / "regression_v1"
    regression_v2_root = tmp_path / "regression_v2"
    for step in (8, 16):
        validation = validation_root / f"step_{step}"
        v1 = regression_v1_root / f"step_{step}"
        v2 = regression_v2_root / f"step_{step}"
        validation.mkdir(parents=True)
        v1.mkdir(parents=True)
        v2.mkdir(parents=True)
        validation.joinpath("summary.json").write_text(
            json.dumps(
                {
                    "schema_valid": 1.0,
                    "groups": {
                        "BROADCAST": {"grounded": 1.0, "legal": 1.0, "exact": step / 20},
                        "ACT": {"legal": 1.0, "exact": step / 20},
                    },
                }
            ),
            encoding="utf-8",
        )
        v1.joinpath("rows.jsonl").write_text(payload, encoding="utf-8")
        v2_rows = base_rows if step == 8 else [dict(row, exact=False) for row in base_rows]
        v2.joinpath("rows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in v2_rows),
            encoding="utf-8",
        )
    result = select_warm_start_v3(
        validation_root,
        regression_v1_root,
        regression_v2_root,
        base_v1,
        base_v2,
    )
    assert result["decision"] == "adapter"
    assert result["selected_step"] == 8
    assert result["selection_protocol"] == "swarm-warm-start-v4-raw-legality"
    assert [candidate["step"] for candidate in result["candidates"]] == [8, 16]


def test_warmstart_v8_matches_sequential_game_contexts() -> None:
    rows = generate_sequential_rows(7_600_000, 2, "test")
    metadata = [json.loads(row["metadata_json"]) for row in rows]
    assert {item["team"] for item in metadata} == {"BLUE", "RED"}
    assert {item["turn"] for item in metadata} == set(range(6))
    assert {item["history_length"] for item in metadata} == {0, 1, 2, 3}
    broadcasts = [
        (row, item) for row, item in zip(rows, metadata, strict=True) if item["phase"] == "BROADCAST"
    ]
    assert any(item["budget_before"] == 0 and item["target_empty"] for _, item in broadcasts)
    for row in rows:
        assert validate_sequential_response(row, row["messages"][-1]["content"]) == {
            "schema_valid": True,
            "grounded": True,
            "legal": True,
            "exact": True,
        }


def test_warm_start_selection_rejects_illegal_raw_broadcasts(tmp_path) -> None:
    base_rows = [
        {"id": "case", "category": "category", "exact": True, "arena_leakage": False}
    ]
    base_v1 = tmp_path / "base-v1.jsonl"
    base_v2 = tmp_path / "base-v2.jsonl"
    payload = json.dumps(base_rows[0]) + "\n"
    base_v1.write_text(payload, encoding="utf-8")
    base_v2.write_text(payload, encoding="utf-8")
    validation = tmp_path / "validation" / "step_64"
    regression_v1 = tmp_path / "regression-v1" / "step_64"
    regression_v2 = tmp_path / "regression-v2" / "step_64"
    for path in (validation, regression_v1, regression_v2):
        path.mkdir(parents=True)
    validation.joinpath("summary.json").write_text(
        json.dumps(
            {
                "schema_valid": 1.0,
                "groups": {
                    "BROADCAST": {"grounded": 1.0, "legal": 0.98, "exact": 0.8},
                    "ACT": {"legal": 1.0, "exact": 0.8},
                },
            }
        ),
        encoding="utf-8",
    )
    regression_v1.joinpath("rows.jsonl").write_text(payload, encoding="utf-8")
    regression_v2.joinpath("rows.jsonl").write_text(payload, encoding="utf-8")
    result = select_warm_start_v3(
        tmp_path / "validation",
        tmp_path / "regression-v1",
        tmp_path / "regression-v2",
        base_v1,
        base_v2,
    )
    assert result["decision"] == "base_model"
    assert not result["candidates"][0]["protocol_gates"]["broadcast_legal"]
