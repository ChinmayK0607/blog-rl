from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.swarm_arena.scripts.run_v13_ordinary_signal_screen import (
    common_command,
    file_sha256,
    parse_opponent,
    parse_opponent_adapter,
)


def test_parse_opponent_requires_known_family_and_three_fields() -> None:
    assert parse_opponent("base:model:revision") == ("base", "model", "revision")
    try:
        parse_opponent("unknown:model:revision")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("unknown opponent families must fail closed")


def test_parse_opponent_adapter_binds_path_and_hash(tmp_path: Path) -> None:
    digest = "a" * 64
    family, path, sha256 = parse_opponent_adapter(
        f"historical:{tmp_path}:{digest}"
    )
    assert family == "historical"
    assert path == tmp_path.resolve()
    assert sha256 == digest


def test_common_command_is_rollout_only_and_focused() -> None:
    args = argparse.Namespace(
        trainer_config="trainer.toml",
        inference_config="inference.toml",
        data_dir="data",
        tokenizer="tokenizer",
        initial_adapter="sft",
        initial_policy_adapter_manifest="policies.json",
        base_revision="base-revision",
        initial_policy_revision="policy-revision",
        base_url=["http://127.0.0.1:8001/v1"],
    )
    command = common_command(args, "source")
    assert "--rollout-only" in command
    assert command[command.index("--shared-return-credit-assignment") + 1] == "focused_agent"
    assert command[command.index("--shared-return-trainable-phase") + 1] == "ACT"


def test_file_sha256_changes_with_bound_runtime_file(tmp_path: Path) -> None:
    path = tmp_path / "trainer.toml"
    path.write_text(json.dumps({"learning_rate": 1e-6}), encoding="utf-8")
    original = file_sha256(path)
    path.write_text(json.dumps({"learning_rate": 2e-6}), encoding="utf-8")
    assert file_sha256(path) != original
