from __future__ import annotations

import json

import pytest

from scripts.publish_warmstart import publish


def test_publisher_refuses_unpromoted_checkpoint(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"decision": "base_model", "selected_step": None}),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"ids_sha256": "test"}), encoding="utf-8")
    config = tmp_path / "training.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="did not pass promotion"):
        publish(
            "CK0607/test",
            adapter,
            selection,
            audit,
            config,
            source_commit="deadbeef",
            training_run_url="https://example.invalid/run",
        )
