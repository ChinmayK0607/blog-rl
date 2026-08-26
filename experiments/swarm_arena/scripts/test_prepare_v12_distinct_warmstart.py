from __future__ import annotations

import hashlib
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("prepare_v12_distinct_warmstart.py")
spec = importlib.util.spec_from_file_location("v12_warmstart", MODULE)
warmstart = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(warmstart)


def make_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    hashes = {}
    files = {}
    root = tmp_path / "adapters"
    for index in range(4):
        policy = f"blue-{index}"
        directory = root / policy
        directory.mkdir(parents=True)
        payload = f"adapter-{index}".encode()
        (directory / "adapter_model.safetensors").write_bytes(payload)
        hashes[policy] = hashlib.sha256(payload).hexdigest()
        files[f"{policy}/adapter_model.safetensors"] = hashes[policy]
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "ready": {"step": 180, "policy_adapter_sha256": hashes},
                "files_sha256": files,
            }
        )
    )
    return manifest, root, hashes


def test_prepare_binds_four_distinct_hashes(tmp_path: Path) -> None:
    manifest, root, hashes = make_fixture(tmp_path)
    base = Path(__file__).resolve().parents[1] / "configs" / "rl_v12_4b_robust_communication_160.toml"
    trainer = tmp_path / "trainer.toml"
    controller = tmp_path / "warmstart.json"
    result = warmstart.prepare(base, manifest, root, trainer, controller)
    with trainer.open("rb") as handle:
        config = tomllib.load(handle)
    assert set(config["model"]["lora"]["initial_adapter_paths_by_run"]) == {
        f"run_blue_{index}" for index in range(4)
    }
    assert {
        policy: row["sha256"] for policy, row in result["policies"].items()
    } == hashes


def test_prepare_rejects_one_corrupt_policy(tmp_path: Path) -> None:
    manifest, root, _ = make_fixture(tmp_path)
    (root / "blue-2" / "adapter_model.safetensors").write_bytes(b"corrupt")
    base = Path(__file__).resolve().parents[1] / "configs" / "rl_v12_4b_robust_communication_160.toml"
    with pytest.raises(ValueError, match="blue-2"):
        warmstart.prepare(base, manifest, root, tmp_path / "trainer.toml", tmp_path / "warmstart.json")
