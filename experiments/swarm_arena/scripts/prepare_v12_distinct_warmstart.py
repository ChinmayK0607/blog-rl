from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import tomli_w


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(
    base_trainer_config: Path,
    v11_checkpoint_manifest: Path,
    adapter_root: Path,
    trainer_output: Path,
    controller_manifest_output: Path,
) -> dict[str, object]:
    manifest = json.loads(v11_checkpoint_manifest.read_text())
    ready = manifest.get("ready")
    if not isinstance(ready, dict) or int(ready.get("step", -1)) != 180:
        raise ValueError("warm start must use the public V11 update-180 ready record")
    hashes = ready.get("policy_adapter_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {
        f"blue-{index}" for index in range(4)
    }:
        raise ValueError("V11 manifest must contain exactly four policy adapter hashes")
    declared_files = manifest.get("files_sha256", {})
    paths = {}
    controller = {
        "version": "swarm-distinct-policy-warmstart-v1",
        "source": "public-v11-update180",
        "source_manifest_sha256": sha256(v11_checkpoint_manifest),
        "policies": {},
    }
    for index in range(4):
        policy_id = f"blue-{index}"
        run_id = f"run_blue_{index}"
        path = (adapter_root / policy_id).resolve()
        adapter_file = path / "adapter_model.safetensors"
        actual = sha256(adapter_file)
        expected = str(hashes[policy_id])
        if actual != expected:
            raise ValueError(f"local V11 adapter hash mismatch for {policy_id}")
        matching_manifest_hashes = {
            value
            for name, value in declared_files.items()
            if policy_id in name and name.endswith("adapter_model.safetensors")
        }
        if matching_manifest_hashes and matching_manifest_hashes != {expected}:
            raise ValueError(f"public file manifest disagrees for {policy_id}")
        paths[run_id] = path
        controller["policies"][policy_id] = {
            "path": str(path),
            "sha256": expected,
            "revision": expected,
        }

    with base_trainer_config.open("rb") as handle:
        trainer = tomllib.load(handle)
    lora = trainer["model"]["lora"]
    lora["initial_adapter_paths_by_run"] = {
        run_id: str(path) for run_id, path in sorted(paths.items())
    }
    lora["initial_adapter_sha256_by_run"] = {
        run_id: controller["policies"][run_id.replace("run_blue_", "blue-")]["sha256"]
        for run_id in sorted(paths)
    }
    trainer_output.parent.mkdir(parents=True, exist_ok=True)
    with trainer_output.open("wb") as handle:
        tomli_w.dump(trainer, handle)
    controller_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    controller_manifest_output.write_text(
        json.dumps(controller, indent=2, sort_keys=True) + "\n"
    )
    return controller


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind a V12 trainer/controller to four distinct public V11 adapters."
    )
    parser.add_argument("--base-trainer-config", type=Path, required=True)
    parser.add_argument("--v11-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--trainer-output", type=Path, required=True)
    parser.add_argument("--controller-manifest-output", type=Path, required=True)
    args = parser.parse_args()
    controller = prepare(
        args.base_trainer_config,
        args.v11_checkpoint_manifest,
        args.adapter_root,
        args.trainer_output,
        args.controller_manifest_output,
    )
    print(json.dumps(controller, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
