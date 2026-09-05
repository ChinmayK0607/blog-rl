from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from pathlib import Path

from safetensors import safe_open


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_argument(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("adapters must use NAME=PATH")
    return name, Path(path)


def _tensor_rank(path: Path) -> int:
    ranks: set[int] = set()
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = handle.get_slice(key).get_shape()
            if ".lora_A." in key:
                ranks.add(int(shape[0]))
            elif ".lora_B." in key:
                ranks.add(int(shape[1]))
    if len(ranks) != 1:
        raise ValueError(f"adapter tensors do not have one LoRA rank: {sorted(ranks)}")
    return ranks.pop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create metadata-corrected, weight-identical views of v10 policy adapters."
    )
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--adapter", action="append", type=_adapter_argument, default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    adapters = dict(args.adapter)
    if set(adapters) != {f"blue-{index}" for index in range(4)}:
        parser.error("adapters must be exactly blue-0 through blue-3")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    trainer = tomllib.loads(args.trainer_config.read_text(encoding="utf-8"))["model"]["lora"]
    expected_rank = int(trainer["rank"])
    expected_alpha = float(trainer["alpha"])
    expected_dropout = float(trainer["dropout"])
    expected_targets = sorted(str(value) for value in trainer["target_modules"])

    records = {}
    args.output_root.mkdir(parents=True)
    for name, source in sorted(adapters.items()):
        weights = source / "adapter_model.safetensors"
        config_path = source / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tensor_rank = _tensor_rank(weights)
        if tensor_rank != expected_rank:
            raise ValueError(f"{name} tensor rank {tensor_rank} != trainer rank {expected_rank}")
        if sorted(str(value) for value in config["target_modules"]) != expected_targets:
            raise ValueError(f"{name} target modules differ from the frozen trainer config")

        destination = args.output_root / name
        destination.mkdir()
        repaired = dict(config)
        repaired["r"] = expected_rank
        repaired["lora_alpha"] = expected_alpha
        repaired["lora_dropout"] = expected_dropout
        repaired_path = destination / "adapter_config.json"
        repaired_path.write_text(
            json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.symlink(weights.resolve(), destination / "adapter_model.safetensors")
        records[name] = {
            "source": str(source),
            "weights_sha256": _sha256(weights),
            "tensor_rank": tensor_rank,
            "original_config_sha256": _sha256(config_path),
            "original_rank": int(config["r"]),
            "original_alpha": float(config["lora_alpha"]),
            "repaired_config_sha256": _sha256(repaired_path),
            "repaired_rank": expected_rank,
            "repaired_alpha": expected_alpha,
        }

    manifest = {
        "version": "swarm-v10-kl-adapter-metadata-repair-v1",
        "scope": "metadata-only repair; adapter safetensors are symlinked and unchanged",
        "trainer_config": str(args.trainer_config),
        "trainer_config_sha256": _sha256(args.trainer_config),
        "expected_dropout": expected_dropout,
        "expected_target_modules": expected_targets,
        "adapters": records,
    }
    manifest_path = args.output_root / "ADAPTER_METADATA_REPAIR.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
