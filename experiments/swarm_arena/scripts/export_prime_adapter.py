from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prime_rl.trainer.weights import peft_adapter_state_dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one Prime multi-run LoRA broadcast as a standard PEFT adapter."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    weights_path = args.input_dir / "adapter_model.safetensors"
    config_path = args.input_dir / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Prime adapter requires config and safetensors weights")

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        raw = {key: handle.get_tensor(key) for key in handle.keys()}
    exported = peft_adapter_state_dict(raw)
    prefix = "base_model.model."
    if len(exported) != len(raw) or not all(
        torch.equal(value, exported[key if key.startswith(prefix) else f"{prefix}{key}"])
        for key, value in raw.items()
    ):
        raise RuntimeError("PEFT export changed adapter tensor ownership")

    args.output_dir.mkdir(parents=True)
    output_weights = args.output_dir / "adapter_model.safetensors"
    save_file(exported, output_weights, metadata={"format": "pt"})
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_base_model = adapter_config.get("base_model_name_or_path")
    adapter_config["base_model_name_or_path"] = args.base_model
    (args.output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "version": "prime-multirun-peft-export-v1",
        "source": str(args.input_dir),
        "source_base_model": source_base_model,
        "export_base_model": args.base_model,
        "source_adapter_sha256": sha256_file(weights_path),
        "export_adapter_sha256": sha256_file(output_weights),
        "tensors": len(exported),
        "all_keys_peft_prefixed": all(
            key.startswith("base_model.model.") for key in exported
        ),
        "tensor_shapes": {
            key: list(value.shape) for key, value in sorted(exported.items())
        },
    }
    (args.output_dir / "EXPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
