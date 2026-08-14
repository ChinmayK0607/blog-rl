from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import tomli
import tomli_w
from swarm_ctf_eval.live_rl_rollout import parity_gate_sha256

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an immutable four-policy live-RL run layout.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    args = parser.parse_args()

    actual_sha256 = sha256_file(args.adapter / "adapter_model.safetensors")
    if actual_sha256 != args.adapter_sha256:
        raise ValueError(f"adapter checksum mismatch: {actual_sha256}")
    with args.trainer_config.open("rb") as handle:
        config = TrainerConfig.model_validate(tomli.load(handle))
    config.output_dir = args.output_dir
    config.model.name = args.model
    if config.model.lora is None:
        raise ValueError("live RL requires trainer LoRA configuration")
    config.model.lora.initial_adapter_path = args.adapter
    config.model.lora.initial_adapter_sha256 = args.adapter_sha256
    if config.rollout_parity_gate is None:
        raise ValueError("live RL requires a trainer pre-step parity gate")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    resolved_path = args.output_dir / "trainer.toml"
    with resolved_path.open("wb") as handle:
        tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), handle)
    for index in range(4):
        run_dir = args.output_dir / f"run_blue_{index}"
        control_dir = run_dir / "control"
        control_dir.mkdir(parents=True)
        orchestrator = OrchestratorConfig.model_validate(
            {
                "output_dir": run_dir,
                "batch_size": 1,
                "group_size": 1,
                "max_steps": config.max_steps,
                "model": {
                    "name": args.model,
                    "lora": {"name": f"blue-{index}", "rank": 16, "alpha": 32},
                },
                "optim": {"lr": config.optim.lr},
                "train": {"env": [{"id": "reverse-text"}]},
                "renderer": {"name": "qwen3"},
                "wandb": None,
            }
        )
        with (control_dir / "orch.toml").open("wb") as handle:
            tomli_w.dump(orchestrator.model_dump(exclude_none=True, mode="json"), handle)
    print(
        {
            "output_dir": str(args.output_dir),
            "trainer_config": str(resolved_path),
            "trainer_parity_gate_sha256": parity_gate_sha256(config.rollout_parity_gate),
            "adapter_sha256": actual_sha256,
        }
    )


if __name__ == "__main__":
    main()
