from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import tomli
import tomli_w
from huggingface_hub import HfApi, hf_hub_download
from swarm_ctf_eval.live_rl_rollout import parity_gate_sha256

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_inputs(
    *,
    base_repo: str,
    base_revision: str,
    adapter_repo: str,
    adapter_revision: str,
    adapter_sha256: str,
    source_url: str,
) -> dict[str, str]:
    """Fail before paid work unless every required input is anonymously accessible."""
    anonymous = HfApi(token=False)
    base = anonymous.model_info(base_repo, revision=base_revision)
    if base.private or base.sha != base_revision:
        raise RuntimeError("base model is not public at the exact pinned revision")
    adapter = anonymous.model_info(adapter_repo, revision=adapter_revision)
    if adapter.private or adapter.sha != adapter_revision:
        raise RuntimeError("adapter is not public at the exact pinned revision")
    with tempfile.TemporaryDirectory(prefix="swarm-public-adapter-") as cache:
        downloaded = Path(
            hf_hub_download(
                repo_id=adapter_repo,
                filename="adapter_model.safetensors",
                revision=adapter_revision,
                token=False,
                cache_dir=cache,
            )
        )
        if sha256_file(downloaded) != adapter_sha256:
            raise RuntimeError("public adapter bytes do not match the local pinned adapter")
    request = Request(source_url, headers={"User-Agent": "swarm-arena-public-preflight/1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"public source returned HTTP {response.status}")
        response.read(1)
    return {
        "base_repo": base_repo,
        "base_revision": base_revision,
        "adapter_repo": adapter_repo,
        "adapter_revision": adapter_revision,
        "source_url": source_url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an immutable four-policy live-RL run layout.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--public-base-repo", required=True)
    parser.add_argument("--public-base-revision", required=True)
    parser.add_argument("--public-adapter-repo", required=True)
    parser.add_argument("--public-adapter-revision", required=True)
    parser.add_argument("--public-source-url", required=True)
    parser.add_argument(
        "--policy-steps",
        type=int,
        required=True,
        help="Number of logical per-policy updates; distinct from trainer packing slices.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="preserve resume-capable per-policy checkpoints at this logical interval",
    )
    args = parser.parse_args()
    if args.policy_steps < 1:
        parser.error("policy-steps must be positive")
    if args.checkpoint_interval is not None and (
        args.checkpoint_interval < 1
        or args.policy_steps % args.checkpoint_interval
    ):
        parser.error("checkpoint interval must positively divide policy steps")

    actual_sha256 = sha256_file(args.adapter / "adapter_model.safetensors")
    if actual_sha256 != args.adapter_sha256:
        raise ValueError(f"adapter checksum mismatch: {actual_sha256}")
    public_inputs = verify_public_inputs(
        base_repo=args.public_base_repo,
        base_revision=args.public_base_revision,
        adapter_repo=args.public_adapter_repo,
        adapter_revision=args.public_adapter_revision,
        adapter_sha256=args.adapter_sha256,
        source_url=args.public_source_url,
    )
    with args.trainer_config.open("rb") as handle:
        config = TrainerConfig.model_validate(tomli.load(handle))
    if config.max_steps is not None:
        raise ValueError(
            "multi-run Swarm trainer max_steps must be omitted: Prime counts packing "
            "slices, while --policy-steps controls logical policy updates"
        )
    if not config.atomic_multi_run_updates:
        raise ValueError("multi-policy Swarm training requires atomic trainer updates")
    config.output_dir = args.output_dir
    config.model.name = args.model
    if config.model.lora is None:
        raise ValueError("live RL requires trainer LoRA configuration")
    config.model.lora.initial_adapter_path = args.adapter
    config.model.lora.initial_adapter_sha256 = args.adapter_sha256
    per_run_paths = config.model.lora.initial_adapter_paths_by_run
    per_run_hashes = config.model.lora.initial_adapter_sha256_by_run
    if per_run_paths:
        expected_runs = {f"run_blue_{index}" for index in range(4)}
        if set(per_run_paths) != expected_runs or set(per_run_hashes) != expected_runs:
            raise ValueError("distinct warm start must bind exactly four run_blue_* adapters")
        for run_id, adapter_path in sorted(per_run_paths.items()):
            actual = sha256_file(adapter_path / "adapter_model.safetensors")
            if actual != per_run_hashes[run_id]:
                raise ValueError(f"distinct warm-start checksum mismatch for {run_id}")
    if config.rollout_parity_gate is None:
        raise ValueError("live RL requires a trainer pre-step parity gate")
    if args.checkpoint_interval is not None:
        if config.ckpt is None or config.ckpt.interval != args.checkpoint_interval:
            raise ValueError(
                "trainer checkpoint interval must exactly match --checkpoint-interval"
            )

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
                "max_steps": args.policy_steps,
                "model": {
                    "name": args.model,
                    "lora": {
                        "name": f"blue-{index}",
                        "rank": config.model.lora.rank,
                        "alpha": config.model.lora.alpha,
                    },
                },
                "optim": {"lr": config.optim.lr},
                "train": {"env": [{"id": "reverse-text"}]},
                "renderer": {"name": "qwen3"},
                "wandb": None,
                "ckpt": (
                    None
                    if args.checkpoint_interval is None
                    else {
                        "interval": args.checkpoint_interval,
                        "keep_last": 2,
                        "keep_interval": args.checkpoint_interval,
                    }
                ),
            }
        )
        with (control_dir / "orch.toml").open("wb") as handle:
            tomli_w.dump(orchestrator.model_dump(exclude_none=True, mode="json"), handle)
    report = {
        "version": "swarm-live-rl-prepare-v2",
        "output_dir": str(args.output_dir),
        "trainer_config": str(resolved_path),
        "trainer_config_sha256": sha256_file(resolved_path),
        "trainer_parity_gate_sha256": parity_gate_sha256(config.rollout_parity_gate),
        "policy_steps": args.policy_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "adapter_sha256": actual_sha256,
        "public_inputs": public_inputs,
    }
    (args.output_dir / "PREPARE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
