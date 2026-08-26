from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tomllib
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from swarm_ctf_eval.handoff_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.rl_production import load_production_plan
from swarm_ctf_eval.safety_supervisor import SharedReturnSpec
from swarm_ctf_eval.task_data_binding import resolve_task_data_binding


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _verify_initial_policy_adapter_manifest(
    trainer: dict, manifest_path: Path | None
) -> str | None:
    lora = trainer["model"]["lora"]
    configured_paths = lora.get("initial_adapter_paths_by_run", {})
    configured_hashes = lora.get("initial_adapter_sha256_by_run", {})
    if manifest_path is None:
        if configured_paths or configured_hashes:
            raise ValueError(
                "distinct trainer warm starts require an initial-policy adapter manifest"
            )
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "swarm-distinct-policy-warmstart-v1":
        raise ValueError("unsupported distinct-policy warm-start manifest")
    policies = manifest.get("policies")
    expected_policies = {f"blue-{index}" for index in range(4)}
    if not isinstance(policies, dict) or set(policies) != expected_policies:
        raise ValueError("warm-start manifest must bind exactly blue-0 through blue-3")
    expected_paths = {}
    expected_hashes = {}
    for index in range(4):
        policy_id = f"blue-{index}"
        run_id = f"run_blue_{index}"
        row = policies[policy_id]
        if not isinstance(row, dict):
            raise ValueError(f"invalid warm-start row for {policy_id}")
        path = Path(str(row["path"])).resolve()
        digest = str(row["sha256"])
        if str(row.get("revision")) != digest:
            raise ValueError(f"warm-start revision mismatch for {policy_id}")
        if _sha256_file(path / "adapter_model.safetensors") != digest:
            raise ValueError(f"warm-start adapter mismatch for {policy_id}")
        expected_paths[run_id] = str(path)
        expected_hashes[run_id] = digest
    if len(set(expected_hashes.values())) != 4:
        raise ValueError("distinct warm start cannot clone one adapter across policies")
    if configured_paths != expected_paths:
        raise ValueError("trainer warm-start paths disagree with controller manifest")
    if configured_hashes != expected_hashes:
        raise ValueError("trainer warm-start hashes disagree with controller manifest")
    return _sha256_file(manifest_path)


def _server_json(base_url: str, suffix: str) -> object:
    with urlopen(f"{base_url.rstrip('/')}{suffix}", timeout=15) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"serving preflight returned HTTP {response.status}")
        return json.loads(response.read())


def _server_health(base_url: str) -> None:
    with urlopen(f"{base_url.rstrip('/')}/health", timeout=15) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"serving health returned HTTP {response.status}")


def _verify_public_inputs(record: dict) -> None:
    for kind in ("base", "adapter"):
        repo = record[f"{kind}_repo"]
        revision = record[f"{kind}_revision"]
        request = Request(
            f"https://huggingface.co/api/models/{repo}/revision/{revision}",
            headers={"User-Agent": "swarm-arena-public-preflight/2"},
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310
            model = json.loads(response.read())
        if model.get("private") is True or model.get("sha") != revision:
            raise RuntimeError(f"{kind} model is not public at its exact revision")
    request = Request(
        record["source_url"],
        headers={"User-Agent": "swarm-arena-public-preflight/2"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"public source returned HTTP {response.status}")
        response.read(1)


def _gpu_inventory() -> list[dict[str, int | str]]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    )
    rows = subprocess.check_output(command, text=True).splitlines()
    inventory = []
    for row in rows:
        index, name, total, used, driver = (
            value.strip() for value in row.split(",", 4)
        )
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "driver_version": driver,
            }
        )
    return inventory


def _validate_shared_return_launcher(plan: object, credit_assignment: str) -> None:
    trainable_phases = (
        ("ACT",)
        if credit_assignment == "focused_agent"
        else plan.trainable_phases
    )
    baselines = [plan.shared_return_baseline]
    if plan.decoy_shared_return_baseline is not None:
        baselines.append(plan.decoy_shared_return_baseline)
    for baseline in baselines:
        SharedReturnSpec(
            replicas=plan.shared_return_replicas,
            trainable_phases=trainable_phases,
            trainable_turn_offsets=plan.trainable_turn_offsets,
            credit_assignment=credit_assignment,
            baseline=baseline,
            action_prompt_profile=plan.action_prompt_profile,
            paired_contrast_centering=plan.paired_contrast_centering,
        ).validate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight for a staged Swarm Arena run."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--production-plan", type=Path, required=True)
    parser.add_argument("--curriculum-artifact", type=Path, required=True)
    parser.add_argument("--runtime-certificate", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--initial-policy-adapter-manifest", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--expected-updates", type=int, default=120)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--shared-return-credit-assignment",
        choices=("shared_team", "focused_agent"),
        default="shared_team",
    )
    parser.add_argument("--shared-return-replicas", type=int, default=4)
    parser.add_argument(
        "--action-prompt-profile",
        choices=("full", "focused_handoff_compact"),
        default="full",
    )
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--skip-serving", action="store_true")
    args = parser.parse_args()

    head = subprocess.check_output(
        ("git", "-C", str(args.repo_root), "rev-parse", "HEAD"), text=True
    ).strip()
    if head != args.source_commit:
        raise ValueError(f"source commit mismatch: expected {args.source_commit}, got {head}")
    dirty = subprocess.check_output(
        ("git", "-C", str(args.repo_root), "status", "--porcelain"), text=True
    ).strip()
    if dirty:
        raise ValueError("repository must be clean before paid execution")
    if args.expected_updates < 1 or args.checkpoint_interval < 1:
        raise ValueError("updates and checkpoint interval must be positive")
    if args.expected_updates % args.checkpoint_interval:
        raise ValueError("checkpoint interval must divide expected updates")

    prepare_path = args.run_dir / "PREPARE.json"
    trainer_path = args.run_dir / "trainer.toml"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    trainer = _read_toml(trainer_path)
    initial_policy_adapter_manifest_sha256 = (
        _verify_initial_policy_adapter_manifest(
            trainer, args.initial_policy_adapter_manifest
        )
    )
    inference = _read_toml(args.inference_config)
    certificate = json.loads(args.runtime_certificate.read_text(encoding="utf-8"))
    certificate_body = {
        key: value for key, value in certificate.items() if key != "sha256"
    }
    certificate_sha256 = hashlib.sha256(
        json.dumps(certificate_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if certificate.get("sha256") != certificate_sha256:
        raise ValueError("runtime certificate body hash mismatch")
    if (
        certificate.get("version") != "swarm-runtime-certificate-v1"
        or certificate.get("status") != "passed"
    ):
        raise ValueError("runtime certificate is not a passed v1 certificate")
    if certificate["source_commit"] != head:
        raise ValueError("runtime certificate was produced from a different source commit")
    if prepare["version"] != "swarm-live-rl-prepare-v2":
        raise ValueError("run directory was not created by the current public preflight")
    if prepare["trainer_config_sha256"] != _sha256_file(trainer_path):
        raise ValueError("resolved trainer config changed after public-input preparation")
    if prepare["policy_steps"] != args.expected_updates:
        raise ValueError("prepared policy-step count does not match staged run")
    if prepare["checkpoint_interval"] != args.checkpoint_interval:
        raise ValueError("prepared checkpoint interval does not match staged run")
    _verify_public_inputs(prepare["public_inputs"])
    if certificate["base_revision"] != prepare["public_inputs"]["base_revision"]:
        raise ValueError("runtime certificate does not bind the public base revision")
    if certificate["trainer_config_sha256"] != _sha256_file(trainer_path):
        raise ValueError("runtime certificate does not bind the resolved trainer config")
    if certificate.get("initial_policy_adapter_manifest_sha256") != (
        initial_policy_adapter_manifest_sha256
    ):
        raise ValueError("runtime certificate does not bind the policy warm-start manifest")
    if (
        certificate["parity_report"]["trainer_parity_gate_sha256"]
        != prepare["trainer_parity_gate_sha256"]
    ):
        raise ValueError("runtime certificate does not bind the trainer parity gate")
    if certificate["inference_config_sha256"] != _sha256_file(args.inference_config):
        raise ValueError("runtime certificate does not bind the inference config")

    if trainer["max_concurrent_runs"] != 4:
        raise ValueError("staged run requires exactly four trainer policy slots")
    if not trainer.get("atomic_multi_run_updates", False):
        raise ValueError("staged run requires atomic four-policy optimizer updates")
    if trainer.get("max_steps") is not None:
        raise ValueError("trainer max_steps must remain controller-owned")
    # The resolved trainer config is already immutably bound by PREPARE.json,
    # the parity report, and the runtime certificate.  Do not duplicate an old
    # experiment's exact hyperparameters here: doing so made this safety check
    # reject intentionally preregistered 4B/rank-32 runs after they had passed
    # exact-runtime calibration.
    if not 0.0 < float(trainer["optim"]["lr"]) <= 1e-3:
        raise ValueError(
            "staged run requires a finite positive learning rate no larger than 1e-3"
        )
    if int(trainer["model"]["lora"]["rank"]) < 1:
        raise ValueError("staged run requires a positive LoRA rank")
    if trainer["ckpt"]["interval"] != args.checkpoint_interval:
        raise ValueError("trainer checkpoint interval mismatch")
    if trainer["ckpt"]["keep_interval"] != args.checkpoint_interval:
        raise ValueError("trainer must permanently retain every evaluation checkpoint")
    if trainer["ckpt"].get("keep_last") != 2:
        raise ValueError("trainer checkpoint rolling retention must remain two")
    if not trainer["ckpt"].get("weights_only"):
        raise ValueError("staged trainer checkpoints must be weights-only")
    if not trainer["ckpt"]["weights"].get("save_adapter_separately"):
        raise ValueError("staged trainer checkpoints must export separate LoRA adapters")
    if not trainer["wandb"].get("offline", False):
        raise ValueError("trainer W&B must be offline so network failure cannot stop training")
    if trainer["model"]["seq_len"] > inference["model"]["max_model_len"]:
        raise ValueError("trainer sequence length exceeds inference model context")
    if not inference.get("enable_lora") or inference.get("max_loras", 0) < 8:
        raise ValueError("inference config cannot host the required LoRA roster")
    if inference.get("max_lora_rank", 0) < trainer["model"]["lora"]["rank"]:
        raise ValueError("inference LoRA rank is smaller than trainer LoRA rank")

    adapter_file = args.initial_adapter / "adapter_model.safetensors"
    adapter_sha256 = _sha256_file(adapter_file)
    if adapter_sha256 != trainer["model"]["lora"]["initial_adapter_sha256"]:
        raise ValueError("initial adapter bytes do not match resolved trainer config")
    if adapter_sha256 != prepare["adapter_sha256"]:
        raise ValueError("initial adapter bytes do not match public preparation record")
    if adapter_sha256 != certificate["adapter_sha256"]:
        raise ValueError("initial adapter bytes do not match runtime certificate")
    if not args.model.is_dir():
        raise FileNotFoundError(f"local pinned model is missing: {args.model}")

    plan, opponent_paths = load_production_plan(args.production_plan)
    raw_plan = json.loads(args.production_plan.read_text(encoding="utf-8"))
    curriculum = json.loads(args.curriculum_artifact.read_text(encoding="utf-8"))
    curriculum_sha256 = hashlib.sha256(
        json.dumps(curriculum, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if raw_plan.get("curriculum_source", {}).get("sha256") != curriculum_sha256:
        raise ValueError("production plan does not bind the supplied curriculum artifact")
    if raw_plan.get("runtime_certificate", {}).get("sha256") != certificate_sha256:
        raise ValueError("production plan does not bind the supplied runtime certificate")
    if plan.backend.calibration_sha256 != certificate_sha256:
        raise ValueError("production backend calibration is not the runtime certificate")
    if plan.backend.kernel_config_sha256 != _sha256_file(args.inference_config):
        raise ValueError("production backend kernel config does not match inference config")
    if plan.backend.name != certificate["backend"]["name"]:
        raise ValueError("production backend name does not match runtime certificate")
    if plan.backend.version != certificate["backend"]["version"]:
        raise ValueError("production backend version does not match runtime certificate")
    if args.shared_return_replicas != plan.shared_return_replicas:
        raise ValueError("launcher replica count does not match the immutable production plan")
    if args.action_prompt_profile != plan.action_prompt_profile:
        raise ValueError("launcher action prompt profile does not match the immutable production plan")
    _validate_shared_return_launcher(plan, args.shared_return_credit_assignment)
    base_snapshots = [
        row for row in plan.opponent_pool.snapshots if row.family == "base"
    ]
    if len(base_snapshots) != 1 or base_snapshots[0].revision != certificate["base_revision"]:
        raise ValueError("runtime certificate does not bind the production base revision")
    schedule = plan.curriculum_schedule(steps=args.expected_updates)
    if len(schedule) != args.expected_updates * plan.groups_per_update:
        raise ValueError("staged schedule length mismatch")
    if args.shared_return_credit_assignment == "focused_agent":
        scheduled_phases = {"ACT"}
        if any(row.handoff_focus_role == "sender" for row in schedule):
            scheduled_phases.add("BROADCAST")
        missing_phases = scheduled_phases - set(plan.trainable_phases)
        if missing_phases:
            raise ValueError(
                f"focused-agent schedule uses phases absent from the production plan: {sorted(missing_phases)}"
            )
    handoff = json.loads((args.data_dir / "handoff_train.json").read_text(encoding="utf-8"))
    pair_indices = {row.pair_index for row in schedule if row.pair_index is not None}
    if max(pair_indices) >= int(handoff["pair_count"]):
        raise ValueError("staged schedule exceeds the bound handoff manifest")
    for pair_index in sorted(pair_indices):
        pair = handoff["pairs"][pair_index]
        reconstruct_manifest_scenario(pair["critical"])
        reconstruct_manifest_scenario(pair["decoy"])
    ordinary_seeds = [row.ordinary_seed for row in schedule if row.ordinary_seed is not None]
    if len(ordinary_seeds) != len(set(ordinary_seeds)):
        raise ValueError("ordinary training seeds are not unique")
    opponent_schedule = plan.opponent_pool.schedule(len(schedule))
    for update in range(args.expected_updates):
        block = opponent_schedule[
            update * plan.groups_per_update : (update + 1) * plan.groups_per_update
        ]
        if {row.family for row in block} != {"base", "sft", "historical", "current"}:
            raise ValueError(f"update {update} lacks an exact opponent rotation")
    for snapshot in plan.opponent_pool.snapshots:
        path = opponent_paths[snapshot.opponent_id]
        if path is None:
            continue
        actual = _sha256_file(path / "adapter_model.safetensors")
        if actual != snapshot.adapter_sha256:
            raise ValueError(f"opponent adapter hash mismatch: {snapshot.opponent_id}")

    binding = resolve_task_data_binding(args.data_dir, "v4")
    for index in range(4):
        orch_path = args.run_dir / f"run_blue_{index}" / "control" / "orch.toml"
        orch = _read_toml(orch_path)
        if orch["max_steps"] != args.expected_updates:
            raise ValueError(f"blue-{index} orchestrator update count mismatch")
        if orch["ckpt"]["interval"] != args.checkpoint_interval:
            raise ValueError(f"blue-{index} checkpoint interval mismatch")
        if orch["ckpt"].get("keep_interval") != args.checkpoint_interval:
            raise ValueError(f"blue-{index} permanent checkpoint retention mismatch")
        if orch["ckpt"].get("keep_last") != 2:
            raise ValueError(f"blue-{index} rolling checkpoint retention mismatch")
    if (args.run_dir / "live_rl_progress.json").exists():
        raise FileExistsError("fresh staged run directory already contains progress")

    free_gib = shutil.disk_usage(args.run_dir).free / 2**30
    if free_gib < args.minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free; require {args.minimum_free_gib:.1f} GiB"
        )
    gpu_inventory = []
    if not args.skip_hardware:
        if importlib.metadata.version("vllm") != plan.backend.version:
            raise ValueError("installed vLLM version differs from the certified backend")
        gpu_inventory = _gpu_inventory()
        if len(gpu_inventory) != 4:
            raise RuntimeError("staged run requires exactly four visible GPUs")
        if min(int(row["memory_total_mib"]) for row in gpu_inventory) < 22_000:
            raise RuntimeError("every visible GPU must have at least 22 GiB VRAM")
        if args.skip_serving:
            if max(int(row["memory_used_mib"]) for row in gpu_inventory) > 1_024:
                raise RuntimeError("hardware preflight requires idle GPUs before model launch")
        elif int(gpu_inventory[0]["memory_used_mib"]) > 1_024:
            raise RuntimeError("GPU 0 must remain idle for the trainer before launch")
        certified_gpus = [
            {
                "index": row["index"],
                "name": row["name"],
                "memory_total_mib": row["memory_total_mib"],
                "driver_version": row["driver_version"],
            }
            for row in certificate["gpu_inventory"]
        ]
        current_gpus = [
            {
                "index": row["index"],
                "name": row["name"],
                "memory_total_mib": row["memory_total_mib"],
                "driver_version": row["driver_version"],
            }
            for row in gpu_inventory
        ]
        if current_gpus != certified_gpus:
            raise RuntimeError("current GPUs differ from the certified runtime inventory")
    serving = []
    if not args.skip_serving:
        if len(args.base_url) != 3:
            raise ValueError("staged run requires exactly three rollout server URLs")
        if args.base_url != certificate["serving_probe"]["base_urls"]:
            raise ValueError("rollout server URLs differ from the certified serving probe")
        for base_url in args.base_url:
            _server_health(base_url)
            registry = _server_json(base_url, "/v1/models")
            serving.append({"base_url": base_url, "models": len(registry["data"])})

    report = {
        "version": "swarm-staged-rl-preflight-v2",
        "status": "passed",
        "source_commit": head,
        "production_plan_sha256": plan.sha256,
        "curriculum_sha256": curriculum_sha256,
        "runtime_certificate_sha256": certificate_sha256,
        "task": {
            "version": binding.task_version,
            "train_sha256": binding.train_sha256,
            "development_sha256": binding.development_sha256,
            "final_sha256": binding.final_sha256,
        },
        "schedule": {
            "updates": args.expected_updates,
            "groups": len(schedule),
            "counts": dict(Counter(row.kind for row in schedule)),
            "handoff_pairs": len(pair_indices),
            "ordinary_seeds": len(ordinary_seeds),
            "maximum_ordinary_size": max(
                row.ordinary_size or 0 for row in schedule
            ),
            "maximum_ordinary_horizon": max(
                row.ordinary_horizon or 0 for row in schedule
            ),
        },
        "adapter_sha256": adapter_sha256,
        "initial_policy_adapter_manifest_sha256": (
            initial_policy_adapter_manifest_sha256
        ),
        "checkpoint_interval": args.checkpoint_interval,
        "free_gib": free_gib,
        "gpus": gpu_inventory,
        "serving": serving,
    }
    output = args.run_dir / "PREFLIGHT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
