from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import tomli
from swarm_ctf_eval.adaptive_curriculum import (
    adapt_stage_assignments,
    summarize_training_progress,
)
from swarm_ctf_eval.async_admission import AsyncRolloutHeader, PolicySnapshot
from swarm_ctf_eval.async_rescore import (
    FilesystemCurrentPolicyRescorer,
    write_current_snapshot_manifest,
)
from swarm_ctf_eval.async_training_queue import AtomicAsyncTrainingQueue
from swarm_ctf_eval.communication_curriculum import (
    reconstruct_manifest_scenario as reconstruct_v3_scenario,
)
from swarm_ctf_eval.episode import EpisodeConfig
from swarm_ctf_eval.handoff_curriculum import (
    reconstruct_manifest_scenario as reconstruct_v4_scenario,
)
from swarm_ctf_eval.hf_choice_generator import HFChoiceGenerator
from swarm_ctf_eval.live_rl_rollout import (
    PolicyEndpoint,
    VLLMChoiceGenerator,
    build_live_credit_group,
    build_live_message_credit_group,
    build_live_shared_return_group,
    parity_gate_sha256,
    protocol_constraint_sha256,
)
from swarm_ctf_eval.message_credit_audit import message_credit_audit_record
from swarm_ctf_eval.message_interventions import TargetSwapIneligibleError
from swarm_ctf_eval.multi_policy_contract import AgentPolicy
from swarm_ctf_eval.prime_multi_run_router import (
    PolicyRunRoute,
    merge_routed_batch_groups,
    route_approved_samples,
    send_approved_batches,
    validate_single_trajectory_packing,
    validate_training_batch_lengths,
)
from swarm_ctf_eval.rl_production import (
    OpponentSnapshot,
    ProductionPlan,
    ScenarioAssignment,
    load_production_plan,
    logical_update_has_signal,
    scenario_sampling_namespace,
)
from swarm_ctf_eval.safety_supervisor import (
    RunLock,
    SharedReturnSpec,
    append_hash_chained_record,
    approve_credit_group,
    approve_message_credit_group,
    approve_shared_return_group,
    canonical_sha256,
    shared_return_evidence_payload,
)
from swarm_ctf_eval.task_data_binding import (
    TaskDataBinding,
    resolve_task_data_binding,
)

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_adaptive_selection(output_dir: Path, selection: dict[str, object]) -> None:
    root = output_dir / "audit" / "adaptive_curriculum"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{selection['stage']}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != selection:
            raise ValueError(f"adaptive curriculum selection changed on resume: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def adapt_next_curriculum_stage(
    schedule: tuple[ScenarioAssignment, ...],
    *,
    production_plan: ProductionPlan,
    curriculum: dict[str, object],
    progress: list[dict[str, object]],
    stage_index: int,
    output_dir: Path,
) -> tuple[ScenarioAssignment, ...]:
    config = production_plan.adaptive_curriculum
    if config is None or stage_index == 0:
        return schedule
    if stage_index >= len(production_plan.stages):
        raise ValueError("adaptive stage index exceeds the production plan")
    previous_start = sum(stage.updates for stage in production_plan.stages[: stage_index - 1])
    previous_end = previous_start + production_plan.stages[stage_index - 1].updates
    if len(progress) < previous_end:
        raise ValueError("adaptive curriculum cannot inspect an incomplete previous stage")
    analysis = summarize_training_progress(
        progress,
        config=config,
        step_start=previous_start,
        step_end=previous_end,
    )
    receiver_by_case: dict[tuple[int, str], str] = {}
    for pair_index, pair in enumerate(curriculum["pairs"]):
        receiver = str(pair["critical"]["receiver"])
        for world in pair["critical"]["worlds"]:
            receiver_by_case[(pair_index, str(world["label"]))] = receiver
    candidate_pool_by_receiver: dict[str, list[tuple[int, str]]] = {}
    for value in config.candidate_cases:
        pair_index_text, world, receiver = value.split(":")
        case = (int(pair_index_text), world)
        if receiver_by_case.get(case) != receiver:
            raise ValueError(f"adaptive candidate pool receiver mismatch: {value}")
        candidate_pool_by_receiver.setdefault(receiver, []).append(case)
    adapted, selection = adapt_stage_assignments(
        schedule,
        stage_name=production_plan.stages[stage_index].name,
        analysis=analysis,
        receiver_by_case=receiver_by_case,
        config=config,
        candidate_pool_by_receiver=candidate_pool_by_receiver,
    )
    write_adaptive_selection(output_dir, selection)
    return adapted


def load_initial_policy_adapter_manifest(
    path: Path | None,
) -> dict[str, tuple[Path, str, str]]:
    """Load a hash-pinned distinct warm start for each of the four policies."""
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != "swarm-distinct-policy-warmstart-v1":
        raise ValueError("unsupported distinct-policy warm-start manifest")
    policies = raw.get("policies")
    expected = {f"blue-{index}" for index in range(4)}
    if not isinstance(policies, dict) or set(policies) != expected:
        raise ValueError("warm-start manifest must bind exactly blue-0 through blue-3")
    result = {}
    for policy_id, row in policies.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid warm-start row for {policy_id}")
        adapter_path = Path(str(row["path"])).resolve()
        sha256 = str(row["sha256"])
        revision = str(row["revision"])
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError(f"invalid warm-start SHA-256 for {policy_id}")
        if not revision:
            raise ValueError(f"missing warm-start revision for {policy_id}")
        adapter_file = adapter_path / "adapter_model.safetensors"
        if not adapter_file.is_file() or sha256_file(adapter_file) != sha256:
            raise ValueError(f"warm-start adapter mismatch for {policy_id}")
        result[policy_id] = (adapter_path, sha256, revision)
    return result


def load_resume_progress(output_dir: Path, *, total_steps: int) -> list[dict[str, object]]:
    progress_path = output_dir / "live_rl_progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError(f"resume progress does not exist: {progress_path}")
    rows = json.loads(progress_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("resume progress must contain at least one completed update")
    expected_steps = list(range(len(rows)))
    actual_steps = [row.get("step") if isinstance(row, dict) else None for row in rows]
    if actual_steps != expected_steps:
        raise ValueError("resume progress steps must be contiguous from zero")
    if len(rows) >= total_steps:
        raise ValueError("resume progress already reached the requested update count")
    policy_hashes = rows[-1].get("policy_adapter_sha256")
    if not isinstance(policy_hashes, dict) or set(policy_hashes) != {
        "blue-0",
        "blue-1",
        "blue-2",
        "blue-3",
    }:
        raise ValueError("resume progress is missing the complete four-policy adapter set")
    if any(not isinstance(value, str) or len(value) != 64 for value in policy_hashes.values()):
        raise ValueError("resume progress contains an invalid policy adapter digest")
    return rows


def resume_adapter_paths(
    output_dir: Path,
    *,
    completed_steps: int,
    expected_sha256: dict[str, str],
) -> dict[str, Path]:
    paths = {
        f"blue-{index}": get_step_path(
            get_broadcast_dir(output_dir / f"run_blue_{index}"),
            completed_steps,
        )
        for index in range(4)
    }
    for name, path in paths.items():
        if not (path / "STABLE").is_file():
            raise FileNotFoundError(f"resume adapter is not stable for {name}: {path}")
        actual = sha256_file(path / "adapter_model.safetensors")
        if actual != expected_sha256[name]:
            raise ValueError(
                f"resume adapter checksum mismatch for {name}: "
                f"expected {expected_sha256[name]}, got {actual}"
            )
    return paths


def signing_key(path: Path) -> bytes:
    if path.exists():
        key = path.read_bytes()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(32)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, key)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if len(key) < 32:
        raise ValueError("supervisor signing key is too short")
    return key


async def checkpoint_barrier(
    root: Path,
    *,
    step: int,
    run_id: str,
    plan_sha256: str,
    policy_adapter_sha256: dict[str, str],
    policy_revision: str,
    timeout: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ready_path = root / f"step_{step}.ready.json"
    continue_path = root / f"step_{step}.continue.json"
    if ready_path.exists() or continue_path.exists():
        raise FileExistsError(f"stale checkpoint barrier files for step {step}")
    payload = {
        "version": "swarm-checkpoint-barrier-v1",
        "step": step,
        "run_id": run_id,
        "production_plan_sha256": plan_sha256,
        "policy_adapter_sha256": policy_adapter_sha256,
        "policy_revision": policy_revision,
    }
    payload["ready_sha256"] = canonical_sha256(payload)
    temporary = ready_path.with_suffix(ready_path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, ready_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if continue_path.is_file():
            continuation = json.loads(continue_path.read_text(encoding="utf-8"))
            if continuation != {
                "version": "swarm-checkpoint-barrier-v1",
                "step": step,
                "ready_sha256": payload["ready_sha256"],
            }:
                raise ValueError(f"invalid checkpoint continuation for step {step}")
            return
        await asyncio.sleep(1.0)
    raise TimeoutError(f"checkpoint evaluation barrier timed out at step {step}")


async def replace_adapter(base_urls: tuple[str, ...], name: str, path: Path) -> None:
    """Atomically refresh one named LoRA and verify the server's registered path."""
    expected_path = str(path.resolve())
    async with httpx.AsyncClient(timeout=120.0) as client:
        current_registries = await asyncio.gather(
            *(client.get(f"{base_url.rstrip('/')}/v1/models") for base_url in base_urls)
        )
        for response in current_registries:
            response.raise_for_status()
        if all(
            _registry_binds_adapter(response.json(), name, expected_path)
            for response in current_registries
        ):
            return
        unloads = await asyncio.gather(
            *(
                client.post(
                    f"{base_url.rstrip('/')}/v1/unload_lora_adapter",
                    json={"lora_name": name},
                )
                for base_url in base_urls
            )
        )
        for response in unloads:
            if response.status_code not in {200, 404}:
                response.raise_for_status()
        loads = await asyncio.gather(
            *(
                client.post(
                    f"{base_url.rstrip('/')}/v1/load_lora_adapter",
                    json={"lora_name": name, "lora_path": str(path)},
                )
                for base_url in base_urls
            )
        )
        for response in loads:
            response.raise_for_status()
        registries = await asyncio.gather(*(client.get(f"{base_url.rstrip('/')}/v1/models") for base_url in base_urls))
    for response in registries:
        response.raise_for_status()
        if not _registry_binds_adapter(response.json(), name, expected_path):
            raise RuntimeError(f"LoRA registry did not bind {name} to {expected_path}")


def _registry_binds_adapter(
    payload: dict[str, object], name: str, expected_path: str
) -> bool:
    rows = payload.get("data")
    if not isinstance(rows, list):
        return False
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("id") == name
    ]
    return len(matches) == 1 and matches[0].get("root") == expected_path


async def wait_for_policy_updates(
    output_dir: Path,
    base_urls: tuple[str, ...],
    *,
    expected_step: int,
    timeout: float,
    hf_generator: HFChoiceGenerator | None = None,
) -> dict[str, str]:
    paths = {
        f"blue-{index}": get_step_path(get_broadcast_dir(output_dir / f"run_blue_{index}"), expected_step)
        for index in range(4)
    }
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all((path / "STABLE").exists() for path in paths.values()):
            break
        await asyncio.sleep(1.0)
    else:
        missing = [name for name, path in paths.items() if not (path / "STABLE").exists()]
        raise TimeoutError(f"trainer did not publish all policy updates: {missing}")
    for name, path in paths.items():
        if hf_generator is None:
            await replace_adapter(base_urls, name, path)
        else:
            await hf_generator.replace_adapter(name, path)
    return {name: sha256_file(path / "adapter_model.safetensors") for name, path in paths.items()}


def run_lock(
    args: argparse.Namespace,
    config: TrainerConfig,
    *,
    data_binding: TaskDataBinding,
    policy_revisions: dict[str, str],
    shared_return_spec: SharedReturnSpec | None,
    opponent: OpponentSnapshot | None = None,
    production_plan: ProductionPlan | None = None,
) -> RunLock:
    if config.rollout_parity_gate is None:
        raise ValueError("trainer config is missing the pre-step parity gate")
    opponent_revision = opponent.revision if opponent is not None else args.opponent_revision
    if opponent_revision is None:
        raise ValueError("run lock requires an immutable opponent revision")
    frozen_revisions = [("red-opponent", opponent_revision)]
    replacement_policy_id = None
    if args.credit_estimator == "policy_replacement":
        if args.replacement_revision is None:
            raise ValueError("policy-replacement credit requires --replacement-revision")
        replacement_policy_id = args.replacement_policy_id
        frozen_revisions.append((args.replacement_policy_id, args.replacement_revision))
    return RunLock(
        args.run_id,
        args.source_commit,
        data_binding.task_version,
        data_binding.train_sha256,
        data_binding.development_sha256,
        data_binding.final_sha256,
        args.base_revision,
        tuple((f"blue-policy-{index}", policy_revisions[f"blue-{index}"]) for index in range(4)),
        tuple(frozen_revisions),
        replacement_policy_id,
        opponent.opponent_id if opponent is not None else "sft-opponent",
        opponent_revision,
        (
            protocol_constraint_sha256("BROADCAST"),
            protocol_constraint_sha256("ACT"),
        ),
        parity_gate_sha256(config.rollout_parity_gate),
        args.credit_estimator,
        shared_return_spec.sha256 if shared_return_spec is not None else None,
        sha256_file(args.trainer_config) if shared_return_spec is not None else None,
        sha256_file(args.inference_config) if shared_return_spec is not None else None,
        production_plan.sha256 if production_plan is not None else None,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded fail-closed four-policy Swarm Arena RL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task-data-version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument(
        "--initial-policy-adapter-manifest",
        type=Path,
        help="optional four-policy hash-pinned warm start; overrides the common adapter",
    )
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument(
        "--actor",
        choices=("vllm", "hf", "prime"),
        default="vllm",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--initial-policy-revision", required=True)
    parser.add_argument(
        "--credit-estimator",
        choices=("shared_return", "message_drop", "policy_replacement"),
        default="shared_return",
    )
    parser.add_argument("--shared-return-replicas", type=int, default=4)
    parser.add_argument(
        "--shared-return-credit-assignment",
        choices=("shared_team", "focused_agent"),
        default="shared_team",
    )
    parser.add_argument(
        "--shared-return-action-prompt-profile",
        choices=("full", "focused_handoff_compact"),
        default="full",
    )
    parser.add_argument(
        "--shared-return-trainable-phase",
        action="append",
        choices=("BROADCAST", "ACT"),
        default=[],
        help="select trainable phases for a non-production rollout diagnostic",
    )
    parser.add_argument(
        "--shared-return-all-turns",
        action="store_true",
        help="select every turn for a non-production rollout diagnostic",
    )
    parser.add_argument("--replacement-policy-id", default="sft-replacement")
    parser.add_argument("--replacement-model-name", default="sft-replacement")
    parser.add_argument("--replacement-revision")
    parser.add_argument("--opponent-revision")
    parser.add_argument("--opponent-model-name", default="sft-opponent")
    parser.add_argument("--opponent-adapter-path", type=Path)
    parser.add_argument("--opponent-adapter-sha256")
    parser.add_argument(
        "--opponent-family",
        choices=("base", "sft", "historical", "current"),
        default="sft",
    )
    parser.add_argument(
        "--production-plan",
        type=Path,
        help=("immutable v4 plan for all-turn spans, exact curriculum, opponent pool, and bounded async admission"),
    )
    parser.add_argument("--async-rescore-dir", type=Path)
    parser.add_argument("--async-rescore-timeout", type=float, default=600.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--groups-per-step", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=7_000_003)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument(
        "--scenario-source",
        choices=("curriculum", "ordinary"),
        default="curriculum",
    )
    parser.add_argument("--curriculum-split", default="train")
    parser.add_argument(
        "--curriculum-kind",
        choices=("alternating", "critical", "decoy"),
        default="alternating",
    )
    parser.add_argument("--curriculum-offset", type=int, default=0)
    parser.add_argument(
        "--resume-existing-progress",
        action="store_true",
        help=(
            "resume a stopped controller from its contiguous progress file and "
            "the trainer's matching stable four-policy broadcast"
        ),
    )
    parser.add_argument(
        "--target-swap-sender-retries",
        type=int,
        default=0,
        help=(
            "boundedly resample only the frozen sender broadcast when a paired "
            "target-swap counterfactual is undefined"
        ),
    )
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument(
        "--ordinary-focused-agent",
        choices=("blue-0", "blue-1", "blue-2", "blue-3"),
        help=(
            "focus one policy in a one-group ordinary rollout-only signal screen; "
            "never valid for optimizer training"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help=(
            "override the episode horizon; curriculum scenarios otherwise use "
            "their manifest-certified horizon and ordinary games default to 2"
        ),
    )
    parser.add_argument("--update-timeout", type=float, default=600.0)
    parser.add_argument("--checkpoint-barrier-dir", type=Path)
    parser.add_argument("--checkpoint-barrier-interval", type=int)
    parser.add_argument("--checkpoint-barrier-timeout", type=float, default=3600.0)
    args = parser.parse_args()
    if args.steps < 1 or args.groups_per_step < 1:
        parser.error("steps and groups-per-step must be positive")
    if not 2 <= args.shared_return_replicas <= 32:
        parser.error("shared-return replicas must be between 2 and 32")
    if args.credit_estimator == "shared_return" and args.inference_config is None:
        parser.error("shared-return credit requires --inference-config")
    if args.actor == "vllm" and not args.base_url:
        parser.error("the vLLM actor requires at least one --base-url")
    if args.curriculum_offset < 0:
        parser.error("curriculum offset cannot be negative")
    if args.target_swap_sender_retries < 0:
        parser.error("target-swap sender retries cannot be negative")
    if args.horizon is not None and args.horizon < 1:
        parser.error("horizon must be positive")
    if args.rollout_only and args.steps != 1:
        parser.error("rollout-only diagnostics require exactly one controller step")
    if args.rollout_only and args.resume_existing_progress:
        parser.error("rollout-only diagnostics cannot resume optimizer progress")
    if args.ordinary_focused_agent is not None and not (
        args.rollout_only
        and args.production_plan is None
        and args.scenario_source == "ordinary"
        and args.groups_per_step == 1
    ):
        parser.error(
            "--ordinary-focused-agent requires a one-group, non-production, "
            "ordinary rollout-only diagnostic"
        )
    if (args.checkpoint_barrier_dir is None) != (args.checkpoint_barrier_interval is None):
        parser.error("checkpoint barrier directory and interval must be provided together")
    if args.checkpoint_barrier_interval is not None and (
        args.checkpoint_barrier_interval < 1 or args.steps % args.checkpoint_barrier_interval
    ):
        parser.error("checkpoint barrier interval must positively divide controller steps")
    if args.checkpoint_barrier_timeout <= 0:
        parser.error("checkpoint barrier timeout must be positive")
    if args.checkpoint_barrier_dir is not None and args.production_plan is None:
        parser.error("checkpoint evaluation barriers require an immutable production plan")
    if args.production_plan is None and args.opponent_revision is None:
        parser.error("--opponent-revision is required without --production-plan")
    if (args.opponent_adapter_path is None) != (args.opponent_adapter_sha256 is None):
        parser.error("opponent adapter path and SHA-256 must be provided together")
    if args.production_plan is not None and args.opponent_adapter_path is not None:
        parser.error("production plans own opponent adapter bindings")
    if args.production_plan is not None:
        if args.credit_estimator != "shared_return":
            parser.error("production plans require shared terminal return")
        if args.task_data_version != "v4":
            parser.error("production plans require --task-data-version v4")
        if args.actor != "vllm":
            parser.error("the optimized production plan currently requires the vLLM actor")
        if args.async_rescore_dir is None:
            parser.error("production plans require --async-rescore-dir")
        if args.async_rescore_timeout <= 0:
            parser.error("async rescore timeout must be positive")
        if args.shared_return_trainable_phase or args.shared_return_all_turns:
            parser.error("production plans own the immutable trainable span selection")
    repository_root = Path(__file__).resolve().parents[3]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if args.source_commit != actual_commit:
        parser.error(f"source commit {args.source_commit} does not match checked-out {actual_commit}")

    with args.trainer_config.open("rb") as handle:
        config = TrainerConfig.model_validate(tomli.load(handle))
    try:
        initial_policy_adapters = load_initial_policy_adapter_manifest(
            args.initial_policy_adapter_manifest
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if initial_policy_adapters:
        if config.model.lora is None:
            parser.error("distinct policy warm starts require trainer LoRA")
        expected_paths = {
            f"run_blue_{index}": initial_policy_adapters[f"blue-{index}"][0]
            for index in range(4)
        }
        expected_hashes = {
            f"run_blue_{index}": initial_policy_adapters[f"blue-{index}"][1]
            for index in range(4)
        }
        if config.model.lora.initial_adapter_paths_by_run != expected_paths:
            parser.error("trainer per-run warm-start paths do not match the controller manifest")
        if config.model.lora.initial_adapter_sha256_by_run != expected_hashes:
            parser.error("trainer per-run warm-start hashes do not match the controller manifest")
    if config.max_steps is not None:
        parser.error(
            "Swarm multi-run trainer max_steps must be omitted because it counts "
            "packing slices, not controller policy updates"
        )
    if config.max_concurrent_runs != 4:
        parser.error("Swarm live RL requires exactly four isolated trainer runs")
    production_plan = None
    opponent_runtime_paths: dict[str, Path | None] = {}
    scenario_schedule = None
    opponent_schedule = None
    if args.production_plan is not None:
        production_plan, opponent_runtime_paths = load_production_plan(args.production_plan)
        if args.groups_per_step != production_plan.groups_per_update:
            parser.error("groups-per-step must match the immutable production plan")
        total_groups = args.steps * args.groups_per_step
        try:
            scenario_schedule = production_plan.curriculum_schedule(steps=args.steps)
        except ValueError as error:
            parser.error(str(error))
        opponent_schedule = production_plan.opponent_pool.schedule(total_groups)
    shared_return_spec = None
    if args.credit_estimator == "shared_return":
        shared_return_spec = SharedReturnSpec(
            (
                production_plan.shared_return_replicas
                if production_plan is not None
                else args.shared_return_replicas
            ),
            trainable_phases=(
                ("ACT",)
                if production_plan is not None
                and args.shared_return_credit_assignment == "focused_agent"
                else production_plan.trainable_phases
                if production_plan is not None
                else tuple(args.shared_return_trainable_phase) or ("BROADCAST",)
            ),
            trainable_turn_offsets=(
                production_plan.trainable_turn_offsets
                if production_plan is not None
                else None
                if args.shared_return_all_turns
                else (0,)
            ),
            credit_assignment=args.shared_return_credit_assignment,
            baseline=(
                production_plan.shared_return_baseline
                if production_plan is not None
                else "leave_one_out_mean"
            ),
            action_prompt_profile=(
                production_plan.action_prompt_profile
                if production_plan is not None
                else args.shared_return_action_prompt_profile
            ),
            paired_contrast_centering=(
                production_plan.paired_contrast_centering
                if production_plan is not None
                else "replica_mean"
            ),
        )
    if args.target_swap_sender_retries and (
        shared_return_spec is None
        or shared_return_spec.baseline not in {
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        }
    ):
        parser.error("target-swap sender retries require a paired target-swap baseline")
    data_binding = resolve_task_data_binding(args.data_dir, args.task_data_version)
    base_urls = tuple(args.base_url)
    key = signing_key(args.output_dir / "control" / "supervisor.key")
    initial_revision = args.initial_policy_revision
    initial_adapter_sha256 = sha256_file(args.initial_adapter / "adapter_model.safetensors")
    result_rows: list[dict[str, object]] = []
    start_step = 0
    resume_paths: dict[str, Path] = {}
    if args.resume_existing_progress:
        result_rows = load_resume_progress(args.output_dir, total_steps=args.steps)
        start_step = len(result_rows)
        resumed_hashes = {
            str(name): str(value)
            for name, value in dict(result_rows[-1]["policy_adapter_sha256"]).items()
        }
        resume_paths = resume_adapter_paths(
            args.output_dir,
            completed_steps=start_step,
            expected_sha256=resumed_hashes,
        )
    adapter_names = tuple(f"blue-{index}" for index in range(4)) + ("sft-opponent",)
    if production_plan is not None:
        adapter_names = tuple(f"blue-{index}" for index in range(4)) + tuple(
            snapshot.model_name
            for snapshot in production_plan.opponent_pool.snapshots
            if snapshot.adapter_sha256 is not None
        )
    if args.credit_estimator == "policy_replacement":
        adapter_names += (args.replacement_model_name,)
    if args.actor == "vllm":
        for name in tuple(f"blue-{index}" for index in range(4)):
            await replace_adapter(
                base_urls,
                name,
                resume_paths[name]
                if args.resume_existing_progress
                else initial_policy_adapters[name][0]
                if initial_policy_adapters
                else args.initial_adapter,
            )
        if production_plan is None:
            await replace_adapter(base_urls, "sft-opponent", args.initial_adapter)
            if args.opponent_adapter_path is not None:
                opponent_adapter_file = (
                    args.opponent_adapter_path / "adapter_model.safetensors"
                )
                actual_opponent_sha256 = sha256_file(opponent_adapter_file)
                if actual_opponent_sha256 != args.opponent_adapter_sha256:
                    raise ValueError(
                        "opponent adapter checksum mismatch: "
                        f"{actual_opponent_sha256}"
                    )
                await replace_adapter(
                    base_urls,
                    args.opponent_model_name,
                    args.opponent_adapter_path,
                )
        else:
            for snapshot in production_plan.opponent_pool.snapshots:
                path = opponent_runtime_paths[snapshot.opponent_id]
                if path is None:
                    continue
                adapter_file = path / "adapter_model.safetensors"
                if not adapter_file.is_file():
                    raise FileNotFoundError(f"missing opponent adapter for {snapshot.opponent_id}: {adapter_file}")
                actual_sha256 = sha256_file(adapter_file)
                if actual_sha256 != snapshot.adapter_sha256:
                    raise ValueError(f"opponent adapter checksum mismatch for {snapshot.opponent_id}: {actual_sha256}")
                await replace_adapter(base_urls, snapshot.model_name, path)
        generator_context = VLLMChoiceGenerator(args.tokenizer)
    else:
        if args.inference_config is None:
            parser.error("the HF actor requires --inference-config")
        with args.inference_config.open("rb") as handle:
            actor_config = tomli.load(handle)
        expected_actor = {
            "version": ("arena-prime-choice-actor-v1" if args.actor == "prime" else "arena-hf-choice-actor-v1"),
            "model": args.tokenizer,
            "dtype": "bfloat16",
            "device": "cuda",
            "max_tokens": 128,
            "lm_head": "bf16xbf16-to-fp32",
            "sampling": "temperature-1-constrained-multinomial",
        }
        attention = actor_config.pop("attention", None)
        use_kv_cache = actor_config.pop("use_kv_cache", None)
        allowed_attention = (
            {"sdpa"}
            if args.actor == "prime"
            else {
                "flash_attention_2",
                "sdpa",
            }
        )
        if attention not in allowed_attention:
            raise ValueError("HF actor attention must be an audited FA2 or SDPA path")
        if not isinstance(use_kv_cache, bool):
            raise ValueError("HF actor use_kv_cache must be explicit")
        if args.actor == "prime" and use_kv_cache:
            raise ValueError("Prime actor requires full-prefix generation")
        if actor_config != expected_actor:
            raise ValueError("HF actor config does not match the audited implementation")
        generator_context = HFChoiceGenerator(
            actor_config["model"],
            args.initial_adapter,
            adapter_names=adapter_names,
            attention=attention,
            device=actor_config["device"],
            max_tokens=actor_config["max_tokens"],
            use_kv_cache=use_kv_cache,
            prime_model_config=(config.model if args.actor == "prime" else None),
            prime_actor_state_dir=(args.output_dir / "actor_state" if args.actor == "prime" else None),
            prime_matmul_precision=config.matmul_precision,
        )

    bindings = tuple(
        AgentPolicy(
            f"{team.lower()}-{index}",
            team,
            f"blue-policy-{index}" if team == "BLUE" else "red-opponent",
            team == "BLUE",
        )
        for team in ("BLUE", "RED")
        for index in range(4)
    )
    routes = tuple(PolicyRunRoute(f"blue-policy-{index}", f"run_blue_{index}") for index in range(4))
    trace = args.output_dir / "audit" / "admission.jsonl"
    message_evidence_trace = args.output_dir / "audit" / "message_credit_evidence.jsonl"
    shared_return_evidence_trace = args.output_dir / "audit" / "shared_return_evidence.jsonl"
    async_queue = None
    async_rescorer = None
    if production_plan is not None:
        async_queue = AtomicAsyncTrainingQueue(
            capacity=production_plan.rollout_queue_capacity,
            audit_path=args.output_dir / "audit" / "async_admission.jsonl",
            allowed_backend_calibrations=frozenset(
                {
                    (
                        production_plan.backend.name,
                        production_plan.backend.version,
                        production_plan.backend.kernel_config_sha256,
                        production_plan.backend.calibration_sha256,
                    )
                }
            ),
            allowed_constraint_sha256s=frozenset(
                {
                    protocol_constraint_sha256("BROADCAST"),
                    protocol_constraint_sha256("ACT"),
                }
            ),
            limits=production_plan.admission_limits,
        )
        assert args.async_rescore_dir is not None
        async_rescorer = FilesystemCurrentPolicyRescorer(
            args.async_rescore_dir,
            timeout=args.async_rescore_timeout,
        )
    curriculum = None
    if args.scenario_source == "curriculum" or production_plan is not None:
        manifest_path = args.data_dir / data_binding.curriculum_manifest(args.curriculum_split)
        curriculum = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not curriculum.get("pairs"):
            raise ValueError(f"empty curriculum manifest: {manifest_path}")
    adapted_stages: set[int] = set()
    stage_start_steps: tuple[int, ...] = ()
    if production_plan is not None and production_plan.adaptive_curriculum is not None:
        assert scenario_schedule is not None and curriculum is not None
        stage_start_steps = tuple(
            sum(stage.updates for stage in production_plan.stages[:index])
            for index in range(len(production_plan.stages))
        )
        for stage_index, stage_start in enumerate(stage_start_steps[1:], start=1):
            if stage_start > start_step:
                break
            scenario_schedule = adapt_next_curriculum_stage(
                scenario_schedule,
                production_plan=production_plan,
                curriculum=curriculum,
                progress=result_rows,
                stage_index=stage_index,
                output_dir=args.output_dir,
            )
            adapted_stages.add(stage_index)
    async with generator_context as generator:
        if args.resume_existing_progress:
            policy_adapter_sha256 = {
                str(name): str(value)
                for name, value in dict(result_rows[-1]["policy_adapter_sha256"]).items()
            }
            policy_revisions = dict(policy_adapter_sha256)
        else:
            policy_revisions = {
                f"blue-{index}": (
                    initial_policy_adapters[f"blue-{index}"][2]
                    if initial_policy_adapters
                    else initial_revision
                )
                for index in range(4)
            }
            policy_adapter_sha256 = {
                f"blue-{index}": (
                    initial_policy_adapters[f"blue-{index}"][1]
                    if initial_policy_adapters
                    else initial_adapter_sha256
                )
                for index in range(4)
            }
        if args.checkpoint_barrier_dir is not None and not args.resume_existing_progress:
            assert production_plan is not None
            await checkpoint_barrier(
                args.checkpoint_barrier_dir,
                step=0,
                run_id=args.run_id,
                plan_sha256=production_plan.sha256,
                policy_adapter_sha256=policy_adapter_sha256,
                policy_revision=canonical_sha256(policy_revisions),
                timeout=args.checkpoint_barrier_timeout,
            )
        for step in range(start_step, args.steps):
            if (
                production_plan is not None
                and production_plan.adaptive_curriculum is not None
                and step in stage_start_steps[1:]
            ):
                assert scenario_schedule is not None and curriculum is not None
                stage_index = stage_start_steps.index(step)
                if stage_index not in adapted_stages:
                    scenario_schedule = adapt_next_curriculum_stage(
                        scenario_schedule,
                        production_plan=production_plan,
                        curriculum=curriculum,
                        progress=result_rows,
                        stage_index=stage_index,
                        output_dir=args.output_dir,
                    )
                    adapted_stages.add(stage_index)
            if production_plan is not None:
                assert args.async_rescore_dir is not None
                write_current_snapshot_manifest(
                    args.async_rescore_dir / "current_snapshots.json",
                    plan_sha256=production_plan.sha256,
                    snapshots=tuple(
                        PolicySnapshot(
                            policy_id=f"blue-policy-{index}",
                            revision=policy_revisions[f"blue-{index}"],
                            adapter_sha256=policy_adapter_sha256[f"blue-{index}"],
                            update_index=step,
                            trainable=True,
                        )
                        for index in range(4)
                    ),
                )
            blue_policies = tuple(
                PolicyEndpoint(
                    f"blue-policy-{index}",
                    policy_revisions[f"blue-{index}"],
                    f"blue-{index}",
                    base_urls,
                )
                for index in range(4)
            )
            routed_groups = []
            step_groups = []
            for group_index in range(args.groups_per_step):
                ordinal = step * args.groups_per_step + group_index
                scheduled_opponent = opponent_schedule[ordinal] if opponent_schedule is not None else None
                if scheduled_opponent is not None and scheduled_opponent.family == "current":
                    scheduled_opponent = replace(
                        scheduled_opponent,
                        model_name="blue-0",
                        revision=policy_revisions["blue-0"],
                        adapter_sha256=policy_adapter_sha256["blue-0"],
                        update_index=step,
                    )
                opponent_revision = (
                    scheduled_opponent.revision if scheduled_opponent is not None else args.opponent_revision
                )
                assert opponent_revision is not None
                opponent_model_name = (
                    scheduled_opponent.model_name
                    if scheduled_opponent is not None
                    else args.opponent_model_name
                )
                policies = blue_policies + (
                    PolicyEndpoint(
                        "red-opponent",
                        opponent_revision,
                        opponent_model_name,
                        base_urls,
                    ),
                )
                if args.credit_estimator == "policy_replacement":
                    assert args.replacement_revision is not None
                    policies += (
                        PolicyEndpoint(
                            args.replacement_policy_id,
                            args.replacement_revision,
                            args.replacement_model_name,
                            base_urls,
                        ),
                    )
                initial_state = None
                scenario_metadata = {"source": "ordinary"}
                sampling_namespace = None
                seed = args.seed_base + step * 10_000 + group_index
                size = args.size
                horizon = args.horizon or 2
                assignment = scenario_schedule[ordinal] if scenario_schedule is not None else None
                if assignment is not None and assignment.kind == "ordinary":
                    assert production_plan is not None
                    assert assignment.ordinary_seed is not None
                    seed = assignment.ordinary_seed
                    size = (
                        assignment.ordinary_size
                        or production_plan.ordinary_sizes[ordinal % len(production_plan.ordinary_sizes)]
                    )
                    horizon = (
                        assignment.ordinary_horizon
                        or production_plan.ordinary_horizons[ordinal % len(production_plan.ordinary_horizons)]
                    )
                    scenario_metadata = {
                        "source": "ordinary",
                        "schedule_ordinal": ordinal,
                        "seed": seed,
                        "size": size,
                        "scheduled_horizon": horizon,
                        "curriculum_stage": assignment.stage,
                    }
                elif curriculum is not None:
                    if assignment is not None:
                        kind = assignment.kind
                        assert kind in {"critical", "decoy"}
                        assert assignment.pair_index is not None
                        pair_index = assignment.pair_index
                    elif args.curriculum_kind == "alternating":
                        kind = "critical" if ordinal % 2 == 0 else "decoy"
                        pair_index = (args.curriculum_offset + ordinal) // 2
                    else:
                        kind = args.curriculum_kind
                        pair_index = args.curriculum_offset + ordinal
                    pair = curriculum["pairs"][pair_index % len(curriculum["pairs"])]
                    reconstruct_scenario = (
                        reconstruct_v4_scenario if args.task_data_version == "v4" else reconstruct_v3_scenario
                    )
                    scenario = reconstruct_scenario(pair[kind])
                    seed = scenario.seed
                    size = scenario.size
                    horizon = args.horizon or scenario.horizon
                    world_metadata = {}
                    if args.task_data_version == "v4":
                        selected_world = assignment.handoff_world if assignment is not None else None
                        if selected_world is None:
                            world_index = pair_index % len(scenario.worlds)
                        else:
                            world_index = next(
                                index
                                for index, candidate in enumerate(scenario.worlds)
                                if candidate.label == selected_world
                            )
                        world = scenario.worlds[world_index]
                        if args.horizon is None and assignment is not None:
                            remaining_turns = assignment.handoff_remaining_turns
                            if remaining_turns is not None:
                                horizon = world.state.turn + remaining_turns
                        initial_state = world.state
                        world_metadata = {
                            "world": world.label,
                            "active_target": world.active_target,
                            "target": world.active_target,
                            "candidate_targets": list(scenario.candidate_targets),
                            "state_sha256": pair[kind]["worlds"][world_index]["state_sha256"],
                            "scheduled_horizon": horizon,
                        }
                    else:
                        initial_state = scenario.state
                        world_metadata = {
                            "target": scenario.target,
                            "state_sha256": pair[kind]["state_sha256"],
                        }
                    scenario_metadata = {
                        "source": "curriculum",
                        "split": args.curriculum_split,
                        "pair_index": pair_index % len(curriculum["pairs"]),
                        "kind": scenario.kind,
                        "seed": scenario.seed,
                        "sender": scenario.sender,
                        "receiver": scenario.receiver,
                        "minimum_certified_advantage": scenario.minimum_advantage,
                        "schedule_ordinal": ordinal,
                        "curriculum_stage": assignment.stage if assignment is not None else None,
                        "handoff_focus_role": (
                            assignment.handoff_focus_role if assignment is not None else "receiver"
                        ),
                        "handoff_trainable_turn_offsets": (
                            assignment.handoff_trainable_turn_offsets
                            if assignment is not None
                            else None
                        ),
                        **world_metadata,
                    }
                if scenario_metadata["source"] == "ordinary":
                    scenario_metadata.update(
                        {
                            "seed": seed,
                            "size": size,
                            "scheduled_horizon": horizon,
                        }
                    )
                game_id = f"{args.run_id}:step-{step}:group-{group_index}:{scenario_metadata['source']}:{seed}"
                fallback_pair_index = (
                    int(scenario_metadata["pair_index"])
                    if assignment is None and curriculum is not None and args.curriculum_kind == "alternating"
                    else None
                )
                sampling_namespace = scenario_sampling_namespace(
                    assignment,
                    run_id=args.run_id,
                    step=step,
                    fallback_pair_index=fallback_pair_index,
                )
                if sampling_namespace is not None:
                    scenario_metadata["sampling_namespace"] = sampling_namespace
                scenario_metadata["opponent"] = (
                    asdict(scheduled_opponent)
                    if scheduled_opponent is not None
                    else {
                        "opponent_id": f"{args.opponent_family}-opponent",
                        "family": args.opponent_family,
                        "model_name": args.opponent_model_name,
                        "revision": opponent_revision,
                    }
                )
                episode_config = EpisodeConfig(
                    horizon=horizon,
                    communication_cost=0.0,
                    invalid_broadcast_cost=0.0,
                    invalid_action_cost=0.0,
                )
                group_shared_return_spec = shared_return_spec
                focused_agent = None
                if (
                    shared_return_spec is not None
                    and shared_return_spec.credit_assignment == "focused_agent"
                ):
                    if scenario_metadata["source"] == "curriculum":
                        focus_role = str(scenario_metadata["handoff_focus_role"])
                        if focus_role not in {"sender", "receiver"}:
                            raise ValueError(f"unknown handoff focus role: {focus_role}")
                        focused_agent = str(scenario_metadata[focus_role])
                        focused_phase = "BROADCAST" if focus_role == "sender" else "ACT"
                        focused_turn_offsets = (
                            assignment.handoff_trainable_turn_offsets
                            if assignment is not None
                            and assignment.handoff_trainable_turn_offsets is not None
                            else (0,)
                        )
                    else:
                        focused_agent = (
                            args.ordinary_focused_agent
                            or f"blue-{group_index % 4}"
                        )
                        focused_phase = "ACT"
                        focused_turn_offsets = shared_return_spec.trainable_turn_offsets
                    if production_plan is not None and focused_phase not in production_plan.trainable_phases:
                        raise ValueError(
                            f"production plan does not allow scheduled focused phase {focused_phase}"
                        )
                    group_shared_return_spec = replace(
                        shared_return_spec,
                        trainable_phases=(focused_phase,),
                        trainable_turn_offsets=focused_turn_offsets,
                    )
                    scenario_metadata["focused_agent"] = focused_agent
                    scenario_metadata["focused_phase"] = focused_phase
                if (
                    group_shared_return_spec is not None
                    and production_plan is not None
                    and assignment is not None
                    and assignment.kind == "decoy"
                    and production_plan.decoy_shared_return_baseline is not None
                ):
                    group_shared_return_spec = replace(
                        group_shared_return_spec,
                        baseline=production_plan.decoy_shared_return_baseline,
                    )
                if (
                    group_shared_return_spec is not None
                    and group_shared_return_spec.baseline in {
                        "paired_target_swap",
                        "paired_receiver_target_swap",
                        "paired_receiver_target_swap_challenge",
                    }
                    and scenario_metadata["source"] == "ordinary"
                ):
                    group_shared_return_spec = replace(
                        group_shared_return_spec,
                        baseline="leave_one_out_mean",
                        paired_contrast_centering="replica_mean",
                    )
                lock = run_lock(
                    args,
                    config,
                    data_binding=data_binding,
                    policy_revisions=policy_revisions,
                    shared_return_spec=group_shared_return_spec,
                    opponent=scheduled_opponent,
                    production_plan=production_plan,
                )
                if args.credit_estimator == "shared_return":
                    assert group_shared_return_spec is not None
                    sender_retry = 0
                    while True:
                        sender_sampling_namespace = (
                            None
                            if sender_retry == 0
                            else f"{sampling_namespace}:target-swap-sender-retry-{sender_retry}"
                        )
                        try:
                            group = await build_live_shared_return_group(
                                generator,
                                group_id=game_id,
                                seed=seed,
                                size=size,
                                config=episode_config,
                                spec=group_shared_return_spec,
                                bindings=bindings,
                                policies=policies,
                                run_lock_sha256=lock.sha256,
                                initial_state=initial_state,
                                sampling_namespace=sampling_namespace,
                                focused_agent=focused_agent,
                                message_drop_agent=(
                                    str(scenario_metadata["sender"])
                                    if group_shared_return_spec.baseline == "paired_message_drop"
                                    else None
                                ),
                                message_drop_turn=(
                                    initial_state.turn
                                    if group_shared_return_spec.baseline == "paired_message_drop"
                                    else None
                                ),
                                message_swap_agent=(
                                    str(scenario_metadata["sender"])
                                    if group_shared_return_spec.baseline
                                    in {
                                        "paired_target_swap",
                                        "paired_receiver_target_swap",
                                        "paired_receiver_target_swap_challenge",
                                    }
                                    else None
                                ),
                                message_swap_turn=(
                                    initial_state.turn
                                    if group_shared_return_spec.baseline
                                    in {
                                        "paired_target_swap",
                                        "paired_receiver_target_swap",
                                        "paired_receiver_target_swap_challenge",
                                    }
                                    else None
                                ),
                                message_swap_targets=(
                                    tuple(str(value) for value in scenario_metadata["candidate_targets"])
                                    if group_shared_return_spec.baseline
                                    in {
                                        "paired_target_swap",
                                        "paired_receiver_target_swap",
                                        "paired_receiver_target_swap_challenge",
                                    }
                                    else None
                                ),
                                message_swap_active_target=(
                                    str(scenario_metadata["active_target"])
                                    if group_shared_return_spec.baseline
                                    in {
                                        "paired_target_swap",
                                        "paired_receiver_target_swap",
                                        "paired_receiver_target_swap_challenge",
                                    }
                                    else None
                                ),
                                message_swap_sender_sampling_namespace=sender_sampling_namespace,
                            )
                        except TargetSwapIneligibleError as error:
                            if sender_retry >= args.target_swap_sender_retries:
                                raise
                            sender_retry += 1
                            append_hash_chained_record(
                                args.output_dir / "audit" / "target_swap_sender_retries.jsonl",
                                {
                                    "game_id": game_id,
                                    "reason": str(error),
                                    "retry": sender_retry,
                                    "sender": scenario_metadata["sender"],
                                    "sender_sampling_namespace": (
                                        f"{sampling_namespace}:target-swap-sender-retry-{sender_retry}"
                                    ),
                                    "step": step,
                                },
                            )
                            continue
                        break
                    scenario_metadata["target_swap_sender_retries"] = sender_retry
                    approvals = approve_shared_return_group(lock, group.evidence, bindings, "BLUE", key)
                    replica_routes = tuple(
                        route_approved_samples(
                            approval,
                            owned,
                            routes,
                            step=step,
                            signing_key=key,
                            trainer_parity_gate_sha256=lock.trainer_parity_gate_sha256,
                        )
                        for approval, owned in zip(
                            approvals,
                            group.owned_samples_by_replica,
                            strict=True,
                        )
                    )
                    routed_group = merge_routed_batch_groups(
                        replica_routes,
                        step=step,
                    )
                    if production_plan is None:
                        routed_groups.append(routed_group)
                    else:
                        assert scheduled_opponent is not None
                        assert async_queue is not None and async_rescorer is not None
                        opponent_artifact_sha256 = scheduled_opponent.adapter_sha256 or canonical_sha256(
                            {"base_model_revision": scheduled_opponent.revision}
                        )
                        behavior_snapshots = tuple(
                            PolicySnapshot(
                                policy_id=f"blue-policy-{index}",
                                revision=policy_revisions[f"blue-{index}"],
                                adapter_sha256=policy_adapter_sha256[f"blue-{index}"],
                                update_index=step,
                                trainable=True,
                            )
                            for index in range(4)
                        ) + (
                            PolicySnapshot(
                                policy_id="red-opponent",
                                revision=scheduled_opponent.revision,
                                adapter_sha256=opponent_artifact_sha256,
                                update_index=scheduled_opponent.update_index,
                                trainable=False,
                            ),
                        )
                        challenge_baseline = (
                            group.evidence.spec.baseline
                            == "paired_receiver_target_swap_challenge"
                        )
                        all_decisions = tuple(
                            decision
                            for replica in group.evidence.replicas
                            for decision in (
                                replica.swapped_decisions
                                if challenge_baseline
                                else replica.decisions
                            )
                        )
                        trainable_decision_ids = frozenset(
                            decision_id
                            for approval in approvals
                            for envelope in approval.envelopes
                            for decision_id in envelope.decision_ids
                        )
                        selected_decisions = tuple(
                            decision for decision in all_decisions if decision.decision_id in trainable_decision_ids
                        )
                        header = AsyncRolloutHeader(
                            rollout_id=game_id,
                            backend_name=production_plan.backend.name,
                            backend_version=production_plan.backend.version,
                            kernel_config_sha256=(production_plan.backend.kernel_config_sha256),
                            calibration_sha256=(production_plan.backend.calibration_sha256),
                            policy_snapshots=behavior_snapshots,
                        )
                        current_snapshots, current_logprobs = await async_rescorer.rescore(
                            rollout_id=game_id,
                            plan_sha256=production_plan.sha256,
                            behavior_snapshots=behavior_snapshots,
                            decisions=selected_decisions,
                        )
                        admission = async_queue.admit(
                            header=header,
                            decisions=all_decisions,
                            trainable_decision_ids=trainable_decision_ids,
                            trainable_branch=("message_swap" if challenge_baseline else "actual"),
                            current_snapshots=current_snapshots,
                            current_policy_logprobs=current_logprobs,
                            routed_batches=routed_group,
                            trainer_step=step,
                        )
                        if not admission.accepted:
                            raise RuntimeError(
                                "bounded async admission rejected the logical group: " + "; ".join(admission.reasons)
                            )
                    append_hash_chained_record(
                        shared_return_evidence_trace,
                        shared_return_evidence_payload(group.evidence),
                    )
                    append_hash_chained_record(
                        trace,
                        {
                            "approvals": [asdict(row) for row in approvals],
                            "credit_estimator": args.credit_estimator,
                            "group_id": group.evidence.group_id,
                            "initial_state_sha256": group.evidence.initial_state_sha256,
                            "spec": asdict(group.evidence.spec),
                            "focused_agent": group.evidence.focused_agent,
                            "replica_returns": [row.replay.terminal_return for row in group.evidence.replicas],
                            "dropped_returns": [
                                row.dropped_replay.terminal_return if row.dropped_replay is not None else None
                                for row in group.evidence.replicas
                            ],
                            "swapped_returns": [
                                row.swapped_replay.terminal_return if row.swapped_replay is not None else None
                                for row in group.evidence.replicas
                            ],
                        },
                    )
                    step_groups.append(
                        {
                            "game_id": game_id,
                            "credit_estimator": args.credit_estimator,
                            "replicas": [
                                {
                                    "game_id": replica.game_id,
                                    "return": replica.replay.terminal_return,
                                    "dropped_return": (
                                        replica.dropped_replay.terminal_return
                                        if replica.dropped_replay is not None
                                        else None
                                    ),
                                    "message_effect": (
                                        replica.replay.terminal_return - replica.dropped_replay.terminal_return
                                        if replica.dropped_replay is not None
                                        else None
                                    ),
                                    "swapped_return": (
                                        replica.swapped_replay.terminal_return
                                        if replica.swapped_replay is not None
                                        else None
                                    ),
                                    "semantic_effect": (
                                        replica.replay.terminal_return - replica.swapped_replay.terminal_return
                                        if replica.swapped_replay is not None
                                        else None
                                    ),
                                    "challenge_effect": (
                                        replica.swapped_replay.terminal_return
                                        - replica.replay.terminal_return
                                        if group.evidence.spec.baseline
                                        == "paired_receiver_target_swap_challenge"
                                        and replica.swapped_replay is not None
                                        else None
                                    ),
                                    "advantages": {
                                        envelope.agent_id: envelope.advantage for envelope in approval.envelopes
                                    },
                                    "focused_action": (
                                        dict(
                                            (
                                                replica.swapped_replay
                                                if group.evidence.spec.baseline
                                                == "paired_receiver_target_swap_challenge"
                                                else replica.replay
                                            ).turns[0].actions
                                        )[focused_agent].to_dict()
                                        if focused_agent is not None
                                        and scenario_metadata.get("focused_phase") == "ACT"
                                        else None
                                    ),
                                }
                                for replica, approval in zip(group.evidence.replicas, approvals, strict=True)
                            ],
                            "scenario": scenario_metadata,
                        }
                    )
                    continue
                if args.credit_estimator == "message_drop":
                    group = await build_live_message_credit_group(
                        generator,
                        game_id=game_id,
                        seed=seed,
                        size=size,
                        config=episode_config,
                        bindings=bindings,
                        policies=policies,
                        run_lock_sha256=lock.sha256,
                        initial_state=initial_state,
                        sampling_namespace=sampling_namespace,
                    )
                    approval = approve_message_credit_group(lock, group.evidence, bindings, "BLUE", key)
                    append_hash_chained_record(
                        message_evidence_trace,
                        message_credit_audit_record(
                            group.evidence,
                            approval,
                            scenario_metadata,
                        ),
                    )
                    counterfactual_returns = {row.replaced_agent: row.terminal_return for row in group.evidence.drops}
                else:
                    group = await build_live_credit_group(
                        generator,
                        game_id=game_id,
                        seed=seed,
                        size=size,
                        config=episode_config,
                        bindings=bindings,
                        policies=policies,
                        replacement_policy_id=args.replacement_policy_id,
                        run_lock_sha256=lock.sha256,
                        initial_state=initial_state,
                        sampling_namespace=sampling_namespace,
                    )
                    approval = approve_credit_group(lock, group.evidence, bindings, "BLUE", key)
                    counterfactual_returns = {
                        row.replaced_agent: row.terminal_return for row in group.evidence.replacements
                    }
                append_hash_chained_record(
                    trace,
                    {
                        "approval": asdict(approval),
                        "actual_return": group.evidence.actual.terminal_return,
                        "credit_estimator": args.credit_estimator,
                        "counterfactual_returns": counterfactual_returns,
                    },
                )
                routed_groups.append(
                    route_approved_samples(
                        approval,
                        group.owned_samples,
                        routes,
                        step=step,
                        signing_key=key,
                        trainer_parity_gate_sha256=lock.trainer_parity_gate_sha256,
                    )
                )
                step_groups.append(
                    {
                        "game_id": game_id,
                        "actual_return": group.evidence.actual.terminal_return,
                        "credit_estimator": args.credit_estimator,
                        "intervention_turn": (
                            group.evidence.intervention_turn if args.credit_estimator == "message_drop" else None
                        ),
                        "advantages": {row.agent_id: row.advantage for row in approval.envelopes},
                        "scenario": scenario_metadata,
                    }
                )
            if args.rollout_only:
                result_rows.append({"step": step, "groups": step_groups})
                diagnostic_path = args.output_dir / "live_rl_diagnostic.json"
                diagnostic_path.write_text(
                    json.dumps(result_rows, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(result_rows[-1], sort_keys=True))
                continue
            logical_update_nonzero_advantage = None
            if production_plan is not None and production_plan.monitor_logical_update_signal:
                logical_update_nonzero_advantage = logical_update_has_signal(step_groups)
                append_hash_chained_record(
                    args.output_dir / "audit" / "logical_update_signal.jsonl",
                    {
                        "step": step,
                        "nonzero_advantage": logical_update_nonzero_advantage,
                        "action": "observe_only_continue",
                    },
                )
            batches = (
                async_queue.pop_logical_update(
                    groups=args.groups_per_step,
                    trainer_step=step,
                )
                if async_queue is not None
                else merge_routed_batch_groups(tuple(routed_groups), step=step)
            )
            if production_plan is None:
                validate_single_trajectory_packing(
                    batches,
                    seq_len=config.model.seq_len,
                )
            else:
                validate_training_batch_lengths(
                    batches,
                    seq_len=config.model.seq_len,
                )
            await send_approved_batches(args.output_dir, batches)
            digests = await wait_for_policy_updates(
                args.output_dir,
                base_urls,
                expected_step=step + 1,
                timeout=args.update_timeout,
                hf_generator=(generator if isinstance(generator, HFChoiceGenerator) else None),
            )
            policy_revisions = digests
            policy_adapter_sha256 = dict(digests)
            policy_revision = canonical_sha256(policy_revisions)
            result_rows.append(
                {
                    "step": step,
                    "groups": step_groups,
                    "logical_update_nonzero_advantage": logical_update_nonzero_advantage,
                    "policy_adapter_sha256": digests,
                    "policy_revision": policy_revision,
                }
            )
            (args.output_dir / "live_rl_progress.json").write_text(
                json.dumps(result_rows, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result_rows[-1], sort_keys=True))
            completed_step = step + 1
            if args.checkpoint_barrier_dir is not None and completed_step % args.checkpoint_barrier_interval == 0:
                assert production_plan is not None
                await checkpoint_barrier(
                    args.checkpoint_barrier_dir,
                    step=completed_step,
                    run_id=args.run_id,
                    plan_sha256=production_plan.sha256,
                    policy_adapter_sha256=policy_adapter_sha256,
                    policy_revision=policy_revision,
                    timeout=args.checkpoint_barrier_timeout,
                )


if __name__ == "__main__":
    asyncio.run(main())
