from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import tomli
from swarm_ctf_eval.communication_curriculum import reconstruct_manifest_scenario
from swarm_ctf_eval.episode import EpisodeConfig
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
from swarm_ctf_eval.multi_policy_contract import AgentPolicy
from swarm_ctf_eval.prime_multi_run_router import (
    PolicyRunRoute,
    merge_routed_batch_groups,
    route_approved_samples,
    send_approved_batches,
    validate_single_trajectory_packing,
)
from swarm_ctf_eval.rl_v3 import RL_TASK_VERSION
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

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


async def replace_adapter(base_urls: tuple[str, ...], name: str, path: Path) -> None:
    """Atomically refresh one named LoRA and verify the server's registered path."""
    async with httpx.AsyncClient(timeout=120.0) as client:
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
        registries = await asyncio.gather(
            *(client.get(f"{base_url.rstrip('/')}/v1/models") for base_url in base_urls)
        )
    expected_path = str(path.resolve())
    for response in registries:
        response.raise_for_status()
        matches = [row for row in response.json()["data"] if row["id"] == name]
        if len(matches) != 1 or matches[0].get("root") != expected_path:
            raise RuntimeError(f"LoRA registry did not bind {name} to {expected_path}")


async def wait_for_policy_updates(
    output_dir: Path,
    base_urls: tuple[str, ...],
    *,
    expected_step: int,
    timeout: float,
    hf_generator: HFChoiceGenerator | None = None,
) -> dict[str, str]:
    paths = {
        f"blue-{index}": get_step_path(
            get_broadcast_dir(output_dir / f"run_blue_{index}"), expected_step
        )
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
    policy_revisions: dict[str, str],
    shared_return_spec: SharedReturnSpec | None,
) -> RunLock:
    index = json.loads((args.data_dir / "index.json").read_text(encoding="utf-8"))
    if config.rollout_parity_gate is None:
        raise ValueError("trainer config is missing the pre-step parity gate")
    frozen_revisions = [("red-opponent", args.opponent_revision)]
    replacement_policy_id = None
    if args.credit_estimator == "policy_replacement":
        if args.replacement_revision is None:
            raise ValueError("policy-replacement credit requires --replacement-revision")
        replacement_policy_id = args.replacement_policy_id
        frozen_revisions.append((args.replacement_policy_id, args.replacement_revision))
    return RunLock(
        args.run_id,
        args.source_commit,
        RL_TASK_VERSION,
        index["splits"]["train"]["sha256"],
        index["splits"]["development"]["sha256"],
        sha256_file(args.data_dir / "final_eval_design.json"),
        args.base_revision,
        tuple(
            (f"blue-policy-{index}", policy_revisions[f"blue-{index}"])
            for index in range(4)
        ),
        tuple(frozen_revisions),
        replacement_policy_id,
        "sft-opponent",
        args.opponent_revision,
        (
            protocol_constraint_sha256("BROADCAST"),
            protocol_constraint_sha256("ACT"),
        ),
        parity_gate_sha256(config.rollout_parity_gate),
        args.credit_estimator,
        shared_return_spec.sha256 if shared_return_spec is not None else None,
        sha256_file(args.trainer_config) if shared_return_spec is not None else None,
        sha256_file(args.inference_config) if shared_return_spec is not None else None,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded fail-closed four-policy Swarm Arena RL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument(
        "--actor",
        choices=("vllm", "hf"),
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
    parser.add_argument("--replacement-policy-id", default="sft-replacement")
    parser.add_argument("--replacement-model-name", default="sft-replacement")
    parser.add_argument("--replacement-revision")
    parser.add_argument("--opponent-revision", required=True)
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
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--update-timeout", type=float, default=600.0)
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
    if args.rollout_only and args.steps != 1:
        parser.error("rollout-only diagnostics require exactly one controller step")
    repository_root = Path(__file__).resolve().parents[3]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if args.source_commit != actual_commit:
        parser.error(
            f"source commit {args.source_commit} does not match checked-out {actual_commit}"
        )

    with args.trainer_config.open("rb") as handle:
        config = TrainerConfig.model_validate(tomli.load(handle))
    if config.max_steps is not None:
        parser.error(
            "Swarm multi-run trainer max_steps must be omitted because it counts "
            "packing slices, not controller policy updates"
        )
    if config.max_concurrent_runs != 4:
        parser.error("Swarm live RL requires exactly four isolated trainer runs")
    shared_return_spec = (
        SharedReturnSpec(args.shared_return_replicas)
        if args.credit_estimator == "shared_return"
        else None
    )
    base_urls = tuple(args.base_url)
    key = signing_key(args.output_dir / "control" / "supervisor.key")
    initial_revision = args.initial_policy_revision
    adapter_names = tuple(f"blue-{index}" for index in range(4)) + (
        "sft-opponent",
    )
    if args.credit_estimator == "policy_replacement":
        adapter_names += (args.replacement_model_name,)
    if args.actor == "vllm":
        for name in adapter_names:
            await replace_adapter(base_urls, name, args.initial_adapter)
        generator_context = VLLMChoiceGenerator(args.tokenizer)
    else:
        if args.inference_config is None:
            parser.error("the HF actor requires --inference-config")
        with args.inference_config.open("rb") as handle:
            actor_config = tomli.load(handle)
        expected_actor = {
            "version": "arena-hf-choice-actor-v1",
            "model": args.tokenizer,
            "attention": "flash_attention_2",
            "dtype": "bfloat16",
            "device": "cuda",
            "max_tokens": 128,
            "use_kv_cache": True,
            "lm_head": "bf16xbf16-to-fp32",
            "sampling": "temperature-1-constrained-multinomial",
        }
        if actor_config != expected_actor:
            raise ValueError("HF actor config does not match the audited implementation")
        generator_context = HFChoiceGenerator(
            actor_config["model"],
            args.initial_adapter,
            adapter_names=adapter_names,
            attention=actor_config["attention"],
            device=actor_config["device"],
            max_tokens=actor_config["max_tokens"],
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
    routes = tuple(
        PolicyRunRoute(f"blue-policy-{index}", f"run_blue_{index}")
        for index in range(4)
    )
    trace = args.output_dir / "audit" / "admission.jsonl"
    message_evidence_trace = (
        args.output_dir / "audit" / "message_credit_evidence.jsonl"
    )
    shared_return_evidence_trace = (
        args.output_dir / "audit" / "shared_return_evidence.jsonl"
    )
    result_rows = []
    curriculum = None
    if args.scenario_source == "curriculum":
        manifest_path = args.data_dir / f"{args.curriculum_split}.json"
        curriculum = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not curriculum.get("pairs"):
            raise ValueError(f"empty curriculum manifest: {manifest_path}")
    async with generator_context as generator:
        policy_revisions = {f"blue-{index}": initial_revision for index in range(4)}
        for step in range(args.steps):
            lock = run_lock(
                args,
                config,
                policy_revisions=policy_revisions,
                shared_return_spec=shared_return_spec,
            )
            policies = tuple(
                PolicyEndpoint(
                    f"blue-policy-{index}",
                    policy_revisions[f"blue-{index}"],
                    f"blue-{index}",
                    base_urls,
                )
                for index in range(4)
            ) + (
                PolicyEndpoint(
                    "red-opponent",
                    args.opponent_revision,
                    "sft-opponent",
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
            routed_groups = []
            step_groups = []
            for group_index in range(args.groups_per_step):
                ordinal = step * args.groups_per_step + group_index
                initial_state = None
                scenario_metadata = {"source": "ordinary"}
                sampling_namespace = None
                seed = args.seed_base + step * 10_000 + group_index
                size = args.size
                if curriculum is not None:
                    if args.curriculum_kind == "alternating":
                        kind = "critical" if ordinal % 2 == 0 else "decoy"
                        pair_index = (args.curriculum_offset + ordinal) // 2
                    else:
                        kind = args.curriculum_kind
                        pair_index = args.curriculum_offset + ordinal
                    pair = curriculum["pairs"][pair_index % len(curriculum["pairs"])]
                    scenario = reconstruct_manifest_scenario(pair[kind])
                    initial_state = scenario.state
                    seed = scenario.seed
                    size = scenario.size
                    scenario_metadata = {
                        "source": "curriculum",
                        "split": args.curriculum_split,
                        "pair_index": pair_index % len(curriculum["pairs"]),
                        "kind": scenario.kind,
                        "seed": scenario.seed,
                        "sender": scenario.sender,
                        "receiver": scenario.receiver,
                        "target": scenario.target,
                        "minimum_certified_advantage": scenario.minimum_advantage,
                        "state_sha256": pair[kind]["state_sha256"],
                    }
                game_id = (
                    f"{args.run_id}:step-{step}:group-{group_index}:"
                    f"{scenario_metadata['source']}:{seed}"
                )
                if curriculum is not None and args.curriculum_kind == "alternating":
                    sampling_namespace = (
                        f"{args.run_id}:step-{step}:pair-{scenario_metadata['pair_index']}"
                    )
                    scenario_metadata["sampling_namespace"] = sampling_namespace
                episode_config = EpisodeConfig(
                    horizon=args.horizon,
                    communication_cost=0.0,
                    invalid_broadcast_cost=0.0,
                    invalid_action_cost=0.0,
                )
                if args.credit_estimator == "shared_return":
                    assert shared_return_spec is not None
                    group = await build_live_shared_return_group(
                        generator,
                        group_id=game_id,
                        seed=seed,
                        size=size,
                        config=episode_config,
                        spec=shared_return_spec,
                        bindings=bindings,
                        policies=policies,
                        run_lock_sha256=lock.sha256,
                        initial_state=initial_state,
                        sampling_namespace=sampling_namespace,
                    )
                    approvals = approve_shared_return_group(
                        lock, group.evidence, bindings, "BLUE", key
                    )
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
                    routed_groups.append(
                        merge_routed_batch_groups(replica_routes, step=step)
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
                            "replica_returns": [
                                row.replay.terminal_return
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
                                    "advantage": approval.envelopes[0].advantage,
                                }
                                for replica, approval in zip(
                                    group.evidence.replicas, approvals, strict=True
                                )
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
                    approval = approve_message_credit_group(
                        lock, group.evidence, bindings, "BLUE", key
                    )
                    append_hash_chained_record(
                        message_evidence_trace,
                        message_credit_audit_record(
                            group.evidence,
                            approval,
                            scenario_metadata,
                        ),
                    )
                    counterfactual_returns = {
                        row.replaced_agent: row.terminal_return
                        for row in group.evidence.drops
                    }
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
                    approval = approve_credit_group(
                        lock, group.evidence, bindings, "BLUE", key
                    )
                    counterfactual_returns = {
                        row.replaced_agent: row.terminal_return
                        for row in group.evidence.replacements
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
                            group.evidence.intervention_turn
                            if args.credit_estimator == "message_drop"
                            else None
                        ),
                        "advantages": {
                            row.agent_id: row.advantage for row in approval.envelopes
                        },
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
            batches = merge_routed_batch_groups(tuple(routed_groups), step=step)
            validate_single_trajectory_packing(
                batches,
                seq_len=config.model.seq_len,
            )
            await send_approved_batches(args.output_dir, batches)
            digests = await wait_for_policy_updates(
                args.output_dir,
                base_urls,
                expected_step=step + 1,
                timeout=args.update_timeout,
                hf_generator=(
                    generator if isinstance(generator, HFChoiceGenerator) else None
                ),
            )
            policy_revisions = digests
            policy_revision = canonical_sha256(policy_revisions)
            result_rows.append(
                {
                    "step": step,
                    "groups": step_groups,
                    "policy_adapter_sha256": digests,
                    "policy_revision": policy_revision,
                }
            )
            (args.output_dir / "live_rl_progress.json").write_text(
                json.dumps(result_rows, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result_rows[-1], sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
