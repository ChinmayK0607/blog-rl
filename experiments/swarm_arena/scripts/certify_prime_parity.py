from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tomli
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.model import forward, setup_model
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.trainer.optim import setup_multi_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.rl.loss import shift_tensor_left
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.utils import setup_torch_distributed


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def adapter_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_run_config(output_dir: Path, run_id: str, model: str) -> None:
    control = output_dir / run_id / "control"
    control.mkdir(parents=True)
    control.joinpath("orch.toml").write_text(
        f'''batch_size = 1
group_size = 1
seq_len = 4096
max_steps = 1

[model]
name = "{model}"

[model.lora]
rank = 16
alpha = 32

[optim]
lr = 0.00001

[renderer]
name = "qwen3"
enable_thinking = false

[[train.env]]
id = "reverse-text"
''',
        encoding="utf-8",
    )


def constrained_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
    allowed_rows: list[list[int]],
) -> torch.Tensor:
    selected_rows = []
    for offset, allowed in enumerate(allowed_rows):
        token_position = prompt_length + offset
        prediction = logits[0, token_position - 1].float()
        legal_ids = torch.tensor(allowed, device=prediction.device, dtype=torch.long)
        selected_rows.append(prediction[input_ids[0, token_position]] - torch.logsumexp(prediction[legal_ids], dim=0))
    return torch.stack(selected_rows)


def constrained_distribution_logprobs(
    logits: torch.Tensor,
    prompt_length: int,
    allowed_rows: list[list[int]],
) -> list[dict[int, float]]:
    distributions = []
    for offset, allowed in enumerate(allowed_rows):
        prediction = logits[0, prompt_length + offset - 1].float()
        legal_ids = torch.tensor(allowed, device=prediction.device, dtype=torch.long)
        legal_logprobs = prediction[legal_ids] - torch.logsumexp(prediction[legal_ids], dim=0)
        distributions.append(dict(zip(allowed, legal_logprobs.detach().cpu().tolist(), strict=True)))
    return distributions


def prepare_sample(
    sample: dict,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], int]:
    prompt_ids = list(sample["prompt_ids"])
    completion_ids = list(sample["completion_ids"])
    allowed_rows = [list(row) for row in sample["allowed_token_ids"]]
    inference_logprobs = torch.tensor(sample["completion_logprobs"], dtype=torch.float32)
    if not (len(completion_ids) == len(allowed_rows) == len(inference_logprobs)):
        raise ValueError("probe completion fields have inconsistent lengths")
    token_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(token_ids.shape[1], device=device).unsqueeze(0)
    return token_ids, position_ids, inference_logprobs, allowed_rows, len(prompt_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify vLLM-to-Prime constrained logprob parity.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--trainer-config", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-mean-logprob-error", type=float)
    parser.add_argument("--max-p99-logprob-error", type=float)
    parser.add_argument("--max-probability-error", type=float)
    parser.add_argument("--max-p99-probability-error", type=float)
    parser.add_argument("--probability-tail-threshold", type=float, default=0.05)
    parser.add_argument("--max-probability-tail-fraction", type=float)
    parser.add_argument("--max-mean-mismatch-kl", type=float, default=0.0005)
    parser.add_argument("--max-mismatch-kl", type=float)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse parity output directory: {args.output_dir}")
    for index in range(4):
        write_run_config(args.output_dir, f"run_blue_{index}", args.model)
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    samples = list(probe["samples"])
    if not samples:
        raise ValueError("probe contains no samples")

    setup_torch_distributed()
    trainer_config_sha256 = None
    if args.trainer_config is None:
        config = TrainerConfig.model_validate(
            {
                "output_dir": args.output_dir,
                "max_concurrent_runs": 4,
                "model": {
                    "name": args.model,
                    "seq_len": 4096,
                    "attn": "flash_attention_2",
                    "impl": "auto",
                    "lora": {
                        "rank": 16,
                        "alpha": 32,
                        "dropout": 0.0,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "initial_adapter_path": args.adapter,
                        "initial_adapter_sha256": args.adapter_sha256,
                    },
                },
                "rollout_parity_gate": {
                    "max_mean_logprob_error": args.max_mean_logprob_error,
                    "max_p99_logprob_error": args.max_p99_logprob_error,
                    "max_probability_error": args.max_probability_error,
                    "max_p99_probability_error": args.max_p99_probability_error,
                    "probability_tail_threshold": args.probability_tail_threshold,
                    "max_probability_tail_fraction": (args.max_probability_tail_fraction),
                    "max_mean_mismatch_kl": args.max_mean_mismatch_kl,
                    "max_mismatch_kl": args.max_mismatch_kl,
                },
            }
        )
    else:
        with args.trainer_config.open("rb") as handle:
            config = TrainerConfig.model_validate(tomli.load(handle))
        config = config.model_copy(update={"output_dir": args.output_dir})
        trainer_config_sha256 = sha256_file(args.trainer_config)
        if config.max_concurrent_runs != 4:
            raise ValueError("parity certification requires exactly four policy slots")
        if config.model.name != args.model:
            raise ValueError("trainer config model does not match --model")
        if config.model.lora is None:
            raise ValueError("trainer config must enable LoRA")
        if config.model.lora.initial_adapter_path != args.adapter:
            raise ValueError("trainer config adapter path does not match --adapter")
        if config.model.lora.initial_adapter_sha256 != args.adapter_sha256:
            raise ValueError("trainer config adapter digest does not match --adapter-sha256")
    parity_gate = config.rollout_parity_gate
    if parity_gate is None:
        raise ValueError("trainer config is missing the rollout parity gate")
    expected_gate = {
        "max_mean_logprob_error": args.max_mean_logprob_error,
        "max_p99_logprob_error": args.max_p99_logprob_error,
        "max_probability_error": args.max_probability_error,
        "max_p99_probability_error": args.max_p99_probability_error,
        "probability_tail_threshold": args.probability_tail_threshold,
        "max_probability_tail_fraction": args.max_probability_tail_fraction,
        "max_mean_mismatch_kl": args.max_mean_mismatch_kl,
        "max_mismatch_kl": args.max_mismatch_kl,
    }
    if parity_gate.model_dump(mode="json") != expected_gate:
        raise ValueError("trainer config parity gate does not match certificate thresholds")
    trainer_parity_gate_sha256 = canonical_sha256(expected_gate)
    device = torch.device("cuda", 0)
    manager = setup_multi_run_manager(args.output_dir, 4, device, config.model.lora)
    parallel_dims = get_parallel_dims(config.model)
    model = setup_model(config.model, parallel_dims)
    optimizer = setup_multi_optimizer(config.optim, parallel_dims)
    manager.discover_runs()
    manager.synchronize_state()
    if set(manager.id_2_idx) != {f"run_blue_{index}" for index in range(4)}:
        raise RuntimeError(f"four policy runs were not discovered: {manager.id_2_idx}")

    slot_digests_before = {
        run_id: adapter_digest(manager.get_state_dict_for_run(index))
        for run_id, index in sorted(manager.id_2_idx.items())
    }
    if len(set(slot_digests_before.values())) != 1:
        raise RuntimeError("four policy slots did not start from identical pinned adapters")

    all_inference = []
    all_trainer = []
    all_branching = []
    sample_summaries = []
    token_summaries = []
    distribution_summaries = []
    model.eval()
    for sample_index, sample in enumerate(samples):
        policy_slot = int(sample.get("policy_slot", sample_index % 4))
        if policy_slot not in range(4):
            raise ValueError(f"parity sample has an invalid policy slot: {policy_slot}")
        run_id = f"run_blue_{policy_slot}"
        slot = manager.id_2_idx[run_id]
        token_ids, position_ids, inference_logprobs, allowed_rows, prompt_length = prepare_sample(sample, device=device)
        labels = shift_tensor_left(token_ids)
        lora_num_tokens = torch.zeros(4, dtype=torch.int32, device=device)
        lora_num_tokens[slot] = token_ids.shape[1]
        set_lora_num_tokens(lora_num_tokens)
        with torch.no_grad():
            output = forward(
                model,
                token_ids,
                position_ids,
                labels=labels,
                temperature=torch.ones_like(token_ids, dtype=torch.float32),
            )
            logits = output.get("logits")
            if logits is None:
                raise RuntimeError("parity requires the unfused trainer LM head logits")
            trainer_logprobs = constrained_logprobs(logits, token_ids, prompt_length, allowed_rows).cpu()
            trainer_distributions = constrained_distribution_logprobs(logits, prompt_length, allowed_rows)
        serving_distributions = sample.get("serving_allowed_logprobs")
        if serving_distributions is not None:
            if len(serving_distributions) != len(allowed_rows):
                raise ValueError("serving distribution rows do not match allowed rows")
            for token_offset, (allowed, serving_row, trainer_row) in enumerate(
                zip(
                    allowed_rows,
                    serving_distributions,
                    trainer_distributions,
                    strict=True,
                )
            ):
                serving = {int(token_id): float(value) for token_id, value in serving_row}
                if len(serving) != len(serving_row) or any(token_id not in allowed for token_id in serving):
                    raise ValueError("invalid serving constrained distribution row")
                if set(serving) != set(allowed):
                    continue
                serving_logprobs = torch.tensor([serving[token_id] for token_id in allowed], dtype=torch.float64)
                trainer_row_logprobs = torch.tensor(
                    [trainer_row[token_id] for token_id in allowed], dtype=torch.float64
                )
                serving_probs = serving_logprobs.exp()
                trainer_probs = trainer_row_logprobs.exp()
                normalization_error = abs(float(serving_probs.sum()) - 1.0)
                probability_errors = (trainer_probs - serving_probs).abs()
                total_variation = 0.5 * float(probability_errors.sum())
                serving_to_trainer_kl = float((serving_probs * (serving_logprobs - trainer_row_logprobs)).sum())
                trainer_to_serving_kl = float((trainer_probs * (trainer_row_logprobs - serving_logprobs)).sum())
                distribution_summaries.append(
                    {
                        "decision_id": sample.get("decision_id"),
                        "agent_id": sample["agent_id"],
                        "policy_slot": policy_slot,
                        "token_offset": token_offset,
                        "allowed_token_count": len(allowed),
                        "serving_normalization_error": normalization_error,
                        "max_probability_error": float(probability_errors.max()),
                        "total_variation": total_variation,
                        "serving_to_trainer_kl": serving_to_trainer_kl,
                        "trainer_to_serving_kl": trainer_to_serving_kl,
                    }
                )
        all_inference.append(inference_logprobs)
        all_trainer.append(trainer_logprobs)
        all_branching.append(torch.tensor([len(row) > 1 for row in allowed_rows]))
        sample_absolute_error = (trainer_logprobs - inference_logprobs).abs()
        sample_probability_error = (trainer_logprobs.exp() - inference_logprobs.exp()).abs()
        sample_log_ratio = trainer_logprobs - inference_logprobs
        sample_mismatch_kl = sample_log_ratio.exp() - sample_log_ratio - 1.0
        for token_offset, (
            token_id,
            allowed,
            rollout_value,
            trainer_value,
            absolute_value,
            probability_value,
            mismatch_value,
        ) in enumerate(
            zip(
                sample["completion_ids"],
                allowed_rows,
                inference_logprobs.tolist(),
                trainer_logprobs.tolist(),
                sample_absolute_error.tolist(),
                sample_probability_error.tolist(),
                sample_mismatch_kl.tolist(),
                strict=True,
            )
        ):
            token_summaries.append(
                {
                    "decision_id": sample.get("decision_id"),
                    "agent_id": sample["agent_id"],
                    "policy_slot": policy_slot,
                    "token_offset": token_offset,
                    "token_id": token_id,
                    "allowed_token_count": len(allowed),
                    "rollout_logprob": rollout_value,
                    "trainer_logprob": trainer_value,
                    "absolute_logprob_error": absolute_value,
                    "probability_error": probability_value,
                    "mismatch_kl": mismatch_value,
                }
            )
        sample_summaries.append(
            {
                "agent_id": sample["agent_id"],
                "max_absolute_logprob_error": float((trainer_logprobs - inference_logprobs).abs().max()),
                "phase": sample["phase"],
                "decision_id": sample.get("decision_id"),
                "game_id": sample.get("game_id"),
                "policy_slot": policy_slot,
                "tokens": len(inference_logprobs),
            }
        )

    inference_logprobs = torch.cat(all_inference)
    trainer_logprobs = torch.cat(all_trainer)
    branching = torch.cat(all_branching)
    absolute_error = (trainer_logprobs - inference_logprobs).abs()
    log_importance_ratio = trainer_logprobs - inference_logprobs
    probability_error = (trainer_logprobs.exp() - inference_logprobs.exp()).abs()
    importance_ratio_error = (log_importance_ratio.exp() - 1.0).abs()
    mismatch_kl = log_importance_ratio.exp() - log_importance_ratio - 1.0
    max_absolute_error = float(absolute_error.max())
    mean_absolute_error = float(absolute_error.mean())
    max_probability_error = float(probability_error.max())
    max_importance_ratio_error = float(importance_ratio_error.max())
    max_mismatch_kl = float(mismatch_kl.max())
    mean_mismatch_kl = float(mismatch_kl.mean())
    p99_absolute_error = float(torch.quantile(absolute_error, 0.99))
    p99_probability_error = float(torch.quantile(probability_error, 0.99))
    probability_tail_fraction = float((probability_error > args.probability_tail_threshold).float().mean())

    def within(value: float, threshold: float | None) -> bool:
        return threshold is None or value <= threshold

    parity_passed = all(
        (
            within(mean_absolute_error, args.max_mean_logprob_error),
            within(p99_absolute_error, args.max_p99_logprob_error),
            within(max_probability_error, args.max_probability_error),
            within(p99_probability_error, args.max_p99_probability_error),
            within(probability_tail_fraction, args.max_probability_tail_fraction),
            within(mean_mismatch_kl, args.max_mean_mismatch_kl),
            within(max_mismatch_kl, args.max_mismatch_kl),
        )
    )

    optimizer_param_ids = []
    for index in sorted(manager.used_idxs):
        current = optimizer.optimizers[index]
        if current is None:
            raise RuntimeError(f"optimizer missing for policy slot {index}")
        optimizer_param_ids.append({id(parameter) for group in current.param_groups for parameter in group["params"]})
    optimizer_sets_disjoint = all(
        not optimizer_param_ids[left] & optimizer_param_ids[right] for left in range(4) for right in range(left + 1, 4)
    )
    if not optimizer_sets_disjoint:
        raise RuntimeError("policy optimizers share parameter objects")

    run_id = "run_blue_0"
    slot = manager.id_2_idx[run_id]
    token_ids, position_ids, _, allowed_rows, prompt_length = prepare_sample(samples[0], device=device)
    labels = shift_tensor_left(token_ids)
    lora_num_tokens = torch.zeros(4, dtype=torch.int32, device=device)
    lora_num_tokens[slot] = token_ids.shape[1]
    set_lora_num_tokens(lora_num_tokens)
    for index in range(4):
        manager.ready_to_update[index] = index == slot
    optimizer.zero_grad()
    model.train()
    output = forward(
        model,
        token_ids,
        position_ids,
        labels=labels,
        temperature=torch.ones_like(token_ids, dtype=torch.float32),
    )
    logits = output.get("logits")
    if logits is None:
        raise RuntimeError("isolation step requires unfused trainer LM head logits")
    loss = -constrained_logprobs(logits, token_ids, prompt_length, allowed_rows).mean()
    loss.backward()
    optimizer.step()
    slot_digests_after = {
        current_run: adapter_digest(manager.get_state_dict_for_run(index))
        for current_run, index in sorted(manager.id_2_idx.items())
    }
    changed_runs = sorted(
        current_run
        for current_run in slot_digests_before
        if slot_digests_before[current_run] != slot_digests_after[current_run]
    )
    isolation_passed = changed_runs == [run_id]

    report = {
        "adapter_sha256": args.adapter_sha256,
        "probe_sha256": sha256_file(args.probe),
        "trainer_config_sha256": trainer_config_sha256,
        "trainer_parity_gate_sha256": trainer_parity_gate_sha256,
        "trainer_model_impl": config.model.impl,
        "trainer_attention": config.model.attn,
        "changed_runs_after_single_policy_step": changed_runs,
        "branching_tokens": int(branching.sum()),
        "completion_tokens": len(inference_logprobs),
        "isolation_passed": isolation_passed,
        "max_absolute_logprob_error": max_absolute_error,
        "max_importance_ratio_error": max_importance_ratio_error,
        "max_mismatch_kl": max_mismatch_kl,
        "max_probability_error": max_probability_error,
        "mean_absolute_logprob_error": mean_absolute_error,
        "mean_branching_absolute_logprob_error": float(absolute_error[branching].mean()),
        "mean_mismatch_kl": mean_mismatch_kl,
        "p95_absolute_logprob_error": float(torch.quantile(absolute_error, 0.95)),
        "p99_absolute_logprob_error": p99_absolute_error,
        "p99_branching_absolute_logprob_error": float(torch.quantile(absolute_error[branching], 0.99)),
        "p99_probability_error": p99_probability_error,
        "probability_error_over_0_05_fraction": probability_tail_fraction,
        "optimizer_parameter_sets_disjoint": optimizer_sets_disjoint,
        "parity_passed": parity_passed,
        "parity_components": {
            "mean_absolute_logprob_error": within(mean_absolute_error, args.max_mean_logprob_error),
            "p99_absolute_logprob_error": within(p99_absolute_error, args.max_p99_logprob_error),
            "max_probability_error": within(max_probability_error, args.max_probability_error),
            "p99_probability_error": within(p99_probability_error, args.max_p99_probability_error),
            "probability_tail_fraction": within(probability_tail_fraction, args.max_probability_tail_fraction),
            "mean_mismatch_kl": within(mean_mismatch_kl, args.max_mean_mismatch_kl),
            "max_mismatch_kl": within(max_mismatch_kl, args.max_mismatch_kl),
        },
        "parity_thresholds": {
            "max_mean_logprob_error": args.max_mean_logprob_error,
            "max_p99_logprob_error": args.max_p99_logprob_error,
            "max_mismatch_kl": args.max_mismatch_kl,
            "max_mean_mismatch_kl": args.max_mean_mismatch_kl,
            "max_probability_error": args.max_probability_error,
            "max_p99_probability_error": args.max_p99_probability_error,
            "probability_tail_threshold": args.probability_tail_threshold,
            "max_probability_tail_fraction": args.max_probability_tail_fraction,
        },
        "policy_slot_digests_before": slot_digests_before,
        "sample_summaries": sample_summaries,
        "top_probability_error_tokens": sorted(
            token_summaries,
            key=lambda row: float(row["probability_error"]),
            reverse=True,
        )[:20],
        "distribution_diagnostics": {
            "complete_rows": len(distribution_summaries),
            "max_serving_normalization_error": (
                max(row["serving_normalization_error"] for row in distribution_summaries)
                if distribution_summaries
                else None
            ),
            "mean_total_variation": (
                sum(row["total_variation"] for row in distribution_summaries) / len(distribution_summaries)
                if distribution_summaries
                else None
            ),
            "max_total_variation": (
                max(row["total_variation"] for row in distribution_summaries) if distribution_summaries else None
            ),
            "mean_serving_to_trainer_kl": (
                sum(row["serving_to_trainer_kl"] for row in distribution_summaries) / len(distribution_summaries)
                if distribution_summaries
                else None
            ),
            "max_serving_to_trainer_kl": (
                max(row["serving_to_trainer_kl"] for row in distribution_summaries) if distribution_summaries else None
            ),
            "mean_trainer_to_serving_kl": (
                sum(row["trainer_to_serving_kl"] for row in distribution_summaries) / len(distribution_summaries)
                if distribution_summaries
                else None
            ),
            "max_trainer_to_serving_kl": (
                max(row["trainer_to_serving_kl"] for row in distribution_summaries) if distribution_summaries else None
            ),
            "top_total_variation_rows": sorted(
                distribution_summaries,
                key=lambda row: row["total_variation"],
                reverse=True,
            )[:20],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    dist.destroy_process_group()
    if not parity_passed or not isolation_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
