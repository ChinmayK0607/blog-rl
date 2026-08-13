from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
        selected_rows.append(
            prediction[input_ids[0, token_position]] - torch.logsumexp(prediction[legal_ids], dim=0)
        )
    return torch.stack(selected_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify vLLM-to-Prime constrained logprob parity.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.01)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse parity output directory: {args.output_dir}")
    for index in range(4):
        write_run_config(args.output_dir, f"run_blue_{index}", args.model)

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    prompt_ids = list(probe["prompt_ids"])
    completion_ids = list(probe["completion_ids"])
    allowed_rows = [list(row) for row in probe["allowed_token_ids"]]
    inference_logprobs = torch.tensor(probe["completion_logprobs"], dtype=torch.float32)
    if not (len(completion_ids) == len(allowed_rows) == len(inference_logprobs)):
        raise ValueError("probe completion fields have inconsistent lengths")

    setup_torch_distributed()
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
        }
    )
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

    run_id = "run_blue_0"
    slot = manager.id_2_idx[run_id]
    token_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(token_ids.shape[1], device=device).unsqueeze(0)
    labels = shift_tensor_left(token_ids)
    lora_num_tokens = torch.zeros(4, dtype=torch.int32, device=device)
    lora_num_tokens[slot] = token_ids.shape[1]
    set_lora_num_tokens(lora_num_tokens)
    model.eval()
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
        trainer_logprobs = constrained_logprobs(logits, token_ids, len(prompt_ids), allowed_rows).cpu()

    absolute_error = (trainer_logprobs - inference_logprobs).abs()
    max_absolute_error = float(absolute_error.max())
    mean_absolute_error = float(absolute_error.mean())
    parity_passed = max_absolute_error <= args.atol

    optimizer_param_ids = []
    for index in sorted(manager.used_idxs):
        current = optimizer.optimizers[index]
        if current is None:
            raise RuntimeError(f"optimizer missing for policy slot {index}")
        optimizer_param_ids.append(
            {id(parameter) for group in current.param_groups for parameter in group["params"]}
        )
    optimizer_sets_disjoint = all(
        not optimizer_param_ids[left] & optimizer_param_ids[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    if not optimizer_sets_disjoint:
        raise RuntimeError("policy optimizers share parameter objects")

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
    loss = -constrained_logprobs(logits, token_ids, len(prompt_ids), allowed_rows).mean()
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
        "changed_runs_after_single_policy_step": changed_runs,
        "completion_tokens": len(completion_ids),
        "inference_logprobs": inference_logprobs.tolist(),
        "isolation_passed": isolation_passed,
        "max_absolute_logprob_error": max_absolute_error,
        "mean_absolute_logprob_error": mean_absolute_error,
        "optimizer_parameter_sets_disjoint": optimizer_sets_disjoint,
        "parity_atol": args.atol,
        "parity_passed": parity_passed,
        "policy_slot_digests_before": slot_digests_before,
        "trainer_logprobs": trainer_logprobs.tolist(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    dist.destroy_process_group()
    if not parity_passed or not isolation_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
