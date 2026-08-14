from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

VERSION = "arena-constrained-policy-kl-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy KL has no token rows")
    result = {}
    for key in ("candidate_to_baseline_kl", "baseline_to_candidate_kl", "total_variation"):
        values = [float(row[key]) for row in rows]
        result[key] = {
            "mean": sum(values) / len(values),
            "p99": _percentile(values, 0.99),
            "max": max(values),
        }
    result["tokens"] = len(rows)
    result["branching_tokens"] = sum(int(row["allowed_token_count"]) > 1 for row in rows)
    return result


def _adapter_argument(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate adapters must use NAME=PATH")
    return name, Path(path)


@torch.inference_mode()
def _legal_distributions(
    model: Any,
    adapter_name: str,
    token_ids: torch.Tensor,
    *,
    prompt_length: int,
    allowed_rows: list[list[int]],
) -> list[torch.Tensor]:
    model.set_adapter(adapter_name)
    backbone = model.get_base_model()
    output = backbone.model(input_ids=token_ids, use_cache=False, return_dict=True)
    positions = torch.tensor(
        [prompt_length + offset - 1 for offset in range(len(allowed_rows))],
        dtype=torch.long,
        device=token_ids.device,
    )
    hidden = output.last_hidden_state[0, positions].to(torch.bfloat16)
    unique_ids = sorted({token_id for allowed in allowed_rows for token_id in allowed})
    unique = torch.tensor(unique_ids, dtype=torch.long, device=token_ids.device)
    weight = backbone.lm_head.weight[unique].to(torch.bfloat16)
    with torch.autocast("cuda", enabled=False):
        logits = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    column = {token_id: index for index, token_id in enumerate(unique_ids)}
    return [
        torch.softmax(
            logits[offset, [column[token_id] for token_id in allowed]], dim=0
        ).cpu()
        for offset, allowed in enumerate(allowed_rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure candidate-vs-warm-start KL on a frozen constrained probe."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-adapter", type=Path, required=True)
    parser.add_argument("--candidate-adapter", action="append", type=_adapter_argument, default=[])
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = dict(args.candidate_adapter)
    expected = {f"blue-{index}" for index in range(4)}
    if set(candidates) != expected:
        parser.error(f"candidate adapters must be exactly {sorted(expected)}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite policy KL report: {args.output}")

    from transformers.utils import import_utils

    import_utils._torchvision_available = False
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model = PeftModel.from_pretrained(
        base,
        args.baseline_adapter,
        adapter_name="baseline",
        is_trainable=False,
    )
    for name, path in sorted(candidates.items()):
        model.load_adapter(path, adapter_name=name, is_trainable=False)
    model.eval()

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    rows = []
    per_policy: dict[str, list[dict[str, Any]]] = {name: [] for name in sorted(candidates)}
    for sample in probe["samples"]:
        policy = f"blue-{int(sample['policy_slot'])}"
        prompt_ids = list(sample["prompt_ids"])
        completion_ids = list(sample["completion_ids"])
        allowed_rows = [list(row) for row in sample["allowed_token_ids"]]
        if len(completion_ids) != len(allowed_rows):
            raise ValueError("probe completion and allowed-token rows differ")
        token_ids = torch.tensor(
            [prompt_ids + completion_ids], dtype=torch.long, device="cuda"
        )
        baseline = _legal_distributions(
            model,
            "baseline",
            token_ids,
            prompt_length=len(prompt_ids),
            allowed_rows=allowed_rows,
        )
        candidate = _legal_distributions(
            model,
            policy,
            token_ids,
            prompt_length=len(prompt_ids),
            allowed_rows=allowed_rows,
        )
        for offset, (base_probs, candidate_probs, allowed) in enumerate(
            zip(baseline, candidate, allowed_rows, strict=True)
        ):
            base_log = base_probs.clamp_min(1e-30).log()
            candidate_log = candidate_probs.clamp_min(1e-30).log()
            row = {
                "policy_id": policy,
                "decision_id": sample["decision_id"],
                "token_offset": offset,
                "allowed_token_count": len(allowed),
                "candidate_to_baseline_kl": float(
                    (candidate_probs * (candidate_log - base_log)).sum()
                ),
                "baseline_to_candidate_kl": float(
                    (base_probs * (base_log - candidate_log)).sum()
                ),
                "total_variation": float((candidate_probs - base_probs).abs().sum() / 2),
            }
            rows.append(row)
            per_policy[policy].append(row)

    report = {
        "version": VERSION,
        "probe_sha256": _sha256_file(args.probe),
        "baseline_adapter_sha256": _sha256_file(
            args.baseline_adapter / "adapter_model.safetensors"
        ),
        "candidate_adapter_sha256": {
            name: _sha256_file(path / "adapter_model.safetensors")
            for name, path in sorted(candidates.items())
        },
        "overall": _summary(rows),
        "per_policy": {
            name: _summary(policy_rows)
            for name, policy_rows in sorted(per_policy.items())
        },
        "scope": "reference-state constrained distributions; diagnostic collapse gate only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
