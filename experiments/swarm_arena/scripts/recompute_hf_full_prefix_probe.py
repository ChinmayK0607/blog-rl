from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute a parity probe with a PEFT actor's full-prefix, no-KV-cache "
            "distributions. This is a numerical diagnostic, not new rollout evidence."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"), required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe: {args.output}")

    from transformers.utils import import_utils

    import_utils._torchvision_available = False
    from peft import PeftModel

    device = torch.device("cuda")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
    ).to(device)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        adapter_name="probe",
        is_trainable=False,
    )
    model.eval()
    backbone = model.get_base_model()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    for sample in probe["samples"]:
        prompt_ids = list(sample["prompt_ids"])
        completion_ids = list(sample["completion_ids"])
        allowed_rows = [list(row) for row in sample["allowed_token_ids"]]
        if len(completion_ids) != len(allowed_rows):
            raise ValueError("completion and allowed-token row counts differ")
        token_ids = torch.tensor(
            [prompt_ids + completion_ids], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            output = backbone.model(
                input_ids=token_ids,
                use_cache=False,
                return_dict=True,
            )
            start = len(prompt_ids) - 1
            hidden = output.last_hidden_state[0, start : start + len(completion_ids)]
            hidden = hidden.to(dtype=torch.bfloat16)
            weight = backbone.lm_head.weight.to(dtype=torch.bfloat16)
            with torch.autocast("cuda", enabled=False):
                logits = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        completion_logprobs: list[float] = []
        serving_rows: list[list[list[float | int]]] = []
        for row_index, (token_id, allowed) in enumerate(
            zip(completion_ids, allowed_rows, strict=True)
        ):
            legal_ids = torch.tensor(allowed, dtype=torch.long, device=device)
            legal_logprobs = torch.log_softmax(logits[row_index, legal_ids], dim=0)
            try:
                selected_index = allowed.index(token_id)
            except ValueError as error:
                raise ValueError("sampled token is absent from its allowed row") from error
            completion_logprobs.append(float(legal_logprobs[selected_index]))
            serving_rows.append(
                [
                    [candidate, float(value)]
                    for candidate, value in zip(
                        allowed, legal_logprobs.tolist(), strict=True
                    )
                ]
            )
        sample["completion_logprobs"] = completion_logprobs
        sample["serving_allowed_logprobs"] = serving_rows

    probe["version"] = "shared-return-parity-probe-hf-full-prefix-diagnostic-v1"
    probe["diagnostic"] = {
        "attention": args.attention,
        "kv_cache": False,
        "source_probe_sha256": sha256_file(args.probe),
        "warning": "Recomputed distributions only; completions came from the source rollout.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print({"output": str(args.output), "samples": len(probe["samples"])})


if __name__ == "__main__":
    main()
