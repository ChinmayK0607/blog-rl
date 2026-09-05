from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, hf_hub_download

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DATASET = "CK0607/swarm-arena-sft-v2"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(value: Any) -> str:
    return f"{float(value):.4f}"


def build_model_card(
    repo_id: str,
    selection: dict[str, Any],
    test: dict[str, Any],
    arena: dict[str, Any],
    comparison: dict[str, Any],
    *,
    source_commit: str,
    training_run_url: str,
) -> str:
    generated = arena["conditions"]["generated"]
    gates = comparison["claim_gates"]
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
datasets:
- {DATASET}
tags:
- lora
- multi-agent
- coordination
- research
---

# {repo_id.split('/', 1)[-1]}

LoRA warm start for the Swarm Arena 4v4 partially observed graph-control
environment. The model emits strict JSON broadcasts and actions for individual
agents. This checkpoint was selected using validation behavior only, then
evaluated once on the untouched SFT test split and frozen 60-case arena.

This is **not** a multi-agent-RL-trained model. It is the supervised protocol and
mechanics warm start intended to make subsequent MARL experiments interpretable.

## Selection and test results

- selected training step: {selection['selected_step']};
- validation candidates: {selection['num_candidates']} ({selection['num_eligible']} passed fixed gates);
- held-out test phase-balanced exactness: {rate(test['selection_score'])};
- held-out action exactness: {rate(test['act']['exact'])};
- held-out broadcast exactness: {rate(test['broadcast']['exact'])};
- held-out unsupported-broadcast rate: {rate(1.0 - float(test['broadcast']['supported']))}.

## Frozen arena results

- strict broadcast rate: {rate(arena['message_strict_rate'])};
- strict generated-condition action rate: {rate(generated['strict_action_rate'])};
- generated-condition mean oracle regret: {float(generated['mean_oracle_regret']):.4f};
- generated-condition mean environment reward: {float(generated['mean_environment_reward']):.4f};
- generated-minus-dropped reward: {float(arena['generated_minus_dropped_reward']):.4f};
- paired coordination-improvement claim gate: {str(gates['coordination_improvement_supported']).lower()}.

The claim gate requires 95% paired intervals to support all three: lower oracle
regret than the untouched base model, generated messages beating dropped
messages, and generated messages beating shuffled messages. A `false` result is
reported as a negative or inconclusive experiment, not relaxed post hoc.

## Reproducibility

- source commit: `{source_commit}`;
- training run: {training_run_url};
- environment: `arena-core-v1`;
- prompt: `arena-v2-structured-priority`;
- arena manifest: `{arena['manifest_sha256']}`;
- deterministic generation with thinking disabled;
- learned artifact: rank-32 PEFT LoRA adapter over {BASE_MODEL}.

The `results/` directory contains the frozen selection decision and final test,
arena, and paired-comparison summaries. Raw trajectories are logged separately
as a W&B evaluation artifact.

## Intended use

Use for research on structured communication, multi-agent credit assignment,
and controlled MARL within the discrete Swarm Arena simulator. Do not interpret
the abstract graph actions as real network-security capabilities.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(
    repo_id: str,
    adapter: Path,
    selection_path: Path,
    test_path: Path,
    arena_path: Path,
    comparison_path: Path,
    *,
    source_commit: str,
    training_run_url: str,
) -> dict[str, Any]:
    expected_adapter_files = ("adapter_config.json", "adapter_model.safetensors")
    for filename in expected_adapter_files:
        if not (adapter / filename).is_file():
            raise FileNotFoundError(adapter / filename)
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required; set HF_TOKEN or run `hf auth login`")

    selection = load_json(selection_path)
    test = load_json(test_path)
    arena = load_json(arena_path)
    comparison = load_json(comparison_path)
    if test.get("split") != "test":
        raise ValueError("refusing to publish a final model without a held-out test summary")
    if arena.get("manifest_sha256") != comparison.get("manifest_sha256"):
        raise ValueError("arena and comparison manifests differ")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swarm-arena-model-") as temporary:
        root = Path(temporary)
        for filename in expected_adapter_files:
            shutil.copy2(adapter / filename, root / filename)
        results = root / "results"
        results.mkdir()
        for name, path in {
            "selection.json": selection_path,
            "test_summary.json": test_path,
            "arena_summary.json": arena_path,
            "arena_comparison.json": comparison_path,
        }.items():
            shutil.copy2(path, results / name)
        (root / "README.md").write_text(
            build_model_card(
                repo_id,
                selection,
                test,
                arena,
                comparison,
                source_commit=source_commit,
                training_run_url=training_run_url,
            ),
            encoding="utf-8",
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=root,
            commit_message=f"Publish selected Swarm Arena adapter from {source_commit}",
        )
        local_adapter_hash = sha256(root / "adapter_model.safetensors")

    info = HfApi(token=False).model_info(repo_id=repo_id)
    if info.private:
        raise RuntimeError("published model repository is not anonymously public")
    remote_files = {sibling.rfilename for sibling in info.siblings}
    expected_remote = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "results/selection.json",
        "results/test_summary.json",
        "results/arena_summary.json",
        "results/arena_comparison.json",
    }
    missing = sorted(expected_remote - remote_files)
    if missing:
        raise RuntimeError(f"published repository is missing files: {missing}")
    with tempfile.TemporaryDirectory(prefix="swarm-arena-verify-") as verification_cache:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="adapter_model.safetensors",
                repo_type="model",
                token=False,
                cache_dir=verification_cache,
            )
        )
        remote_adapter_hash = sha256(downloaded)
    if remote_adapter_hash != local_adapter_hash:
        raise RuntimeError("downloaded adapter hash does not match the selected local adapter")
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": info.sha,
        "selected_step": selection["selected_step"],
        "adapter_sha256": remote_adapter_hash,
        "verified_files": sorted(expected_remote),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish and verify the selected Swarm Arena LoRA adapter.")
    parser.add_argument("--repo-id", default="CK0607/Qwen3-4B-Swarm-Arena-SFT-v2")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--arena-summary", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--training-run-url", required=True)
    args = parser.parse_args()
    result = publish(
        args.repo_id,
        args.adapter,
        args.selection,
        args.test_summary,
        args.arena_summary,
        args.comparison,
        source_commit=args.source_commit,
        training_run_url=args.training_run_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
