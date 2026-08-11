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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def build_model_card(
    repo_id: str,
    selection: dict[str, Any],
    audit: dict[str, Any],
    *,
    source_commit: str,
    training_run_url: str,
) -> str:
    selected_step = selection["selected_step"]
    candidate = next(item for item in selection["candidates"] if item["step"] == selected_step)
    validation = candidate["validation"]
    broadcast = validation["groups"]["BROADCAST"]
    action = validation["groups"]["ACT"]
    v1 = candidate["regression_v1"]
    v2 = candidate["regression_v2"]
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- multi-agent
- reinforcement-learning
- research
---

# {repo_id.split("/", 1)[-1]}

Regression-safe protocol warm start for Swarm Arena, a deterministic 4v4
partially observed graph-control environment. The adapter teaches a single
agent to emit grounded broadcasts and legal action IDs before multi-agent RL.

This model has **not** learned swarm cooperation. SFT targets use only local,
prompt-visible information and intentionally exclude joint oracle behavior.
Cooperation, message utility, and causal intervention effects remain RL and
evaluation questions.

## Frozen promotion result

- protocol: `{selection["selection_protocol"]}`;
- selected step: {selected_step};
- schema-valid responses: {percent(validation["schema_valid"])};
- grounded broadcasts: {percent(broadcast["grounded"])};
- legal actions: {percent(action["legal"])};
- paired regression v1 delta: {100 * v1["overall"]["adapter_minus_base"]:+.2f} points;
- paired regression v2 delta: {100 * v2["overall"]["adapter_minus_base"]:+.2f} points;
- regression v1/v2 gates: {str(v1["gates"]["passed"]).lower()} / {str(v2["gates"]["passed"]).lower()}.

Promotion requires perfect schema validity, at least 99% grounded broadcasts,
at least 99% legal actions, no more than a 2-point overall regression on either
frozen suite, no category regression above 5 points, and no arena-key leakage
increase. Gates were fixed before this checkpoint was evaluated.

## Training

- 2,560 training rows: 640 balanced arena-protocol examples, 640 exact
  instruction-preservation examples, and 1,280 deterministic base-behavior
  replay examples;
- rank-8 LoRA on `q_proj` and `v_proj` only (2,949,120 trainable parameters);
- Qwen3 thinking disabled; strict JSON output protocol;
- audited row-ID hash: `{audit["ids_sha256"]}`;
- training run: {training_run_url};
- source commit: `{source_commit}`.

The repository contains the simulator, data builder, audit, frozen evaluators,
raw aggregate evidence, and exact Prime-RL configuration:
<https://github.com/ChinmayK0607/blog-rl/tree/exp/swarm-arena-4b/experiments/swarm_arena>.

## Intended use

Use as the starting policy for controlled MARL research inside the discrete
Swarm Arena simulator. Do not interpret abstract node-control actions as real
network-security capabilities or the SFT checkpoint as evidence of emergent
collective intelligence.
"""


def publish(
    repo_id: str,
    adapter: Path,
    selection_path: Path,
    audit_path: Path,
    training_config: Path,
    *,
    source_commit: str,
    training_run_url: str,
) -> dict[str, Any]:
    expected_adapter_files = ("adapter_config.json", "adapter_model.safetensors")
    for filename in expected_adapter_files:
        if not (adapter / filename).is_file():
            raise FileNotFoundError(adapter / filename)
    selection = load_json(selection_path)
    if selection.get("decision") != "adapter" or selection.get("selected_step") is None:
        raise ValueError("refusing to publish a warm start that did not pass promotion")
    audit = load_json(audit_path)
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required; run `hf auth login`")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swarm-warmstart-model-") as temporary:
        root = Path(temporary)
        for filename in expected_adapter_files:
            shutil.copy2(adapter / filename, root / filename)
        evidence = root / "evidence"
        evidence.mkdir()
        shutil.copy2(selection_path, evidence / "selection.json")
        shutil.copy2(audit_path, evidence / "data_audit.json")
        shutil.copy2(training_config, root / "training.toml")
        (root / "README.md").write_text(
            build_model_card(
                repo_id,
                selection,
                audit,
                source_commit=source_commit,
                training_run_url=training_run_url,
            ),
            encoding="utf-8",
        )
        local_hash = sha256(root / "adapter_model.safetensors")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=root,
            commit_message=f"Publish regression-safe Swarm Arena warm start from {source_commit}",
        )

    info = api.model_info(repo_id=repo_id)
    expected_remote = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "evidence/data_audit.json",
        "evidence/selection.json",
        "training.toml",
    }
    remote_files = {sibling.rfilename for sibling in info.siblings}
    missing = sorted(expected_remote - remote_files)
    if missing:
        raise RuntimeError(f"published repository is missing files: {missing}")
    with tempfile.TemporaryDirectory(prefix="swarm-warmstart-verify-") as cache:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="adapter_model.safetensors",
                token=token,
                cache_dir=cache,
            )
        )
        remote_hash = sha256(downloaded)
    if remote_hash != local_hash:
        raise RuntimeError("downloaded adapter hash does not match the selected local adapter")
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": info.sha,
        "selected_step": selection["selected_step"],
        "adapter_sha256": remote_hash,
        "verified_files": sorted(expected_remote),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish and verify a promoted Swarm Arena warm-start adapter.")
    parser.add_argument("--repo-id", default="CK0607/Qwen3-4B-Swarm-Arena-Warmstart-v4")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--training-run-url", required=True)
    args = parser.parse_args()
    result = publish(
        args.repo_id,
        args.adapter,
        args.selection,
        args.data_audit,
        args.training_config,
        source_commit=args.source_commit,
        training_run_url=args.training_run_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
