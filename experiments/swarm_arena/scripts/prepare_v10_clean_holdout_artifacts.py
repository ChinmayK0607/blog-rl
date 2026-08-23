from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in lock.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != lock.get("sha256"):
        raise ValueError("clean holdout lock body hash mismatch")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and hash-check public v10 held-out artifacts on a GPU host.")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()

    lock = _load_lock(args.lock)
    bindings = lock["bindings"]
    base_dir = args.workspace / "models" / "qwen3-4b-cdbee75f"
    sft_dir = args.workspace / "artifacts" / "qwen3-4b-sft-v2-d1a55d55"
    live_root = args.workspace / "artifacts" / "swarm-arena-live-runs"
    base_dir.parent.mkdir(parents=True, exist_ok=True)
    sft_dir.parent.mkdir(parents=True, exist_ok=True)
    live_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=bindings["base"]["repo_id"],
        revision=bindings["base"]["revision"],
        local_dir=base_dir,
        token=False,
    )
    snapshot_download(
        repo_id=bindings["baseline"]["repo_id"],
        revision=bindings["baseline"]["revision"],
        local_dir=sft_dir,
        token=False,
    )
    artifact = bindings["public_artifact"]
    snapshot_download(
        repo_id=artifact["repo_id"],
        revision=artifact["repo_revision"],
        local_dir=live_root,
        allow_patterns=[f"{artifact['step40_path']}/*"],
        token=False,
    )

    sft_adapter = sft_dir / "adapter_model.safetensors"
    if _sha256_file(sft_adapter) != bindings["baseline"]["adapter_sha256"]:
        raise ValueError("downloaded SFT adapter hash mismatch")
    candidate_root = live_root / artifact["step40_path"]
    observed = {}
    for policy_id, expected in bindings["candidate"]["adapter_sha256"].items():
        adapter = candidate_root / f"policy-{policy_id}" / "adapter_model.safetensors"
        observed[policy_id] = _sha256_file(adapter)
        if observed[policy_id] != expected:
            raise ValueError(f"downloaded candidate adapter hash mismatch: {policy_id}")

    status = {
        "version": "swarm-v10-clean-holdout-artifacts-v1",
        "lock_sha256": lock["sha256"],
        "base_revision": bindings["base"]["revision"],
        "baseline_revision": bindings["baseline"]["revision"],
        "base_dir": str(base_dir),
        "sft_dir": str(sft_dir),
        "candidate_root": str(candidate_root),
        "candidate_adapter_sha256": observed,
        "status": "ready",
    }
    status_path = args.workspace / "run-inputs" / "v10-clean-holdout-artifacts.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
