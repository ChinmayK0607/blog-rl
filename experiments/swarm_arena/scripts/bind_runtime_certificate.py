from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _gpu_inventory() -> list[dict[str, str | int]]:
    rows = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        text=True,
    ).splitlines()
    inventory = []
    for row in rows:
        index, name, total, driver = (value.strip() for value in row.split(",", 3))
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "driver_version": driver,
            }
        )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind the exact serving and trainer calibration to one runtime certificate."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--initial-policy-adapter-manifest", type=Path)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--serving-probe", type=Path, required=True)
    parser.add_argument("--parity-probe", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        raise ValueError("repository must be clean before binding a runtime certificate")

    adapter_file = args.adapter / "adapter_model.safetensors"
    adapter_sha256 = _sha256_file(adapter_file)
    trainer_sha256 = _sha256_file(args.trainer_config)
    inference_sha256 = _sha256_file(args.inference_config)
    serving = json.loads(args.serving_probe.read_text(encoding="utf-8"))
    parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
    parity_probe = json.loads(args.parity_probe.read_text(encoding="utf-8"))
    initial_policy_adapter_manifest_sha256 = (
        _sha256_file(args.initial_policy_adapter_manifest)
        if args.initial_policy_adapter_manifest is not None
        else None
    )
    if serving.get("status") != "passed" or serving.get("servers") != 3:
        raise ValueError("serving probe must pass against exactly three rollout servers")
    if serving.get("adapter_sha256") != adapter_sha256:
        raise ValueError("serving probe adapter does not match the initial adapter")
    if not isinstance(serving.get("base_urls"), list) or len(serving["base_urls"]) != 3:
        raise ValueError("serving probe must identify all three rollout server URLs")
    if parity.get("parity_passed") is not True:
        raise ValueError("numerical parity report did not pass")
    if parity.get("isolation_passed") is not True:
        raise ValueError("four-policy optimizer isolation report did not pass")
    if parity.get("adapter_sha256") != adapter_sha256:
        raise ValueError("parity report adapter does not match the initial adapter")
    if parity_probe.get("adapter_sha256") != adapter_sha256:
        raise ValueError("runtime parity probe adapter does not match the initial adapter")
    if parity_probe.get("servers") != 3 or len(parity_probe.get("samples", [])) != 32:
        raise ValueError("runtime parity probe must contain 32 samples from three servers")
    if {row.get("server_url") for row in parity_probe["samples"]} != set(
        parity_probe.get("base_urls", [])
    ):
        raise ValueError("runtime parity probe did not exercise every declared server")
    if parity.get("probe_sha256") != _sha256_file(args.parity_probe):
        raise ValueError("parity report does not bind the supplied runtime probe")
    if parity.get("trainer_config_sha256") != trainer_sha256:
        raise ValueError("parity report was not produced from the resolved trainer config")
    if parity.get("initial_policy_adapter_manifest_sha256") != (
        initial_policy_adapter_manifest_sha256
    ):
        raise ValueError("parity report did not bind the policy adapter manifest")
    if parity_probe.get("initial_policy_adapter_manifest_sha256") != (
        initial_policy_adapter_manifest_sha256
    ):
        raise ValueError("parity probe did not bind the policy adapter manifest")
    if parity.get("policy_adapter_sha256") != parity_probe.get(
        "policy_adapter_sha256"
    ):
        raise ValueError("parity report and runtime probe bind different policy adapters")

    body = {
        "version": "swarm-runtime-certificate-v1",
        "status": "passed",
        "source_commit": head,
        "base_revision": args.base_revision,
        "adapter_sha256": adapter_sha256,
        "trainer_config_sha256": trainer_sha256,
        "inference_config_sha256": inference_sha256,
        "initial_policy_adapter_manifest_sha256": (
            initial_policy_adapter_manifest_sha256
        ),
        "policy_adapter_sha256": parity.get("policy_adapter_sha256", {}),
        "backend": {
            "name": "vllm",
            "version": importlib.metadata.version("vllm"),
        },
        "gpu_inventory": _gpu_inventory(),
        "serving_probe": {
            "sha256": _sha256_file(args.serving_probe),
            "base_urls": serving["base_urls"],
            "broadcast_completion_tokens": serving["broadcast"]["completion_tokens"],
            "action_completion_tokens": serving["action"]["completion_tokens"],
        },
        "parity_report": {
            "sha256": _sha256_file(args.parity_report),
            "probe_sha256": _sha256_file(args.parity_probe),
            "trainer_parity_gate_sha256": parity["trainer_parity_gate_sha256"],
            "mean_absolute_logprob_error": parity["mean_absolute_logprob_error"],
            "p99_absolute_logprob_error": parity["p99_absolute_logprob_error"],
            "mean_mismatch_kl": parity["mean_mismatch_kl"],
            "max_mismatch_kl": parity["max_mismatch_kl"],
        },
    }
    certificate = {**body, "sha256": _digest(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
