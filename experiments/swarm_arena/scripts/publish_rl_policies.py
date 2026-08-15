from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, snapshot_download
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_argument(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path or Path(name).name != name:
        raise argparse.ArgumentTypeError("reports must use a flat NAME=PATH value")
    return name, Path(path)


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _write_checksums(root: Path) -> dict[str, str]:
    checksums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _files(root)
        if path.name != "SHA256SUMS"
    }
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def _model_card(repo_id: str, step: int, provenance: dict[str, Any]) -> str:
    return f"""---
library_name: peft
base_model: {provenance['base_model']}
tags:
- lora
- multi-agent-rl
- swarm-arena
---

# {repo_id.split('/', 1)[-1]}

Four distinct LoRA policies trained over one frozen Qwen3 1.7B backbone in the
Swarm Arena 4v4 partially observed graph-control simulator. Policy directories
`policy_blue_0` through `policy_blue_3` retain separate optimizer identities and
must be assigned to their corresponding BLUE roles.

Selected trainer step: `{step}`. Exact provenance, policy hashes, public input
revisions, and compact evaluation reports are in `PROVENANCE.json`,
`SHA256SUMS`, and `results/`.

The reward is the zero-sum terminal control-margin delta. There is no speaking,
silence, capture, or learned-judge bonus. Higher return is evidence of task
learning; a communication claim additionally requires normal messages to beat
dropped, shuffled, and delayed-message interventions on held-out cases.

This is research software for a discrete simulator. It is not evidence of broad
swarm intelligence or real-world cybersecurity capability.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish and anonymously verify four selected Swarm Arena RL policies."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--initial-adapter-repo", required=True)
    parser.add_argument("--initial-adapter-revision", required=True)
    parser.add_argument("--report", action="append", type=_report_argument, default=[])
    args = parser.parse_args()
    if args.step < 1:
        parser.error("step must be positive")
    reports = dict(args.report)
    if len(reports) != len(args.report):
        parser.error("report names must be unique")

    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    api = HfApi(token=token)
    expected_owner = args.repo_id.split("/", 1)[0]
    owner = str(api.whoami().get("name", ""))
    if owner.casefold() != expected_owner.casefold():
        raise RuntimeError(
            f"authenticated Hugging Face owner is {owner!r}, expected {expected_owner!r}"
        )

    policy_sources = {
        f"blue-{index}": args.run_dir
        / "exports"
        / f"step_{args.step}"
        / f"blue-{index}"
        for index in range(4)
    }
    policy_hashes = {}
    for name, source in policy_sources.items():
        for filename in ("STABLE", "adapter_config.json", "adapter_model.safetensors"):
            if filename == "STABLE":
                continue
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
        with safe_open(
            source / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as handle:
            if not handle.keys() or not all(
                key.startswith("base_model.model.") for key in handle.keys()
            ):
                raise RuntimeError(f"policy adapter is not PEFT-compatible: {source}")
        policy_hashes[name] = sha256_file(source / "adapter_model.safetensors")
    if len(set(policy_hashes.values())) != 4:
        raise RuntimeError("selected policy adapters are not four distinct checkpoints")
    for path in reports.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    provenance = {
        "version": "swarm-arena-rl-policy-bundle-v1",
        "source_commit": args.source_commit,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "initial_adapter_repo": args.initial_adapter_repo,
        "initial_adapter_revision": args.initial_adapter_revision,
        "run_dir": str(args.run_dir),
        "selected_step": args.step,
        "policy_adapter_sha256": policy_hashes,
        "reports": sorted(reports),
        "claim_boundary": (
            "Return gains are task capability evidence. Communication requires paired "
            "message-intervention gains; neither establishes broad swarm intelligence."
        ),
    }

    with tempfile.TemporaryDirectory(prefix="swarm-rl-publish-") as staging_name:
        staging = Path(staging_name)
        for name, source in policy_sources.items():
            destination = staging / f"policy_{name.replace('-', '_')}"
            destination.mkdir()
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copy2(source / filename, destination / filename)
        results = staging / "results"
        results.mkdir()
        for name, source in sorted(reports.items()):
            shutil.copy2(source, results / name)
        (staging / "PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            _model_card(args.repo_id, args.step, provenance), encoding="utf-8"
        )
        checksums = _write_checksums(staging)
        api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=staging,
            commit_message=f"Publish verified four-policy RL step {args.step}",
        )
        revision = str(commit.oid or api.model_info(args.repo_id).sha)

    public = HfApi(token=False).model_info(args.repo_id, revision=revision)
    if public.private:
        raise RuntimeError("published model repository is not anonymously public")
    with tempfile.TemporaryDirectory(prefix="swarm-rl-verify-") as verification_name:
        downloaded = Path(
            snapshot_download(
                repo_id=args.repo_id,
                revision=revision,
                token=False,
                local_dir=verification_name,
            )
        )
        expected_manifest = "".join(
            f"{digest}  {name}\n" for name, digest in checksums.items()
        )
        if (downloaded / "SHA256SUMS").read_text(encoding="utf-8") != expected_manifest:
            raise RuntimeError("public checksum manifest differs from the staged manifest")
        for name, digest in checksums.items():
            if sha256_file(downloaded / name) != digest:
                raise RuntimeError(f"public artifact checksum mismatch: {name}")

    print(
        json.dumps(
            {
                "repo": f"https://huggingface.co/{args.repo_id}",
                "revision": revision,
                "private": False,
                "selected_step": args.step,
                "policy_adapter_sha256": policy_hashes,
                "verified_files": len(checksums),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
