from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


EXPECTED_OWNER = "CK0607"
CANONICAL_GIT_COMMIT = "bc41a3b6"
ROOT = Path("/root/blog-rl")


@dataclass(frozen=True)
class Target:
    repo_id: str
    repo_type: str
    sources: tuple[tuple[Path, Path], ...]
    title: str
    provenance: dict[str, object]


TARGETS = (
    Target(
        repo_id="CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible",
        repo_type="model",
        sources=(
            (
                ROOT
                / "outputs/swarm_arena/qwen3_1_7b_warmstart_v2/weights/step_320/lora_adapters",
                Path("."),
            ),
        ),
        title="Qwen3 1.7B Swarm Arena warm-start v2, step 320",
        provenance={
            "base_model": "Qwen/Qwen3-1.7B",
            "base_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "selection_artifact_sha256": "2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b",
            "raw_gate_status": "NONELIGIBLE",
            "diagnostic_note": (
                "Mechanics-constrained cross-play is diagnostic only; its protocol validity is "
                "enforced structurally and is not learned protocol competence."
            ),
        },
    ),
    Target(
        repo_id="CK0607/Qwen3-4B-Swarm-Arena-SFT-v8-step256-noneligible",
        repo_type="model",
        sources=(
            (
                ROOT
                / "outputs/swarm_arena/qwen3_4b_warmstart_v8/weights/step_256/lora_adapters",
                Path("."),
            ),
            (
                ROOT / "experiments/swarm_arena/configs/sft_warmstart_4b_v8.toml",
                Path("training/sft_warmstart_4b_v8.toml"),
            ),
        ),
        title="Qwen3 4B Swarm Arena warm-start v8, step 256",
        provenance={
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "base_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "selection_artifact_sha256": "44deb6b2e18cb22ee9be49d59e36dce2beb530ebdb161560e6130108a2ce4bdc",
            "raw_gate_status": "NONELIGIBLE",
            "diagnostic_note": (
                "Mechanics-constrained cross-play is diagnostic only; its protocol validity is "
                "enforced structurally and is not learned protocol competence."
            ),
        },
    ),
    Target(
        repo_id="CK0607/swarm-arena-crossplay-results",
        repo_type="dataset",
        sources=(
            (
                ROOT / "experiments/swarm_arena/results/warmstart_1_7b_v2",
                Path("results/warmstart_1_7b_v2"),
            ),
            (
                ROOT / "experiments/swarm_arena/results/warmstart_4b_v8",
                Path("results/warmstart_4b_v8"),
            ),
            (
                ROOT / "experiments/swarm_arena/results/warmstart_4b_existing",
                Path("results/warmstart_4b_existing"),
            ),
        ),
        title="Swarm Arena cross-play diagnostics and development results",
        provenance={
            "contents": (
                "Raw rows, manifests, summaries, validation selection artifacts, and structured "
                "decoding bias diagnostics. No model weights or model caches."
            ),
            "claim_boundary": (
                "Exploratory and diagnostic evidence. The selected adapters remain NONELIGIBLE "
                "under the frozen raw-output RL-readiness gate."
            ),
        },
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def write_checksums(root: Path) -> dict[str, str]:
    checksums = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in files(root)
        if path.name != "SHA256SUMS"
    }
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def copy_source(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=False)
    else:
        shutil.copy2(source, destination)


def readme(target: Target) -> str:
    if target.repo_type == "model":
        frontmatter = "---\nlibrary_name: peft\ntags:\n- lora\n- multi-agent-rl\n- swarm-arena\n---\n\n"
    else:
        frontmatter = "---\ntags:\n- multi-agent-rl\n- swarm-arena\n- cross-play\n---\n\n"
    return (
        frontmatter
        + f"# {target.title}\n\n"
        + "Generated from the Swarm Arena experiment branch at canonical Git commit "
        + f"`{CANONICAL_GIT_COMMIT}`. See `PROVENANCE.json` and `SHA256SUMS`.\n\n"
        + "Important: mechanics-constrained evaluations guarantee valid JSON, grounded facts, "
        + "budget-feasible broadcasts, and legal actions. They do not demonstrate that those "
        + "behaviors were learned, nor do they by themselves establish cooperation.\n"
    )


def stage(target: Target, staging_root: Path) -> tuple[Path, dict[str, str]]:
    root = staging_root / target.repo_id.split("/", 1)[1]
    root.mkdir(parents=True)
    for source, relative in target.sources:
        destination = root if relative == Path(".") else root / relative
        copy_source(source, destination)
    provenance = {
        "repo_id": target.repo_id,
        "repo_type": target.repo_type,
        "canonical_git_commit": CANONICAL_GIT_COMMIT,
        **target.provenance,
    }
    (root / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(readme(target), encoding="utf-8")
    return root, write_checksums(root)


def verify_snapshot(
    target: Target,
    revision: str,
    expected: dict[str, str],
    verification_root: Path,
) -> None:
    downloaded = Path(
        snapshot_download(
            repo_id=target.repo_id,
            repo_type=target.repo_type,
            revision=revision,
            local_dir=verification_root / target.repo_id.split("/", 1)[1],
        )
    )
    manifest = (downloaded / "SHA256SUMS").read_text(encoding="utf-8")
    expected_manifest = "".join(f"{digest}  {name}\n" for name, digest in expected.items())
    if manifest != expected_manifest:
        raise RuntimeError(f"uploaded checksum manifest differs for {target.repo_id}")
    for name, digest in expected.items():
        path = downloaded / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"uploaded file verification failed: {target.repo_id}/{name}")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied through the process environment")
    api = HfApi(token=token)
    identity = api.whoami()
    owner = str(identity.get("name", ""))
    if owner.casefold() != EXPECTED_OWNER.casefold():
        raise RuntimeError(f"refusing upload: authenticated owner is {owner!r}, expected {EXPECTED_OWNER!r}")

    with tempfile.TemporaryDirectory(prefix="swarm-hf-stage-") as staging_name, tempfile.TemporaryDirectory(
        prefix="swarm-hf-verify-"
    ) as verification_name:
        staging_root = Path(staging_name)
        verification_root = Path(verification_name)
        for target in TARGETS:
            staged, expected = stage(target, staging_root)
            api.create_repo(
                repo_id=target.repo_id,
                repo_type=target.repo_type,
                private=True,
                exist_ok=True,
            )
            commit = api.upload_folder(
                repo_id=target.repo_id,
                repo_type=target.repo_type,
                folder_path=staged,
                commit_message=f"Upload verified Swarm Arena artifacts from {CANONICAL_GIT_COMMIT}",
            )
            revision = str(commit.oid or api.repo_info(target.repo_id, repo_type=target.repo_type).sha)
            verify_snapshot(target, revision, expected, verification_root)
            total_bytes = sum((staged / name).stat().st_size for name in expected)
            print(
                json.dumps(
                    {
                        "repo": f"https://huggingface.co/{'datasets/' if target.repo_type == 'dataset' else ''}{target.repo_id}",
                        "revision": revision,
                        "verified_files": len(expected),
                        "verified_bytes": total_bytes,
                        "private": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
