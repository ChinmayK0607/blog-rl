from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

MIRROR_VERSION = "swarm-live-artifact-mirror-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def checkpoint_files(run_dir: Path, step: int, ready: dict[str, Any]) -> dict[str, Path]:
    expected = ready.get("policy_adapter_sha256")
    if not isinstance(expected, dict):
        raise ValueError(f"step {step} ready record lacks policy hashes")
    files: dict[str, Path] = {}
    for index in range(4):
        name = f"blue-{index}"
        root = run_dir / f"run_blue_{index}" / "checkpoints" / f"step_{step}"
        required = {
            f"checkpoints/step-{step}/policy-{name}/adapter_model.safetensors": (
                root / "weight" / "adapter_model.safetensors"
            ),
            f"checkpoints/step-{step}/policy-{name}/adapter_config.json": (
                root / "weight" / "adapter_config.json"
            ),
        }
        stable = (root / "STABLE", root / "weight" / "STABLE")
        if any(not path.is_file() for path in (*stable, *required.values())):
            raise FileNotFoundError(f"step {step} policy {name} is not a complete checkpoint")
        adapter_name = f"checkpoints/step-{step}/policy-{name}/adapter_model.safetensors"
        actual = sha256_file(required[adapter_name])
        if actual != expected.get(name):
            raise ValueError(f"step {step} policy {name} checksum mismatch")
        files.update(required)
    return files


def compact_files(run_dir: Path, extra_artifacts: tuple[Path, ...] = ()) -> dict[str, Path]:
    candidates = (
        "PREPARE.json",
        "PREFLIGHT.json",
        "live_rl_progress.json",
        "STATUS.md",
        "WATCHER_LATEST_STATUS.md",
        "logs/controller.log",
        "logs/trainer.log",
        "logs/pulses.log",
        "logs/wandb.log",
        "audit/rollout_parity_quarantine.jsonl",
    )
    files = {
        f"live/{relative}": run_dir / relative
        for relative in candidates
        if (run_dir / relative).is_file()
    }
    for artifact in extra_artifacts:
        if artifact.is_file():
            files[f"launch/{artifact.name}"] = artifact
    for path in sorted((run_dir / "evaluations").glob("update-*/*")):
        if path.is_file() and path.name in {"manifest.json", "rows.jsonl", "summary.json"}:
            files[f"evaluations/{path.parent.name}/{path.name}"] = path
    return files


@dataclass(frozen=True)
class MirrorState:
    mirrored_steps: tuple[int, ...] = ()
    last_progress_step: int = -1
    deadline_finalized: bool = False

    @classmethod
    def load(cls, path: Path) -> "MirrorState":
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != MIRROR_VERSION:
            raise ValueError("unknown live mirror state version")
        return cls(
            mirrored_steps=tuple(int(step) for step in payload.get("mirrored_steps", [])),
            last_progress_step=int(payload.get("last_progress_step", -1)),
            deadline_finalized=bool(payload.get("deadline_finalized", False)),
        )

    def write(self, path: Path) -> None:
        atomic_json(
            path,
            {
                "version": MIRROR_VERSION,
                "mirrored_steps": list(self.mirrored_steps),
                "last_progress_step": self.last_progress_step,
                "deadline_finalized": self.deadline_finalized,
            },
        )


def progress_step(path: Path) -> int:
    if not path.is_file():
        return -1
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("step"), int):
        return int(payload["step"])
    records = payload.get("updates", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return -1
    # Controller records are zero-indexed completed updates: record step 0 means
    # one optimizer update is durable.
    steps = [int(row["step"]) + 1 for row in records if isinstance(row, dict) and "step" in row]
    return max(steps, default=-1)


class LiveMirror:
    def __init__(
        self,
        *,
        repo_id: str,
        run_id: str,
        run_dir: Path,
        token: str,
        extra_artifacts: tuple[Path, ...] = (),
    ) -> None:
        self.repo_id = repo_id
        self.run_id = run_id
        self.run_dir = run_dir
        self.api = HfApi(token=token)
        self.public_api = HfApi(token=False)
        self.prefix = f"runs/{run_id}"
        self.extra_artifacts = extra_artifacts

    def _commit(
        self,
        files: dict[str, Path],
        *,
        message: str,
        extra: dict[str, bytes] | None = None,
    ) -> str:
        operations = [
            CommitOperationAdd(path_in_repo=f"{self.prefix}/{name}", path_or_fileobj=path)
            for name, path in sorted(files.items())
        ]
        for name, content in sorted((extra or {}).items()):
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{self.prefix}/{name}", path_or_fileobj=io.BytesIO(content)
                )
            )
        if not operations:
            raise ValueError("refusing to create an empty mirror commit")
        commit = self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="model",
            operations=operations,
            commit_message=message,
        )
        return str(commit.oid)

    def preflight(self) -> str:
        self.api.create_repo(self.repo_id, repo_type="model", private=False, exist_ok=True)
        info = self.public_api.model_info(self.repo_id)
        if info.private:
            raise RuntimeError("live artifact repository is not anonymously public")
        heartbeat = {
            "version": MIRROR_VERSION,
            "run_id": self.run_id,
            "created_unix": int(time.time()),
            "status": "preflight",
        }
        data = (json.dumps(heartbeat, indent=2, sort_keys=True) + "\n").encode()
        revision = self._commit(
            {}, message=f"Initialize live mirror for {self.run_id}", extra={"HEARTBEAT.json": data}
        )
        downloaded = Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="model",
                filename=f"{self.prefix}/HEARTBEAT.json",
                revision=revision,
                token=False,
            )
        )
        if downloaded.read_bytes() != data:
            raise RuntimeError("anonymous heartbeat verification failed")
        return revision

    def upload_compact(self, step: int, *, reason: str) -> str:
        manifest = {
            "version": MIRROR_VERSION,
            "run_id": self.run_id,
            "progress_step": step,
            "reason": reason,
            "mirrored_unix": int(time.time()),
        }
        return self._commit(
            compact_files(self.run_dir, self.extra_artifacts),
            message=f"Mirror {self.run_id} progress at step {step} ({reason})",
            extra={
                "live/MIRROR.json": (
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                ).encode()
            },
        )

    def upload_checkpoint(self, step: int, ready: dict[str, Any]) -> str:
        files = checkpoint_files(self.run_dir, step, ready)
        manifest = {
            "version": MIRROR_VERSION,
            "run_id": self.run_id,
            "step": step,
            "ready": ready,
            "files_sha256": {name: sha256_file(path) for name, path in sorted(files.items())},
            "mirrored_unix": int(time.time()),
        }
        revision = self._commit(
            files,
            message=f"Mirror complete four-policy checkpoint {self.run_id} step {step}",
            extra={
                f"checkpoints/step-{step}/MANIFEST.json": (
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                ).encode()
            },
        )
        for name, expected in manifest["files_sha256"].items():
            if not name.endswith("adapter_model.safetensors"):
                continue
            downloaded = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="model",
                    filename=f"{self.prefix}/{name}",
                    revision=revision,
                    token=False,
                )
            )
            if sha256_file(downloaded) != expected:
                raise RuntimeError(f"public checkpoint verification failed: {name}")
        return revision


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously mirror compact RL evidence and complete LoRA "
            "checkpoints off-node."
        )
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=int, required=True)
    parser.add_argument("--final-sync-margin", type=int, default=2700)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--compact-interval-steps", type=int, default=5)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.deadline_epoch <= int(time.time()):
        parser.error("deadline must be in the future")
    if args.final_sync_margin < 300:
        parser.error("final sync margin must be at least five minutes")
    if args.poll_seconds <= 0 or args.compact_interval_steps < 1:
        parser.error("poll and compact intervals must be positive")
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required for off-node mirroring")
    artifacts = tuple(args.artifact)
    if any(not path.is_file() for path in artifacts):
        raise FileNotFoundError("every bound launch artifact must be a file")
    if len({path.name for path in artifacts}) != len(artifacts):
        raise ValueError("bound launch artifact basenames must be unique")
    mirror = LiveMirror(
        repo_id=args.repo_id,
        run_id=args.run_id,
        run_dir=args.run_dir,
        token=token,
        extra_artifacts=artifacts,
    )
    revision = mirror.preflight()
    print(
        json.dumps(
            {"status": "preflight_verified", "revision": revision}, sort_keys=True
        ),
        flush=True,
    )
    if args.preflight_only:
        return

    state_path = args.run_dir / "control" / "live_mirror_state.json"
    state = MirrorState.load(state_path)
    while True:
        try:
            step = progress_step(args.run_dir / "live_rl_progress.json")
            mirrored = set(state.mirrored_steps)
            barrier_dir = args.run_dir / "control" / "checkpoint_barriers"
            for ready_path in sorted(barrier_dir.glob("step_*.ready.json")):
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                ready_step = int(ready["step"])
                if ready_step > 0 and ready_step not in mirrored:
                    mirror.upload_checkpoint(ready_step, ready)
                    mirrored.add(ready_step)
            deadline_due = time.time() >= args.deadline_epoch - args.final_sync_margin
            compact_due = (
                step >= 0
                and step != state.last_progress_step
                and (step == 0 or step % args.compact_interval_steps == 0)
            )
            if compact_due or (deadline_due and not state.deadline_finalized):
                mirror.upload_compact(step, reason="deadline" if deadline_due else "progress")
            state = MirrorState(
                mirrored_steps=tuple(sorted(mirrored)),
                last_progress_step=step if compact_due else state.last_progress_step,
                deadline_finalized=state.deadline_finalized or deadline_due,
            )
            state.write(state_path)
            status = {
                "version": MIRROR_VERSION,
                "checked_unix": int(time.time()),
                "progress_step": step,
                "mirrored_steps": list(state.mirrored_steps),
                "deadline_finalized": state.deadline_finalized,
                "status": "healthy",
            }
            atomic_json(args.run_dir / "LIVE_MIRROR_STATUS.json", status)
            print(json.dumps(status, sort_keys=True), flush=True)
            if (args.run_dir / "evaluations" / "COMPLETE").is_file():
                mirror.upload_compact(step, reason="complete")
                return
        except Exception as error:  # keep retrying without affecting the optimizer
            status = {
                "version": MIRROR_VERSION,
                "checked_unix": int(time.time()),
                "status": "retrying",
                "error": f"{type(error).__name__}: {error}",
            }
            atomic_json(args.run_dir / "LIVE_MIRROR_STATUS.json", status)
            print(json.dumps(status, sort_keys=True), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
