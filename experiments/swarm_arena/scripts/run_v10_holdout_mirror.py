from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

VERSION = "swarm-v10-holdout-mirror-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.rstrip())


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class MirrorState:
    raw_rows: int = 0
    raw_byte_offset: int = 0
    shard_index: int = 0
    revision: str | None = None
    final_audit_digest: str | None = None

    @classmethod
    def load(cls, path: Path) -> MirrorState:
        if not path.is_file():
            return cls()
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            int(value["raw_rows"]),
            int(value["raw_byte_offset"]),
            int(value["shard_index"]),
            value.get("revision"),
            value.get("final_audit_digest"),
        )

    def write(self, path: Path) -> None:
        _atomic_json(
            path,
            {
                "raw_rows": self.raw_rows,
                "raw_byte_offset": self.raw_byte_offset,
                "shard_index": self.shard_index,
                "revision": self.revision,
                "final_audit_digest": self.final_audit_digest,
            },
        )


def _write_raw_shard(
    *,
    raw_path: Path,
    output_path: Path,
    start_offset: int,
    expected_evaluation_ids: list[str],
) -> int:
    """Write only raw records matching the next durable compact rows.

    A process interruption can leave one raw-first record without its compact
    row. On resume the evaluator may regenerate that game. Matching IDs here
    skips such orphans/duplicates without mutating the forensic local trace.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("rb") as source, gzip.open(output_path, "wb", compresslevel=6) as target:
        source.seek(start_offset)
        matched = 0
        while matched < len(expected_evaluation_ids):
            line = source.readline()
            if not line:
                raise ValueError("raw trace ended before the requested shard boundary")
            if not line.rstrip():
                continue
            record = json.loads(line)
            if str(record["evaluation_id"]) != expected_evaluation_ids[matched]:
                continue
            target.write(line)
            matched += 1
        return source.tell()


def _snapshot_rows(rows_path: Path, output_path: Path, stop: int) -> list[str]:
    evaluation_ids: list[str] = []
    with rows_path.open("rb") as source, output_path.open("wb") as target:
        for line in source:
            if not line.rstrip():
                continue
            if len(evaluation_ids) >= stop:
                break
            record = json.loads(line)
            evaluation_ids.append(str(record["evaluation_id"]))
            target.write(line)
    if len(evaluation_ids) != stop:
        raise ValueError(f"compact rows changed during snapshot: {len(evaluation_ids)} != {stop}")
    return evaluation_ids


def _verify_public(
    *,
    repo_id: str,
    revision: str,
    expected: dict[str, str],
) -> None:
    for path_in_repo, expected_sha256 in expected.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                revision=revision,
                token=False,
            )
        )
        if _sha256_file(downloaded) != expected_sha256:
            raise ValueError(f"anonymous mirror verification failed: {path_in_repo}")


def _sync(
    *,
    output_dir: Path,
    lock_path: Path,
    config_path: Path,
    repo_id: str,
    repo_path: str,
    chunk_rows: int,
    state: MirrorState,
) -> MirrorState:
    rows_path = output_dir / "rows.jsonl"
    raw_path = output_dir / "raw.jsonl"
    manifest_path = output_dir / "manifest.json"
    complete_path = output_dir / "COMPLETE"
    post_audit_paths = {
        "policy_kl_probe.json": output_dir / "policy_kl_probe.json",
        "policy_kl.json": output_dir / "policy_kl.json",
        "collapse_audit.json": output_dir / "collapse_audit.json",
    }
    # The evaluator fsync-order is raw first, then its compact row. A durable
    # row therefore certifies that the corresponding raw line is available.
    available = _line_count(rows_path)
    complete = complete_path.is_file()
    post_audits_complete = all(path.is_file() for path in post_audit_paths.values())
    final_audit_digest = (
        hashlib.sha256(
            "".join(f"{name}:{_sha256_file(path)}\n" for name, path in sorted(post_audit_paths.items())).encode()
        ).hexdigest()
        if post_audits_complete
        else None
    )
    remaining = available - state.raw_rows
    needs_final_audit_sync = complete and post_audits_complete and final_audit_digest != state.final_audit_digest
    if remaining < chunk_rows and not (complete and remaining > 0) and not needs_final_audit_sync:
        return state
    take = min(chunk_rows, remaining)
    start = state.raw_rows
    stop = start + take
    spool = output_dir / ".mirror_spool"
    spool.mkdir(parents=True, exist_ok=True)
    rows_snapshot = spool / "rows.jsonl"
    evaluation_ids = _snapshot_rows(rows_path, rows_snapshot, stop)
    shard = None
    next_offset = state.raw_byte_offset
    if take:
        shard = spool / f"raw-{start:06d}-{stop - 1:06d}.jsonl.gz"
        next_offset = _write_raw_shard(
            raw_path=raw_path,
            output_path=shard,
            start_offset=state.raw_byte_offset,
            expected_evaluation_ids=evaluation_ids[start:stop],
        )
    sync_manifest = spool / "SYNC_MANIFEST.json"
    payload = {
        "version": VERSION,
        "rows_available": available,
        "raw_rows_mirrored": stop,
        "complete": complete,
        "lock_sha256": _sha256_file(lock_path),
        "config_sha256": _sha256_file(config_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "rows_sha256": _sha256_file(rows_snapshot),
        "new_raw_shard": (
            {
                "start": start,
                "stop": stop,
                "sha256": _sha256_file(shard),
            }
            if shard is not None
            else None
        ),
        "post_audits_complete": post_audits_complete,
        "final_audit_digest": final_audit_digest,
    }
    _atomic_json(sync_manifest, payload)
    prefix = repo_path.strip("/")
    uploads = {
        f"{prefix}/LOCK.json": lock_path,
        f"{prefix}/config.json": config_path,
        f"{prefix}/manifest.json": manifest_path,
        f"{prefix}/rows.jsonl": rows_snapshot,
        f"{prefix}/SYNC_MANIFEST.json": sync_manifest,
    }
    if shard is not None:
        uploads[f"{prefix}/raw_shards/{shard.name}"] = shard
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        uploads[f"{prefix}/summary.json"] = summary_path
    if complete:
        uploads[f"{prefix}/COMPLETE"] = complete_path
    if post_audits_complete:
        uploads.update({f"{prefix}/{name}": path for name, path in post_audit_paths.items()})

    token = get_token()
    if not token:
        raise ValueError("Hugging Face authentication is unavailable on the host")
    result = HfApi(token=token).create_commit(
        repo_id=repo_id,
        repo_type="model",
        commit_message=(
            f"mirror v10 clean holdout rows {start}-{stop - 1}" if take else "mirror v10 clean holdout final audits"
        ),
        operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=local) for path, local in uploads.items()],
    )
    revision = result.oid
    _verify_public(
        repo_id=repo_id,
        revision=revision,
        expected={path: _sha256_file(local) for path, local in uploads.items()},
    )
    return MirrorState(
        stop,
        next_offset,
        state.shard_index + int(bool(take)),
        revision,
        final_audit_digest if post_audits_complete else state.final_audit_digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Periodically mirror resumable v10 held-out rows and compressed raw shards."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-id", default="CK0607/swarm-arena-live-runs")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--chunk-rows", type=int, default=100)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.chunk_rows < 1 or args.interval_seconds < 1:
        parser.error("chunk rows and interval must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / ".mirror_state.json"
    status_path = args.output_dir / "MIRROR_STATUS.json"
    state = MirrorState.load(state_path)
    while True:
        try:
            state = _sync(
                output_dir=args.output_dir,
                lock_path=args.lock,
                config_path=args.config,
                repo_id=args.repo_id,
                repo_path=args.repo_path,
                chunk_rows=args.chunk_rows,
                state=state,
            )
            state.write(state_path)
            rows = _line_count(args.output_dir / "rows.jsonl")
            complete = (args.output_dir / "COMPLETE").is_file()
            status = {
                "version": VERSION,
                "status": "healthy",
                "rows": rows,
                "raw_rows": ">=rows (raw-first evaluator write order)",
                "mirrored_raw_rows": state.raw_rows,
                "revision": state.revision,
                "complete": complete,
                "post_audits_complete": all(
                    (args.output_dir / name).is_file()
                    for name in (
                        "policy_kl_probe.json",
                        "policy_kl.json",
                        "collapse_audit.json",
                    )
                ),
                "checked_unix": int(time.time()),
            }
            _atomic_json(status_path, status)
            print(json.dumps(status, sort_keys=True), flush=True)
            if (
                complete
                and rows == state.raw_rows
                and status["post_audits_complete"]
                and state.final_audit_digest is not None
            ):
                return
            if rows - state.raw_rows >= args.chunk_rows:
                continue
        except Exception as error:
            status = {
                "version": VERSION,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "checked_unix": int(time.time()),
            }
            _atomic_json(status_path, status)
            print(json.dumps(status, sort_keys=True), flush=True)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
