import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from symbolic_tool_calling_v1.schemas import ArtifactManifest, ChecksumEntry


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any, *, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return f"{prefix}{digest}"


def token_count(text: str) -> int:
    """A deterministic byte-independent token proxy for offline pipeline validation."""
    import re

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _checksum(path: Path, root: Path) -> ChecksumEntry:
    data = path.read_bytes()
    return ChecksumEntry(path=str(path.relative_to(root)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def write_artifact(
    output_dir: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    artifact_version: str,
    config: BaseModel,
    records: Iterable[BaseModel],
    records_filename: str,
    summary: dict[str, Any],
    repo: Path,
) -> ArtifactManifest:
    output_dir.mkdir(parents=True, exist_ok=False)
    materialized = list(records)
    records_path = output_dir / records_filename
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    schema_path = output_dir / "schema_version.txt"

    records_path.write_text("".join(f"{canonical_json(record)}\n" for record in materialized))
    config_path.write_text(f"{canonical_json(config)}\n")
    summary_path.write_text(f"{canonical_json(summary)}\n")
    schema_path.write_text(f"{artifact_version}\n")
    checksums = tuple(_checksum(path, output_dir) for path in (records_path, config_path, summary_path, schema_path))
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        artifact_version=artifact_version,
        created_at=datetime.now(UTC),
        git_commit=git_commit(repo),
        config_snapshot=config.model_dump(mode="json"),
        records=len(materialized),
        summary_stats_path=summary_path.name,
        checksums=checksums,
    )
    (output_dir / "manifest.json").write_text(f"{canonical_json(manifest)}\n")
    return manifest


def read_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line]


def verify_artifact(output_dir: Path) -> ArtifactManifest:
    manifest = ArtifactManifest.model_validate_json((output_dir / "manifest.json").read_text())
    for expected in manifest.checksums:
        actual = _checksum(output_dir / expected.path, output_dir)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {expected.path}")
    return manifest
