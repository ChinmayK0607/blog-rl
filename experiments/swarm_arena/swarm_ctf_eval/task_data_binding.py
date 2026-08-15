from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rl_v3 import RL_TASK_VERSION

RL_TASK_V4_VERSION = "arena-rl-v4-information-handoff"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class TaskDataBinding:
    task_version: str
    train_sha256: str
    development_sha256: str
    final_sha256: str
    curriculum_manifests: tuple[tuple[str, str], ...]

    def curriculum_manifest(self, split: str) -> str:
        manifests = dict(self.curriculum_manifests)
        try:
            return manifests[split]
        except KeyError as error:
            raise ValueError(f"unknown curriculum split {split!r}") from error


def resolve_task_data_binding(
    data_dir: Path,
    task_data_version: str,
) -> TaskDataBinding:
    """Bind a live RL run to one checked-in task-data generation."""
    index = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    if task_data_version == "v3":
        splits = index["splits"]
        manifests = tuple(
            sorted((split, str(row["manifest"])) for split, row in splits.items())
        )
        return TaskDataBinding(
            task_version=RL_TASK_VERSION,
            train_sha256=_require_sha256(
                splits["train"]["sha256"], "v3 train manifest"
            ),
            development_sha256=_require_sha256(
                splits["development"]["sha256"], "v3 development manifest"
            ),
            final_sha256=_sha256_file(data_dir / "final_eval_design.json"),
            curriculum_manifests=manifests,
        )
    if task_data_version == "v4":
        handoff = index["handoff"]
        ordinary = index["ordinary"]
        handoff_hashes = {
            split: _require_sha256(row["sha256"], f"v4 handoff {split} manifest")
            for split, row in handoff.items()
        }
        ordinary_hashes = {
            split: _require_sha256(row["sha256"], f"v4 ordinary {split} manifest")
            for split, row in ordinary.items()
        }
        return TaskDataBinding(
            task_version=RL_TASK_V4_VERSION,
            train_sha256=_canonical_sha256(
                {
                    "handoff": handoff_hashes["train"],
                    "curriculum_file": _sha256_file(data_dir / "curriculum.json"),
                }
            ),
            development_sha256=_canonical_sha256(
                {
                    "handoff": handoff_hashes["development"],
                    "ordinary": ordinary_hashes["development"],
                }
            ),
            final_sha256=_canonical_sha256(
                {
                    "handoff": handoff_hashes["frozen_ood"],
                    "ordinary": ordinary_hashes["frozen_ood"],
                    "progress_eval_design_file": _sha256_file(
                        data_dir / "progress_eval_design.json"
                    ),
                }
            ),
            curriculum_manifests=tuple(
                sorted(
                    (split, str(row["manifest"]))
                    for split, row in handoff.items()
                )
            ),
        )
    raise ValueError(f"unsupported task data version: {task_data_version}")
