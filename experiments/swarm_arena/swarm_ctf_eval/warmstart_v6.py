from __future__ import annotations

from pathlib import Path
from typing import Any

from .warmstart_v5 import write_dataset as write_v5_dataset

DATASET_VERSION = "arena-warmstart-v6"


def write_dataset(replay_rows_path: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return write_v5_dataset(
        replay_rows_path,
        output_dir,
        dataset_version=DATASET_VERSION,
        **kwargs,
    )
