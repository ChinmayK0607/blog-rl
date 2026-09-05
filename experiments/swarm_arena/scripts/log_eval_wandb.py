from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, int | float]:
    flattened: dict[str, int | float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_numbers(child, child_prefix))
    elif isinstance(value, bool):
        flattened[prefix] = int(value)
    elif isinstance(value, int | float):
        flattened[prefix] = value
    return flattened


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    resolved = Path(path)
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"summary does not exist: {resolved}")
    return label, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Log frozen evaluation summaries and raw artifacts to W&B.")
    parser.add_argument("--project", default="swarm-arena-sft")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-type", default="evaluation")
    parser.add_argument("--summary", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    args = parser.parse_args()

    run = wandb.init(project=args.project, name=args.run_name, job_type=args.job_type)
    assert run is not None
    artifact = wandb.Artifact(f"{run.id}-evaluation", type="evaluation")

    for label, path in args.summary:
        summary = json.loads(path.read_text(encoding="utf-8"))
        for key, value in flatten_numbers(summary, label).items():
            run.summary[key] = value
        artifact.add_file(str(path), name=f"summaries/{label}.json")

    for path in args.artifact:
        if path.is_dir():
            artifact.add_dir(str(path), name=f"raw/{path.name}")
        elif path.is_file():
            artifact.add_file(str(path), name=f"raw/{path.name}")
        else:
            parser.error(f"artifact does not exist: {path}")

    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
