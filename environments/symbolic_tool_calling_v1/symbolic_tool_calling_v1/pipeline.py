import argparse
import json
from pathlib import Path

from symbolic_tool_calling_v1.artifacts import canonical_json, read_jsonl, verify_artifact
from symbolic_tool_calling_v1.collection import (
    RolloutCollectionConfig,
    TaskDatasetConfig,
    write_rollout_dataset,
    write_task_dataset,
)
from symbolic_tool_calling_v1.compaction import CompactionConfig, write_compaction_dataset
from symbolic_tool_calling_v1.models import BenchmarkTask, FrozenModel
from symbolic_tool_calling_v1.schemas import CompactionSegment, RolloutRecord
from symbolic_tool_calling_v1.training import TrainingTransformConfig, write_training_dataset


class PipelineConfig(FrozenModel):
    tasks: TaskDatasetConfig = TaskDatasetConfig()
    rollouts: RolloutCollectionConfig = RolloutCollectionConfig()
    compaction: CompactionConfig = CompactionConfig()
    training: TrainingTransformConfig = TrainingTransformConfig()


def run_pipeline(output_dir: Path, config: PipelineConfig, repo: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "pipeline_config.json").write_text(f"{canonical_json(config)}\n")

    tasks_dir = output_dir / "tasks"
    task_manifest = write_task_dataset(tasks_dir, config.tasks, repo)
    verify_artifact(tasks_dir)
    tasks = read_jsonl(tasks_dir / "tasks.jsonl", BenchmarkTask)

    rollouts_dir = output_dir / "rollouts"
    rollout_manifest = write_rollout_dataset(rollouts_dir, tasks, config.rollouts, repo)
    verify_artifact(rollouts_dir)
    rollouts = read_jsonl(rollouts_dir / "rollouts.jsonl", RolloutRecord)

    compaction_dir = output_dir / "compaction"
    compaction_manifest = write_compaction_dataset(compaction_dir, rollouts, config.compaction, repo)
    verify_artifact(compaction_dir)

    training_dir = output_dir / "training"
    segments = read_jsonl(compaction_dir / "segments.jsonl", CompactionSegment)
    training_manifest = write_training_dataset(training_dir, segments, config.training, repo)
    verify_artifact(training_dir)

    index = {
        "tasks": task_manifest.model_dump(mode="json"),
        "rollouts": rollout_manifest.model_dump(mode="json"),
        "compaction": compaction_manifest.model_dump(mode="json"),
        "training": training_manifest.model_dump(mode="json"),
    }
    (output_dir / "pipeline_index.json").write_text(f"{canonical_json(index)}\n")
    return index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the symbolic benchmark artifact pipeline.")
    parser.add_argument("output_dir", type=Path, help="New, non-existing output directory")
    parser.add_argument("--config", type=Path, help="PipelineConfig JSON file")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Prime-RL repository root")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PipelineConfig()
    if args.config:
        config = PipelineConfig.model_validate(json.loads(args.config.read_text()))
    index = run_pipeline(args.output_dir, config, args.repo.resolve())
    print(json.dumps({name: manifest["artifact_id"] for name, manifest in index.items()}, indent=2))
