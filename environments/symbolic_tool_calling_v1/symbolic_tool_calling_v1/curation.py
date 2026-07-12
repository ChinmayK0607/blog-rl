import argparse
import json
from collections import defaultdict
from pathlib import Path

from pydantic import Field

from symbolic_tool_calling_v1.artifacts import stable_hash, write_artifact
from symbolic_tool_calling_v1.models import BenchmarkTask, FrozenModel

CURATION_VERSION = "1.0.0"


class CurationSource(FrozenModel):
    name: str
    results_jsonl: Path
    groups_jsonl: Path
    quota: int = Field(gt=0)
    allowed_horizons: tuple[str, ...] = ()


class CurationConfig(FrozenModel):
    curation_version: str = CURATION_VERSION
    sources: tuple[CurationSource, ...]
    allowed_success_counts: tuple[int, ...] = (1, 2, 3)


def _select_stratified(candidates: list[tuple[BenchmarkTask, int]], quota: int, strata: tuple[int, ...]):
    by_successes: dict[int, list[BenchmarkTask]] = defaultdict(list)
    for task, successes in candidates:
        if successes in strata:
            by_successes[successes].append(task)
    for tasks in by_successes.values():
        tasks.sort(key=lambda task: task.task_id)
    selected = []
    while len(selected) < quota:
        progressed = False
        for successes in strata:
            if by_successes[successes]:
                selected.append((by_successes[successes].pop(0), successes))
                progressed = True
                if len(selected) == quota:
                    break
        if not progressed:
            raise ValueError(f"only {len(selected)} eligible mixed tasks available for quota {quota}")
    return selected


def curate(config: CurationConfig) -> tuple[list[BenchmarkTask], dict]:
    selected_tasks: list[BenchmarkTask] = []
    source_summaries = {}
    for source in config.sources:
        records = [json.loads(line) for line in source.results_jsonl.read_text().splitlines() if line]
        spec_by_id = {
            record["task"]["spec"]["task_id"]: BenchmarkTask.model_validate(record["task"]["spec"])
            for record in records
        }
        groups = [json.loads(line) for line in source.groups_jsonl.read_text().splitlines() if line]
        candidates = [
            (spec_by_id[group["task_id"]], group["successes"])
            for group in groups
            if group["bucket"] == "mixed"
            and (not source.allowed_horizons or spec_by_id[group["task_id"]].horizon_bucket in source.allowed_horizons)
        ]
        selected = _select_stratified(candidates, source.quota, config.allowed_success_counts)
        selected_tasks.extend(task for task, _ in selected)
        source_summaries[source.name] = {
            "eligible": len(candidates),
            "selected": len(selected),
            "selected_by_successes": {
                str(successes): sum(selected_successes == successes for _, selected_successes in selected)
                for successes in config.allowed_success_counts
            },
        }
    if len({task.task_id for task in selected_tasks}) != len(selected_tasks):
        raise ValueError("curation sources produced duplicate task ids")
    summary = {
        "num_tasks": len(selected_tasks),
        "by_horizon": {
            horizon: sum(task.horizon_bucket == horizon for task in selected_tasks)
            for horizon in sorted({task.horizon_bucket for task in selected_tasks})
        },
        "sources": source_summaries,
    }
    return selected_tasks, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a balanced mixed-outcome symbolic RL taskset.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = CurationConfig.model_validate_json(args.config.read_text())
    tasks, summary = curate(config)
    manifest = write_artifact(
        args.output_dir,
        artifact_id=f"curated-tasks-{stable_hash(config.model_dump(mode='json'))[:12]}",
        artifact_type="tasks",
        artifact_version=config.curation_version,
        config=config,
        records=tasks,
        records_filename="tasks.jsonl",
        summary=summary,
        repo=args.repo.resolve(),
    )
    print(json.dumps({"artifact_id": manifest.artifact_id, **summary}, indent=2))
