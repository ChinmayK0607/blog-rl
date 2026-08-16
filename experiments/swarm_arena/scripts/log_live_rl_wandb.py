from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize_logical_update(record: dict[str, Any]) -> dict[str, int | float | str]:
    groups = list(record["groups"])
    if not groups:
        raise ValueError("logical update has no groups")
    returns: list[float] = []
    advantages: list[float] = []
    kinds: Counter[str] = Counter()
    stage_names: set[str] = set()
    returns_by_kind: dict[str, list[float]] = defaultdict(list)
    returns_by_opponent: dict[str, list[float]] = defaultdict(list)
    for group in groups:
        scenario = group["scenario"]
        kind = str(scenario.get("kind", "ordinary"))
        stage = scenario.get("curriculum_stage")
        if stage:
            stage_names.add(str(stage))
        kinds[kind] += 1
        group_returns = [float(row["return"]) for row in group["replicas"]]
        group_advantages = [float(row["advantage"]) for row in group["replicas"]]
        returns.extend(group_returns)
        advantages.extend(group_advantages)
        returns_by_kind[kind].extend(group_returns)
        opponent = str(scenario["opponent"]["family"])
        returns_by_opponent[opponent].extend(group_returns)
    metrics: dict[str, int | float | str] = {
        "controller/update": int(record["step"]) + 1,
        "controller/groups": len(groups),
        "controller/mean_terminal_return": statistics.mean(returns),
        "controller/mean_abs_advantage": statistics.mean(map(abs, advantages)),
        "controller/nonzero_advantage_rate": statistics.mean(
            abs(value) > 1e-12 for value in advantages
        ),
        "curriculum/stage": ",".join(sorted(stage_names)) or "legacy-fixed",
    }
    for kind, count in sorted(kinds.items()):
        metrics[f"curriculum/{kind}_fraction"] = count / len(groups)
        metrics[f"return/by_kind/{kind}"] = statistics.mean(returns_by_kind[kind])
    for opponent, values in sorted(returns_by_opponent.items()):
        metrics[f"return/by_opponent/{opponent}"] = statistics.mean(values)
    return metrics


def summarize_evaluation(summary: dict[str, Any]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    capability = summary.get("capability_rl_minus_sft", {})
    for suite in ("ordinary_legacy", "ordinary_hard"):
        if suite in capability:
            endpoint = capability[suite]
            metrics[f"eval/{suite}_rl_minus_sft"] = float(endpoint["mean_difference"])
    mappings = {
        "communication_effects": "eval/candidate_critical_normal_minus_dropped",
        "rl_specific_communication_lift": "eval/rl_specific_communication_lift",
        "critical_minus_decoy_specificity": "eval/critical_minus_decoy_specificity",
        "matched_decoy_normal_minus_dropped": "eval/matched_decoy_normal_minus_dropped",
    }
    for source, destination in mappings.items():
        endpoint = summary.get(source)
        if source == "communication_effects" and endpoint:
            endpoint = endpoint.get("normal_minus_dropped")
        if endpoint:
            metrics[destination] = float(endpoint["mean_difference"])
    for key, value in summary.get("candidate_protocol", {}).items():
        metrics[f"eval/protocol/{key}"] = float(value)
    for key, value in summary.get("claim_checks", {}).items():
        if isinstance(value, bool):
            metrics[f"eval/check/{key}"] = int(value)
    return metrics


def _read_json_retry(path: Path) -> Any:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            last_error = error
            time.sleep(0.2)
    assert last_error is not None
    raise last_error


def watcher_complete(
    logged_steps: set[int],
    *,
    expected_updates: int,
    finish_marker: Path | None,
) -> bool:
    training_complete = max(logged_steps, default=0) >= expected_updates
    evaluation_complete = finish_marker is None or finish_marker.is_file()
    return training_complete and evaluation_complete


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Failure-isolated W&B logger for Swarm Arena controller and eval metrics."
    )
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--project", default="swarm-arena-rl")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--group")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument(
        "--finish-marker",
        type=Path,
        help="when set, keep ingesting evaluations until this explicit marker exists",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--compact-artifact",
        action="append",
        type=Path,
        default=[],
        help="small file to bind into the final W&B run artifact; directories are rejected",
    )
    args = parser.parse_args()
    if args.expected_updates < 1 or args.poll_seconds <= 0:
        parser.error("expected updates and poll interval must be positive")

    import wandb

    run = wandb.init(
        project=args.project,
        id=args.run_id,
        resume="allow" if args.run_id else None,
        name=args.run_name,
        group=args.group,
        tags=args.tag,
        job_type="controller-evaluation",
        mode="offline" if args.offline else "online",
        config={
            "progress_path": str(args.progress),
            "eval_root": None if args.eval_root is None else str(args.eval_root),
            "expected_updates": args.expected_updates,
            "finish_marker": None if args.finish_marker is None else str(args.finish_marker),
        },
    )
    assert run is not None
    logged_steps: set[int] = set()
    logged_evals: set[Path] = set()
    while True:
        if args.progress.is_file():
            records = _read_json_retry(args.progress)
            for record in records:
                update = int(record["step"]) + 1
                if update in logged_steps:
                    continue
                metrics = summarize_logical_update(record)
                run.log(metrics, step=update)
                logged_steps.add(update)
        if args.eval_root is not None and args.eval_root.is_dir():
            for summary_path in sorted(args.eval_root.glob("update-*/summary.json")):
                if summary_path in logged_evals:
                    continue
                update = int(summary_path.parent.name.removeprefix("update-"))
                run.log(summarize_evaluation(_read_json_retry(summary_path)), step=update)
                logged_evals.add(summary_path)
        if args.once or watcher_complete(
            logged_steps,
            expected_updates=args.expected_updates,
            finish_marker=args.finish_marker,
        ):
            break
        time.sleep(args.poll_seconds)

    artifact = wandb.Artifact(f"{run.id}-compact-run-record", type="rl-run-record")
    if args.progress.is_file():
        artifact.add_file(str(args.progress), name="live_rl_progress.json")
    for path in args.compact_artifact:
        if not path.is_file():
            raise ValueError(f"compact artifact must be an existing file: {path}")
        artifact.add_file(str(path), name=f"inputs/{path.name}")
    if args.eval_root is not None:
        for summary_path in sorted(args.eval_root.glob("update-*/summary.json")):
            artifact.add_file(
                str(summary_path),
                name=f"eval/{summary_path.parent.name}-summary.json",
            )
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
