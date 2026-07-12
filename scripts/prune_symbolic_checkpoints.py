#!/usr/bin/env python3
"""Keep only reward-best symbolic RL checkpoints.

Prime-RL's trainer checkpoint retention is step based (latest / interval).  For
the symbolic RL experiments we care about reward quality instead: keep the best
checkpoint by validation reward when validation metrics exist, and by train
reward otherwise.  The script also keeps the train-best checkpoint as a separate
candidate when validation exists, because a single validation sample can be
noisy and we do not want to discard the strongest train checkpoint by accident.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRAIN_ROLLOUT_PATTERNS = (
    "run_default/rollouts/step_{step}/train_rollouts.jsonl",
    "rollouts/step_{step}/train_rollouts.jsonl",
)

STEP_DIRS = ("checkpoints", "weights", "run_default/checkpoints", "run_default/broadcasts")


@dataclass(frozen=True)
class StepScore:
    step: int
    train_reward: float | None
    val_reward: float | None
    checkpoint_stable: bool
    weight_stable: bool
    broadcast_stable: bool

    @property
    def has_any_artifact(self) -> bool:
        return self.checkpoint_stable or self.weight_stable or self.broadcast_stable

    @property
    def has_saveable_checkpoint(self) -> bool:
        return self.checkpoint_stable or self.weight_stable


def _artifact_is_stable(path: Path, rel: str) -> bool:
    is_stable = (path / "STABLE").exists()
    if rel == "checkpoints":
        # Torch DCP checkpoints in these runs are usable once the DCP metadata
        # exists.  The trainer writes into step_* before metadata is finalized,
        # so an external pruner must treat metadata-less checkpoint dirs as
        # active/incomplete and leave them alone.
        is_stable = is_stable or (path / "trainer" / ".metadata").exists()
    return is_stable


def _step_from_name(path: Path) -> int | None:
    if not path.name.startswith("step_"):
        return None
    try:
        return int(path.name.removeprefix("step_"))
    except ValueError:
        return None


def _stable_steps(root: Path, rel: str) -> set[int]:
    base = root / rel
    if not base.exists():
        return set()
    out: set[int] = set()
    for child in base.glob("step_*"):
        step = _step_from_name(child)
        is_stable = _artifact_is_stable(child, rel)
        if step is not None and is_stable:
            out.add(step)
    return out


def _all_artifact_steps(root: Path) -> set[int]:
    steps: set[int] = set()
    for rel in STEP_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for child in base.glob("step_*"):
            step = _step_from_name(child)
            if step is not None:
                steps.add(step)
    return steps


def _rollout_reward(row: dict[str, Any]) -> float | None:
    if isinstance(row.get("reward"), int | float):
        return float(row["reward"])
    rewards = row.get("rewards")
    if isinstance(rewards, dict):
        vals = [float(v) for v in rewards.values() if isinstance(v, int | float)]
        if vals:
            return sum(vals)
    return None


def _mean_jsonl_reward(path: Path) -> float | None:
    rewards: list[float] = []
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            reward = _rollout_reward(row)
            if reward is not None and math.isfinite(reward):
                rewards.append(reward)
    if not rewards:
        return None
    return sum(rewards) / len(rewards)


def train_reward(root: Path, step: int) -> float | None:
    for pattern in TRAIN_ROLLOUT_PATTERNS:
        reward = _mean_jsonl_reward(root / pattern.format(step=step))
        if reward is not None:
            return reward
    return None


def _numeric_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            metrics.update(_numeric_metrics(value, name))
    elif isinstance(obj, int | float) and math.isfinite(float(obj)):
        metrics[prefix] = float(obj)
    return metrics


def _metric_is_val_reward(key: str) -> bool:
    lower = key.lower()
    if "train/" in lower:
        return False
    if not any(token in lower for token in ("eval", "val", "validation")):
        return False
    return any(token in lower for token in ("reward", "avg@", "pass@"))


def val_reward(root: Path, step: int) -> float | None:
    candidates = [
        root / "best_checkpoints_metrics.jsonl",
        root / "run_default" / "metrics.jsonl",
        root / "metrics.jsonl",
    ]
    values: list[float] = []
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("step", -1)) != step:
                    continue
                values.extend(v for k, v in _numeric_metrics(row).items() if _metric_is_val_reward(k))
    for path in list(root.glob(f"**/step_{step}/eval*.json")) + list(root.glob(f"**/step_{step}/val*.json")):
        with path.open() as f:
            row = json.load(f)
        values.extend(v for k, v in _numeric_metrics(row).items() if _metric_is_val_reward(k))
    for path in list(root.glob(f"**/step_{step}/eval*.jsonl")) + list(root.glob(f"**/step_{step}/val*.jsonl")):
        reward = _mean_jsonl_reward(path)
        if reward is not None:
            values.append(reward)
    if not values:
        return None
    return sum(values) / len(values)


def collect_scores(root: Path) -> list[StepScore]:
    checkpoint_steps = _stable_steps(root, "checkpoints")
    weight_steps = _stable_steps(root, "weights")
    broadcast_steps = _stable_steps(root, "run_default/broadcasts")
    steps = sorted(_all_artifact_steps(root) | checkpoint_steps | weight_steps | broadcast_steps)
    return [
        StepScore(
            step=step,
            train_reward=train_reward(root, step),
            val_reward=val_reward(root, step),
            checkpoint_stable=step in checkpoint_steps,
            weight_stable=step in weight_steps,
            broadcast_stable=step in broadcast_steps,
        )
        for step in steps
    ]


def _score_value(score: StepScore, field: str) -> tuple[float, float, int]:
    primary = getattr(score, field)
    secondary = score.train_reward if field == "val_reward" else score.val_reward
    return (
        float("-inf") if primary is None else primary,
        float("-inf") if secondary is None else secondary,
        score.step,
    )


def selected_steps(scores: list[StepScore], *, require_val: bool = False) -> set[int]:
    if require_val:
        candidates = [s for s in scores if s.has_saveable_checkpoint and s.val_reward is not None]
        if not candidates:
            raise ValueError("no validation-scored checkpoints found; refusing to prune by train reward")
        return {max(candidates, key=lambda s: _score_value(s, "val_reward")).step}

    candidates = [s for s in scores if s.has_saveable_checkpoint and s.train_reward is not None]
    if not candidates:
        candidates = [s for s in scores if s.has_saveable_checkpoint]
    if not candidates:
        return set()

    selected = {max(candidates, key=lambda s: _score_value(s, "train_reward")).step}
    val_candidates = [s for s in candidates if s.val_reward is not None]
    if val_candidates:
        selected.add(max(val_candidates, key=lambda s: _score_value(s, "val_reward")).step)
    return selected


def prune(root: Path, dry_run: bool, require_val: bool = False) -> dict[str, Any]:
    scores = collect_scores(root)
    keep = selected_steps(scores, require_val=require_val)
    deleted: list[str] = []
    protected_unstable: list[str] = []
    for rel in STEP_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for child in sorted(base.glob("step_*")):
            step = _step_from_name(child)
            if step is None or step in keep:
                continue
            if not _artifact_is_stable(child, rel):
                protected_unstable.append(str(child))
                continue
            deleted.append(str(child))
            if not dry_run:
                shutil.rmtree(child, ignore_errors=True)

    manifest = {
        "root": str(root),
        "dry_run": dry_run,
        "selected_steps": sorted(keep),
        "selection_policy": (
            "keep validation-best only"
            if require_val
            else "keep train-best and, when validation metrics exist, validation-best"
        ),
        "scores": [asdict(score) for score in scores],
        "deleted": deleted,
        "protected_unstable": protected_unstable,
    }
    if not dry_run:
        (root / "best_checkpoints.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-val",
        action="store_true",
        help="Prune strictly by validation reward; fail without a validation-scored checkpoint.",
    )
    args = parser.parse_args()

    manifests = [prune(root, args.dry_run, require_val=args.require_val) for root in args.run_roots]
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
