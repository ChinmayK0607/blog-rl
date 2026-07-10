import json
from pathlib import Path

import pytest

from scripts.prune_symbolic_checkpoints import collect_scores, prune, selected_steps


def _write_jsonl(path: Path, rewards: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for reward in rewards:
            f.write(json.dumps({"rewards": {"success": reward}}) + "\n")


def _stable_step(root: Path, rel: str, step: int) -> None:
    path = root / rel / f"step_{step}"
    path.mkdir(parents=True)
    (path / "STABLE").touch()
    (path / "payload").write_text("x")


def _dcp_step(root: Path, step: int) -> None:
    path = root / "checkpoints" / f"step_{step}" / "trainer"
    path.mkdir(parents=True)
    (path / ".metadata").write_text("metadata")


def test_selects_train_best_when_no_validation_metrics(tmp_path: Path):
    for step, rewards in [(10, [0, 1]), (20, [1, 1]), (30, [0, 0])]:
        _stable_step(tmp_path, "checkpoints", step)
        _stable_step(tmp_path, "weights", step)
        _write_jsonl(tmp_path / "run_default" / "rollouts" / f"step_{step}" / "train_rollouts.jsonl", rewards)

    scores = collect_scores(tmp_path)

    assert selected_steps(scores) == {20}


def test_prune_keeps_union_of_train_best_and_val_best(tmp_path: Path):
    for step, rewards in [(10, [1, 1]), (20, [0, 1]), (30, [0, 0])]:
        _stable_step(tmp_path, "checkpoints", step)
        _stable_step(tmp_path, "weights", step)
        _stable_step(tmp_path, "run_default/broadcasts", step)
        _write_jsonl(tmp_path / "run_default" / "rollouts" / f"step_{step}" / "train_rollouts.jsonl", rewards)
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step": 10, "eval/symbolic/all/avg@1": 0.2}),
                json.dumps({"step": 20, "eval/symbolic/all/avg@1": 0.9}),
                json.dumps({"step": 30, "eval/symbolic/all/avg@1": 0.1}),
            ]
        )
        + "\n"
    )

    manifest = prune(tmp_path, dry_run=False)

    assert manifest["selected_steps"] == [10, 20]
    assert (tmp_path / "checkpoints" / "step_10").exists()
    assert (tmp_path / "checkpoints" / "step_20").exists()
    assert not (tmp_path / "checkpoints" / "step_30").exists()
    assert not (tmp_path / "weights" / "step_30").exists()
    assert not (tmp_path / "run_default" / "broadcasts" / "step_30").exists()
    assert json.loads((tmp_path / "best_checkpoints.json").read_text())["selected_steps"] == [10, 20]


def test_require_val_keeps_validation_best_only(tmp_path: Path):
    for step, rewards in [(10, [1, 1]), (20, [0, 1])]:
        _stable_step(tmp_path, "weights", step)
        _write_jsonl(tmp_path / "run_default" / "rollouts" / f"step_{step}" / "train_rollouts.jsonl", rewards)
    (tmp_path / "run_default" / "rollouts" / "step_10" / "eval_rollouts_symbolic-val.jsonl").write_text(
        json.dumps({"rewards": {"success": 0.2}}) + "\n"
    )
    (tmp_path / "run_default" / "rollouts" / "step_20" / "eval_rollouts_symbolic-val.jsonl").write_text(
        json.dumps({"rewards": {"success": 0.9}}) + "\n"
    )

    manifest = prune(tmp_path, dry_run=False, require_val=True)

    assert manifest["selected_steps"] == [20]
    assert not (tmp_path / "weights" / "step_10").exists()
    assert (tmp_path / "weights" / "step_20").exists()


def test_require_val_refuses_train_fallback(tmp_path: Path):
    _stable_step(tmp_path, "weights", 10)
    _write_jsonl(tmp_path / "run_default" / "rollouts" / "step_10" / "train_rollouts.jsonl", [1, 1])

    with pytest.raises(ValueError, match="no validation-scored checkpoints"):
        prune(tmp_path, dry_run=True, require_val=True)


def test_selection_ignores_broadcast_only_steps(tmp_path: Path):
    _stable_step(tmp_path, "run_default/broadcasts", 10)
    _write_jsonl(tmp_path / "run_default" / "rollouts" / "step_10" / "train_rollouts.jsonl", [1, 1])
    _stable_step(tmp_path, "weights", 20)
    _write_jsonl(tmp_path / "run_default" / "rollouts" / "step_20" / "train_rollouts.jsonl", [0, 1])

    manifest = prune(tmp_path, dry_run=False)

    assert manifest["selected_steps"] == [20]
    assert not (tmp_path / "run_default" / "broadcasts" / "step_10").exists()
    assert (tmp_path / "weights" / "step_20").exists()


def test_treats_dcp_metadata_checkpoint_as_saveable(tmp_path: Path):
    _dcp_step(tmp_path, 10)
    _write_jsonl(tmp_path / "run_default" / "rollouts" / "step_10" / "train_rollouts.jsonl", [1, 1])

    manifest = prune(tmp_path, dry_run=False)

    assert manifest["selected_steps"] == [10]
    assert (tmp_path / "checkpoints" / "step_10" / "trainer" / ".metadata").exists()
