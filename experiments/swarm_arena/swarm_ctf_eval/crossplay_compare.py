from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .crossplay_eval import CROSSPLAY_VERSION, _mean_ci, _paired_randomization_p


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _paired_focal_returns(
    rows: list[dict[str, Any]],
    focal_policy: str,
) -> dict[int, float]:
    by_seed: dict[int, dict[str, float]] = {}
    opponents = ({row["blue_model"] for row in rows} | {row["red_model"] for row in rows}) - {
        focal_policy
    }
    if len(opponents) != 1:
        raise ValueError("history comparison requires exactly one focal and one opponent policy")
    opponent = next(iter(opponents))
    for row in rows:
        if row["blue_condition"] != "generated" or row["red_condition"] != "generated":
            continue
        if {row["blue_model"], row["red_model"]} != {focal_policy, opponent}:
            raise ValueError("all history rows must use the same policy pair")
        orientation = "blue" if row["blue_model"] == focal_policy else "red"
        value = float(row["metrics"][orientation.upper()]["terminal_return"])
        seed_rows = by_seed.setdefault(int(row["seed"]), {})
        if orientation in seed_rows:
            raise ValueError(f"duplicate {orientation} side for seed {row['seed']}")
        seed_rows[orientation] = value
    incomplete = [seed for seed, values in by_seed.items() if set(values) != {"blue", "red"}]
    if incomplete:
        raise ValueError(f"incomplete side swaps for seeds: {incomplete}")
    return {
        seed: statistics.mean((values["blue"], values["red"]))
        for seed, values in sorted(by_seed.items())
    }


def _paired_effect(left: dict[int, float], right: dict[int, float]) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired runs must contain identical seeds")
    differences = [left[seed] - right[seed] for seed in sorted(left)]
    return {
        "paired_seeds": len(differences),
        "mean_return_difference": statistics.mean(differences),
        "mean_return_difference_95": _mean_ci(differences),
        "randomization_p_two_sided": _paired_randomization_p(differences),
        "positive_seed_rate": statistics.mean(value > 0 for value in differences),
        "seed_differences": [
            {"seed": seed, "return_difference": difference}
            for seed, difference in zip(sorted(left), differences, strict=True)
        ],
    }


def compare_history_runs(
    normal_dir: Path,
    focal_no_history_dir: Path,
    opponent_no_history_dir: Path,
    focal_policy: str,
) -> dict[str, Any]:
    directories = (normal_dir, focal_no_history_dir, opponent_no_history_dir)
    manifests = [_read_json(directory / "manifest.json") for directory in directories]
    for manifest in manifests:
        if manifest["version"] != CROSSPLAY_VERSION:
            raise ValueError("history run uses a different cross-play version")
    identity_fields = ("blue_model", "blue_artifact_id", "red_model", "red_artifact_id", "cases")
    for field in identity_fields:
        if any(manifest[field] != manifests[0][field] for manifest in manifests[1:]):
            raise ValueError(f"history manifests disagree on {field}")
    expected_windows = ((3, 3), (0, 3), (3, 0))
    observed_windows = tuple(
        (manifest["blue_history_window"], manifest["red_history_window"])
        for manifest in manifests
    )
    if observed_windows != expected_windows:
        raise ValueError(
            f"expected normal/focal-off/opponent-off windows {expected_windows}, got {observed_windows}"
        )

    normal = _paired_focal_returns(_read_rows(normal_dir), focal_policy)
    focal_no_history = _paired_focal_returns(_read_rows(focal_no_history_dir), focal_policy)
    opponent_no_history = _paired_focal_returns(_read_rows(opponent_no_history_dir), focal_policy)
    return {
        "crossplay_version": CROSSPLAY_VERSION,
        "focal_policy": focal_policy,
        "paired_seed_count": len(normal),
        "focal_history_benefit": {
            "definition": "focal history-3 return minus focal history-0 return, opponent held at history 3",
            **_paired_effect(normal, focal_no_history),
        },
        "opponent_history_pressure": {
            "definition": "opponent history-0 return minus opponent history-3 return for the focal policy, focal held at history 3; positive means the adaptive opponent is harder",
            **_paired_effect(opponent_no_history, normal),
        },
        "interval_method": "paired seed-level nonparametric bootstrap, 20000 deterministic resamples",
        "test_method": "paired two-sided sign randomization, exact for this 8-seed development suite",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare asymmetric cross-play history ablations.")
    parser.add_argument("--normal-dir", type=Path, required=True)
    parser.add_argument("--focal-no-history-dir", type=Path, required=True)
    parser.add_argument("--opponent-no-history-dir", type=Path, required=True)
    parser.add_argument("--focal-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_history_runs(
        args.normal_dir,
        args.focal_no_history_dir,
        args.opponent_no_history_dir,
        args.focal_policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
