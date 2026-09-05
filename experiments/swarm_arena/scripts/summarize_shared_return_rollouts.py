from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

VERSION = "arena-shared-return-rollout-summary-v2-content-addressed"
EPSILON = 1e-12


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family(group: dict[str, Any]) -> str:
    return str(group["scenario"].get("kind", "ordinary"))


def _metrics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    replicas = [replica for group in groups for replica in group["replicas"]]
    returns = [float(replica["return"]) for replica in replicas]
    advantages = [float(replica["advantage"]) for replica in replicas]
    varying_groups = sum(
        max(float(row["return"]) for row in group["replicas"])
        - min(float(row["return"]) for row in group["replicas"])
        > EPSILON
        for group in groups
    )
    return {
        "groups": len(groups),
        "replicas": len(replicas),
        "mean_return": statistics.fmean(returns),
        "return_stdev": statistics.pstdev(returns),
        "minimum_return": min(returns),
        "maximum_return": max(returns),
        "mean_absolute_advantage": statistics.fmean(abs(row) for row in advantages),
        "nonzero_advantage_rate": statistics.fmean(
            abs(row) > EPSILON for row in advantages
        ),
        "positive_advantages": sum(row > EPSILON for row in advantages),
        "negative_advantages": sum(row < -EPSILON for row in advantages),
        "zero_advantages": sum(abs(row) <= EPSILON for row in advantages),
        "return_variance_group_rate": varying_groups / len(groups),
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one rollout diagnostic is required")
    groups = []
    inputs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_groups = [group for step in payload for group in step["groups"]]
        if not file_groups:
            raise ValueError(f"rollout diagnostic contains no groups: {path}")
        groups.extend(file_groups)
        inputs.append(
            {
                "sha256": _sha256_file(path),
                "groups": len(file_groups),
            }
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_family[_family(group)].append(group)

    paired: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in groups:
        scenario = group["scenario"]
        kind = scenario.get("kind")
        namespace = scenario.get("sampling_namespace")
        pair_index = scenario.get("pair_index")
        if kind in {"critical", "decoy"} and namespace and pair_index is not None:
            paired[(int(pair_index), str(namespace))][str(kind)] = group
    paired_differences = []
    for kinds in paired.values():
        if set(kinds) != {"critical", "decoy"}:
            continue
        paired_differences.extend(
            float(critical["return"]) - float(decoy["return"])
            for critical, decoy in zip(
                kinds["critical"]["replicas"],
                kinds["decoy"]["replicas"],
                strict=True,
            )
        )

    return {
        "version": VERSION,
        "inputs": sorted(inputs, key=lambda row: row["sha256"]),
        "overall": _metrics(groups),
        "families": {
            family: _metrics(rows) for family, rows in sorted(by_family.items())
        },
        "paired_handoff": {
            "complete_pairs": len(paired_differences) // 4,
            "replica_differences": len(paired_differences),
            "critical_minus_decoy_mean_return": (
                statistics.fmean(paired_differences) if paired_differences else None
            ),
        },
        "scope": (
            "rollout reward/advantage-density diagnostic only; no optimizer update "
            "and no claim of learned communication"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize one or more shared-return rollout-only diagnostics."
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(summarize(args.inputs), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
