from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for step in rows:
        for group in step["groups"]:
            scenario = group["scenario"]
            key = (int(scenario["pair_index"]), str(scenario["world"]), str(scenario["receiver"]))
            cells[key].extend(group["replicas"])
    summaries = []
    receiver_nonzero: dict[str, list[bool]] = defaultdict(list)
    for (pair, world, receiver), replicas in sorted(cells.items()):
        effects = [float(row["message_effect"]) for row in replicas]
        advantages = [float(row["advantages"][receiver]) for row in replicas]
        nonzero_effects = sum(abs(value) > 1e-12 for value in effects)
        nonzero_advantages = sum(abs(value) > 1e-12 for value in advantages)
        receiver_nonzero[receiver].extend(abs(value) > 1e-12 for value in advantages)
        summaries.append(
            {
                "pair_index": pair,
                "world": world,
                "receiver": receiver,
                "replicas": len(replicas),
                "mean_normal_minus_dropped_return": statistics.fmean(effects),
                "nonzero_message_effect_count": nonzero_effects,
                "nonzero_advantage_count": nonzero_advantages,
                "unique_message_effects": sorted(set(effects)),
            }
        )
    receiver_rates = {
        receiver: statistics.fmean(values) if values else 0.0
        for receiver, values in sorted(receiver_nonzero.items())
    }
    passed = (
        set(receiver_rates) == {"blue-0", "blue-1"}
        and all(rate >= 0.25 for rate in receiver_rates.values())
        and all(row["nonzero_message_effect_count"] > 0 for row in summaries)
    )
    return {
        "version": "swarm-paired-signal-summary-v1",
        "status": "passed" if passed else "failed",
        "gate": "each world has a nonzero paired terminal effect and each receiver has >=25% nonzero centered advantages",
        "receiver_nonzero_advantage_rates": receiver_rates,
        "cells": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.diagnostic.read_text(encoding="utf-8")))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
