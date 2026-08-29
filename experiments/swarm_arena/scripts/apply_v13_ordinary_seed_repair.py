#!/usr/bin/env python3
"""Apply a completed training-only seed search to the V13 signal screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_repair(manifest: dict[str, Any], search: dict[str, Any], *, search_sha256: str) -> dict[str, Any]:
    if search.get("training_only") is not True or search.get("optimizer_updates") != 0:
        raise ValueError("seed search must be training-only with zero optimizer updates")
    if search.get("sha256") != digest({key: value for key, value in search.items() if key != "sha256"}):
        raise ValueError("seed-search body checksum mismatch")
    selected = search.get("selected", [])
    if len(selected) != 8:
        raise ValueError("V13 seed repair requires exactly eight selected cells")
    cases = {str(row["case_id"]): dict(row) for row in manifest["cases"]}
    repaired: list[dict[str, Any]] = []
    for row in selected:
        case_id = str(row["case_id"])
        if row.get("accepted") is not True or case_id not in cases:
            raise ValueError(f"invalid selected seed-search row: {case_id}")
        case = cases[case_id]
        if (
            case["focused_agent"] != row["policy"]
            or case["opponent_family"] != row["family"]
        ):
            raise ValueError(f"selected cell identity mismatch: {case_id}")
        if not (
            float(row["return_range"]) > 0
            and int(row["action_diversity"]) >= 2
            and any(float(value) > 0 for value in row["advantages"])
            and any(float(value) < 0 for value in row["advantages"])
            and all(abs(float(value)) > 1e-12 for value in row["advantages"])
        ):
            raise ValueError(f"selected cell does not satisfy repair gate: {case_id}")
        case.update(seed=int(row["seed"]), size=int(row["size"]), horizon=int(row["horizon"]))
        repaired.append(
            {
                "case_id": case_id,
                "diagnostic_sha256": row["diagnostic_sha256"],
                "seed": int(row["seed"]),
                "size": int(row["size"]),
                "horizon": int(row["horizon"]),
            }
        )
    original_body = {key: value for key, value in manifest.items() if key != "sha256"}
    body = {
        **original_body,
        "cases": [cases[str(row["case_id"])] for row in manifest["cases"]],
        "seed_repair": {
            "version": "arena-rl-v13-ordinary-seed-repair-v1",
            "source_screen_sha256": manifest["sha256"],
            "search_file_sha256": search_sha256,
            "search_sha256": search["sha256"],
            "repaired_cells": sorted(repaired, key=lambda row: row["case_id"]),
            "optimizer_updates": 0,
            "frozen_data_opened": False,
        },
    }
    return {**body, "sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = apply_repair(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.search.read_text(encoding="utf-8")),
        search_sha256=file_sha256(args.search),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sha256": result["sha256"], "repaired_cells": 8}, sort_keys=True))


if __name__ == "__main__":
    main()
