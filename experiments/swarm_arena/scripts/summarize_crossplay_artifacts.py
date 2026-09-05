from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from swarm_ctf_eval.crossplay_eval import summarize, summarize_side_swapped


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and summarize a complete source-bound cross-play run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_record = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = {key: value for key, value in manifest_record.items() if key != "sha256"}
    manifest_sha256 = canonical_sha256(manifest)
    if manifest_record.get("sha256") != manifest_sha256:
        raise ValueError("cross-play manifest digest mismatch")
    rows_path = args.run_dir / "rows.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    expected = set()
    for seed, _, _ in manifest["cases"]:
        for blue_condition, red_condition in manifest["conditions"]:
            expected.add(
                (
                    seed,
                    manifest["blue_model"],
                    manifest["red_model"],
                    blue_condition,
                    red_condition,
                )
            )
            if manifest["swap_sides"] and manifest["blue_model"] != manifest["red_model"]:
                expected.add(
                    (
                        seed,
                        manifest["red_model"],
                        manifest["blue_model"],
                        red_condition,
                        blue_condition,
                    )
                )
    observed = [
        (
            row["seed"],
            row["blue_model"],
            row["red_model"],
            row["blue_condition"],
            row["red_condition"],
        )
        for row in rows
    ]
    if len(observed) != len(set(observed)):
        raise ValueError("cross-play rows contain duplicate experiment identities")
    if set(observed) != expected:
        raise ValueError(
            f"cross-play matrix is incomplete: expected {len(expected)}, got {len(observed)}"
        )
    for row in rows:
        if row["crossplay_version"] != manifest["version"]:
            raise ValueError("row cross-play version differs from manifest")
        if row["prompt_version"] != manifest["prompt_version"]:
            raise ValueError("row prompt version differs from manifest")

    repository_root = Path(__file__).resolve().parents[3]
    analysis_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    result = {
        "schema_version": "crossplay-artifact-summary-v1",
        "analysis_source_commit": analysis_commit,
        "run_source_commit": manifest["source_commit"],
        "manifest_sha256": manifest_sha256,
        "rows_sha256": sha256_file(rows_path),
        "rows_bytes": rows_path.stat().st_size,
        "matrix_complete": True,
        "row_count": len(rows),
        "condition_summary": summarize(rows, manifest_sha256),
        "side_swapped_summary": summarize_side_swapped(rows, manifest["blue_model"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["side_swapped_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
