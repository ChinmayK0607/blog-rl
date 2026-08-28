#!/usr/bin/env python3
"""Run the frozen V13 ordinary signal screen through rollout-only diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_opponent(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3 or parts[0] not in {"base", "sft", "historical", "current"}:
        raise argparse.ArgumentTypeError("opponent must be FAMILY:MODEL:REVISION")
    return parts[0], parts[1], parts[2]


def common_command(args: argparse.Namespace, source_commit: str) -> list[str]:
    command = [
        sys.executable,
        "experiments/swarm_arena/scripts/run_live_rl.py",
        "--trainer-config",
        str(args.trainer_config),
        "--inference-config",
        str(args.inference_config),
        "--data-dir",
        str(args.data_dir),
        "--task-data-version",
        "v4",
        "--tokenizer",
        args.tokenizer,
        "--initial-adapter",
        str(args.initial_adapter),
        "--initial-policy-adapter-manifest",
        str(args.initial_policy_adapter_manifest),
        "--source-commit",
        source_commit,
        "--base-revision",
        args.base_revision,
        "--initial-policy-revision",
        args.initial_policy_revision,
        "--credit-estimator",
        "shared_return",
        "--shared-return-replicas",
        "4",
        "--shared-return-credit-assignment",
        "focused_agent",
        "--shared-return-trainable-phase",
        "ACT",
        "--scenario-source",
        "ordinary",
        "--steps",
        "1",
        "--groups-per-step",
        "1",
        "--rollout-only",
    ]
    for base_url in args.base_url:
        command.extend(("--base-url", base_url))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--initial-policy-adapter-manifest", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--initial-policy-revision", required=True)
    parser.add_argument("--opponent", action="append", type=parse_opponent, required=True)
    args = parser.parse_args()
    opponents = {family: (model, revision) for family, model, revision in args.opponent}
    if set(opponents) != {"base", "sft", "historical", "current"}:
        raise ValueError("screen requires exactly one model-controlled opponent per family")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run_body = {
        "version": "arena-rl-v13-ordinary-signal-run-v1",
        "screen_manifest_sha256": manifest["sha256"],
        "screen_manifest_file_sha256": file_sha256(args.manifest),
        "source_commit": source_commit,
        "trainer_config_sha256": file_sha256(args.trainer_config),
        "inference_config_sha256": file_sha256(args.inference_config),
        "initial_policy_adapter_manifest_sha256": file_sha256(
            args.initial_policy_adapter_manifest
        ),
        "tokenizer": args.tokenizer,
        "base_revision": args.base_revision,
        "initial_policy_revision": args.initial_policy_revision,
        "base_urls": args.base_url,
        "opponents": {
            family: {"model": model, "revision": revision}
            for family, (model, revision) in sorted(opponents.items())
        },
        "optimizer_updates": 0,
    }
    run_manifest = {**run_body, "sha256": digest(run_body)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = args.output_dir / "RUN_MANIFEST.json"
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text(encoding="utf-8")) != run_manifest:
            raise ValueError("existing screen root has a different run manifest")
    else:
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    common = common_command(args, source_commit)
    for index, case in enumerate(manifest["cases"], start=1):
        case_dir = args.output_dir / case["case_id"]
        diagnostic = case_dir / "live_rl_diagnostic.json"
        if diagnostic.exists():
            print(json.dumps({"case": case["case_id"], "status": "already_complete"}))
            continue
        model, revision = opponents[case["opponent_family"]]
        command = common + [
            "--output-dir",
            str(case_dir),
            "--run-id",
            f"v13-ordinary-screen:{case['case_id']}",
            "--seed-base",
            str(case["seed"]),
            "--size",
            str(case["size"]),
            "--horizon",
            str(case["horizon"]),
            "--ordinary-focused-agent",
            case["focused_agent"],
            "--opponent-family",
            case["opponent_family"],
            "--opponent-model-name",
            model,
            "--opponent-revision",
            revision,
        ]
        subprocess.run(command, check=True)
        print(
            json.dumps(
                {
                    "case": case["case_id"],
                    "completed": index,
                    "total": len(manifest["cases"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
