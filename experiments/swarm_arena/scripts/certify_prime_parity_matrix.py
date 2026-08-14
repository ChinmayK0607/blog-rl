from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a predeclared Prime trainer parity matrix on isolated GPUs."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[3]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.source_commit:
        parser.error(
            f"source commit {args.source_commit} does not match checked-out {actual_commit}"
        )
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    actual_probe_sha256 = sha256_file(args.probe)
    if actual_probe_sha256 != args.probe_sha256:
        parser.error(
            f"probe digest mismatch: expected {args.probe_sha256}, "
            f"got {actual_probe_sha256}"
        )
    if matrix.get("version") != "swarm-arena-parity-matrix-v1":
        parser.error("unknown parity matrix version")
    if matrix.get("selection_rule") != "first_passing_variant_in_declared_order":
        parser.error("parity matrix must predeclare first-passing selection")
    variants = matrix.get("variants")
    if not isinstance(variants, list) or not 1 <= len(variants) <= 4:
        parser.error("parity matrix requires between one and four variants")
    variant_ids = [row.get("id") for row in variants]
    if any(not isinstance(value, str) or not value for value in variant_ids):
        parser.error("parity matrix variant IDs must be non-empty strings")
    if len(set(variant_ids)) != len(variant_ids):
        parser.error("parity matrix variant IDs must be unique")

    commands = []
    for gpu_index, variant in enumerate(variants):
        trainer_config = repository_root / variant["trainer_config"]
        if not trainer_config.is_file():
            parser.error(f"trainer config does not exist: {trainer_config}")
        variant_dir = args.output_dir / "variants" / variant["id"]
        report = args.output_dir / "reports" / f"{variant['id']}.json"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            f"--rdzv-id={variant['id']}",
            str(Path(__file__).with_name("certify_prime_parity.py")),
            "--model",
            args.model,
            "--adapter",
            str(args.adapter),
            "--adapter-sha256",
            args.adapter_sha256,
            "--trainer-config",
            str(trainer_config),
            "--probe",
            str(args.probe),
            "--output-dir",
            str(variant_dir),
            "--report",
            str(report),
        ]
        commands.append(
            {
                "id": variant["id"],
                "gpu": gpu_index,
                "trainer_config": str(trainer_config),
                "trainer_config_sha256": sha256_file(trainer_config),
                "command": command,
                "report": report,
            }
        )

    plan = {
        "source_commit": args.source_commit,
        "matrix_sha256": sha256_file(args.matrix),
        "probe_sha256": actual_probe_sha256,
        "selection_rule": matrix["selection_rule"],
        "variants": [
            {
                "id": row["id"],
                "gpu": row["gpu"],
                "trainer_config": row["trainer_config"],
                "trainer_config_sha256": row["trainer_config_sha256"],
                "command": row["command"],
            }
            for row in commands
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse parity matrix output: {args.output_dir}")
    import torch

    if torch.cuda.device_count() < len(commands):
        raise RuntimeError(
            f"parity matrix requires {len(commands)} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    (args.output_dir / "logs").mkdir(parents=True)
    (args.output_dir / "reports").mkdir()
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    processes = []
    logs = []
    try:
        for row in commands:
            log_path = args.output_dir / "logs" / f"{row['id']}.log"
            log_handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
            process = subprocess.Popen(
                row["command"],
                cwd=repository_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((row, process))
            logs.append(log_handle)
        for _, process in processes:
            process.wait()
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log_handle in logs:
            log_handle.close()

    results = []
    for row, process in processes:
        report = None
        if row["report"].is_file():
            report = json.loads(row["report"].read_text(encoding="utf-8"))
        results.append(
            {
                "id": row["id"],
                "gpu": row["gpu"],
                "returncode": process.returncode,
                "trainer_config_sha256": row["trainer_config_sha256"],
                "report": report,
            }
        )
    selected = next(
        (
            row["id"]
            for row in results
            if row["report"] is not None
            and row["report"].get("parity_passed") is True
            and row["report"].get("isolation_passed") is True
        ),
        None,
    )
    summary = {
        **plan,
        "results": results,
        "selected_variant": selected,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_variant": selected, "variants": results}, sort_keys=True))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
