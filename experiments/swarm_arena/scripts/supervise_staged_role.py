"""Turn controller/pulse process failures into durable terminal signals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("controller", "pulses"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Python script and its arguments are required")
    code = subprocess.run([sys.executable, *command], check=False).returncode
    if code and not (args.run_dir / "evaluations/COMPLETE").exists():
        rejected = sorted((args.run_dir / "control/checkpoint_barriers").glob("step_*.rejected.json"))
        marker = args.run_dir / ("REJECTED.json" if rejected else "ABORTED.json")
        record = {
            "version": "staged-process-terminal-v1",
            "role": args.role,
            "returncode": code,
            "status": "scientific_gate_rejection" if rejected else "operational_abort",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "rejection_records": [str(path.relative_to(args.run_dir)) for path in rejected],
            "action": "stop_gpu_work_preserve_sync_verify_then_exact_pod_teardown",
        }
        if not marker.exists():
            temporary = marker.with_suffix(f".tmp-{os.getpid()}")
            with temporary.open("w") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Multiple failing roles preserve the first terminal cause.
            try:
                os.link(temporary, marker)
            except FileExistsError:
                pass
            temporary.unlink()
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
