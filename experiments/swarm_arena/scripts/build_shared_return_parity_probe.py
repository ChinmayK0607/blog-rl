from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_ctf_eval.shared_return_parity import build_shared_return_parity_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a trainer-parity probe from verified shared-return evidence."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite parity probe: {args.output}")
    probe = build_shared_return_parity_probe(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(probe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print({"output": str(args.output), "samples": len(probe["samples"])})


if __name__ == "__main__":
    main()
