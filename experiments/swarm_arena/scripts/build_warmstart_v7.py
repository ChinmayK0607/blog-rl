from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_ctf_eval.warmstart_v7 import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build observation-boundary replay-protected warm-start v7 data."
    )
    parser.add_argument("--replay-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.replay_rows, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
