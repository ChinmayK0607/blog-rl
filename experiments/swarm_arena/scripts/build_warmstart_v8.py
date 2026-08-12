from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_ctf_eval.warmstart_v8 import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sequential protocol warm-start v8 data.")
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.source_dataset, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
