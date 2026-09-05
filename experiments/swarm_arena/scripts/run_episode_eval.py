from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_ctf_eval.episode_model_eval import run
from swarm_ctf_eval.local_hf import LocalHFArenaModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen multi-turn RL-native arena evaluation.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--batch-max-new-tokens", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = LocalHFArenaModel(args.model, args.adapter, args.batch_max_new_tokens)
    print(json.dumps(run(model, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
