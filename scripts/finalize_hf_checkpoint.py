#!/usr/bin/env python3
"""Finalize a run's HF weight export into a clean, vLLM-loadable checkpoint.

Prime-RL keeps two on-disk formats: bulky DCP trainer checkpoints (``checkpoints/``,
tens of GB each, optimizer state, resume-only) and lean HF ``weights/step_N`` exports.
For eval/serving we only need the HF export -- but PPO exports carry a trainer-only
``value_head.*`` tensor that vLLM refuses to load ("no module named value_head").

This collapses a ``weights/step_N`` dir to a single ``model.safetensors`` with the
``value_head.*`` tensors dropped (no-op for GRPO exports, which have none), removing
the old shards + index. Aux files (config, tokenizer, chat template) are left in place.
Idempotent and safe: writes a temp file first, then swaps.

Usage:
  # finalize one step dir in place
  uv run --no-sync python scripts/finalize_hf_checkpoint.py <run>/weights/step_70
  # finalize every step dir under a run's weights/
  uv run --no-sync python scripts/finalize_hf_checkpoint.py --run-root <run>
"""

from __future__ import annotations

import argparse
import glob
import os

from safetensors.torch import load_file, save_file


def finalize_dir(step_dir: str) -> bool:
    shards = sorted(glob.glob(os.path.join(step_dir, "model-*-of-*.safetensors")))
    single = os.path.join(step_dir, "model.safetensors")
    index = os.path.join(step_dir, "model.safetensors.index.json")
    sources = shards if shards else ([single] if os.path.exists(single) else [])
    if not sources:
        print(f"  skip {step_dir}: no safetensors found")
        return False

    state: dict = {}
    for path in sources:
        for k, v in load_file(path).items():
            state[k] = v
    dropped = [k for k in state if k.startswith("value_head.")]
    for k in dropped:
        del state[k]

    already_single = bool(shards) is False and os.path.exists(single)
    if already_single and not dropped:
        print(f"  ok {step_dir}: already single-file, no value_head")
        return False

    tmp = single + ".tmp"
    save_file(state, tmp, metadata={"format": "pt"})
    for path in shards:
        os.remove(path)
    if os.path.exists(index):
        os.remove(index)
    os.replace(tmp, single)
    print(f"  finalized {step_dir}: {len(state)} tensors, dropped={dropped or 'none'}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step_dir", nargs="?", help="a single weights/step_N dir")
    ap.add_argument("--run-root", help="finalize every weights/step_* under this run root")
    args = ap.parse_args()

    if args.run_root:
        step_dirs = sorted(glob.glob(os.path.join(args.run_root, "weights", "step_*")))
        if not step_dirs:
            raise SystemExit(f"no weights/step_* under {args.run_root}")
        for d in step_dirs:
            finalize_dir(d)
    elif args.step_dir:
        finalize_dir(args.step_dir)
    else:
        raise SystemExit("provide a step_dir or --run-root")


if __name__ == "__main__":
    main()
