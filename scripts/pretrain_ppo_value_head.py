#!/usr/bin/env python3
"""Offline pretraining of the PPO value head to kill the critic cold-start.

Prime-RL's PPO critic is a zero-initialized ``nn.Linear(hidden, 1)`` (bias-free)
and ``load_dcp_from_hf`` explicitly re-zeros it on every warm-start. On hard,
long-rollout tasks that means the first many PPO steps burn compute learning a
value scale from scratch while the policy is already competent -> noisy early
advantages. This script warm-starts the critic instead.

What it does
------------
Under the *exact policy PPO will initialize from* (``--model``, e.g. the full-GRPO
export used as the Phase-B warm start), it computes the hidden state at every
segment-start critic position -- the token just before a segment's first action,
which is precisely where ``stamp_ppo_streams`` trains the value head -- for a set
of previously collected rollouts, and ridge-regresses those hidden states onto the
rollout return (``rewards.success``, the PPO ``value_target``). The fitted weight
is saved as ``value_head.weight`` [1, hidden] and loaded via
``trainer.model.ppo_value_head_init``.

The backbone is frozen and the value head is linear, so pretraining is a single
closed-form ridge solve -- no optimizer loop, seconds of linear algebra after the
(no-grad) forward passes.

Usage:
  uv run --no-sync python scripts/pretrain_ppo_value_head.py \
      --model /home/ubuntu/semi/artifacts/cmp-full-grpo-v1/weights/step_90 \
      --rollouts /home/ubuntu/semi/artifacts/hardb-full-grpo-v1/run_default/rollouts \
      --steps 1-25 --token-budget 384 --max-rollouts 400 --max-seq-len 16384 \
      --out /home/ubuntu/semi/artifacts/hardb-ppo-warmcritic/value_head.safetensors
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def segment_groups(nodes: list[dict[str, Any]], token_budget: int) -> list[list[int]]:
    """Split a rollout's nodes into compacted segments by newly-introduced tokens.

    Mirrors the orchestrator / analyze_symbolic_compaction segmentation so the
    critic positions line up with what PPO actually trains on.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    introduced = 0
    previous_sampled = -1
    for index, node in enumerate(nodes):
        introduced += len(node["token_ids"])
        if not node.get("sampled") or not any(node.get("mask") or []):
            continue
        turn_tokens = sum(len(n["token_ids"]) for n in nodes[previous_sampled + 1 : index + 1])
        if current and introduced > token_budget:
            groups.append(current)
            current = []
            introduced = turn_tokens
        current.append(index)
        previous_sampled = index
    if current:
        groups.append(current)
    return groups


def reward_of(r: dict[str, Any]) -> float:
    rw = r.get("rewards") or {}
    return float(rw.get("success", next(iter(rw.values()), 0.0)))


def parse_steps(spec: str) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out


def load_rollouts(rollouts_dir: str, steps: set[int] | None, max_rollouts: int) -> list[dict[str, Any]]:
    files = sorted(
        glob.glob(os.path.join(rollouts_dir, "step_*", "train_rollouts.jsonl")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]),
    )
    rows: list[dict[str, Any]] = []
    for path in files:
        step = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if steps is not None and step not in steps:
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if len(rows) >= max_rollouts:
            break
    return rows[:max_rollouts]


def critic_positions(nodes: list[dict[str, Any]], token_budget: int) -> list[int]:
    """Flat-sequence index of the critic state (before first action) for each segment."""
    node_start: list[int] = []
    pos = 0
    for nd in nodes:
        node_start.append(pos)
        pos += len(nd["token_ids"])
    positions: list[int] = []
    for seg in segment_groups(nodes, token_budget):
        first_action: int | None = None
        for ni in seg:
            mask = nodes[ni].get("mask") or []
            for j, m in enumerate(mask):
                if m:
                    first_action = node_start[ni] + j
                    break
            if first_action is not None:
                break
        if first_action is not None:
            positions.append(max(first_action - 1, 0))
    return positions


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="policy PPO initializes from (HF dir)")
    ap.add_argument("--rollouts", required=True, help="<run>/run_default/rollouts dir with train_rollouts.jsonl")
    ap.add_argument("--steps", default="1-25", help="step range to draw (policy near PPO init), e.g. '1-25'")
    ap.add_argument("--token-budget", type=int, default=384)
    ap.add_argument("--max-rollouts", type=int, default=400)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--ridge", type=float, default=10.0, help="ridge (L2) regularization strength")
    ap.add_argument("--out", required=True, help="output value_head.safetensors path")
    args = ap.parse_args()

    steps = parse_steps(args.steps)
    rows = load_rollouts(args.rollouts, steps, args.max_rollouts)
    if not rows:
        raise SystemExit(f"no rollouts under {args.rollouts} for steps={args.steps!r}")
    print(f"loaded {len(rows)} rollouts from steps={args.steps}")

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    backbone = model.model  # base decoder: last_hidden_state is post-final-norm (== value-head input)
    hidden_size = model.config.hidden_size

    feats: list[torch.Tensor] = []
    targets: list[float] = []
    n_pos = 0
    for i, r in enumerate(rows):
        nodes = r["nodes"]
        flat = [t for nd in nodes for t in nd["token_ids"]]
        if not flat:
            continue
        positions = [p for p in critic_positions(nodes, args.token_budget) if p < args.max_seq_len]
        if not positions:
            continue
        ids = torch.tensor(flat[: args.max_seq_len], dtype=torch.long, device=device).unsqueeze(0)
        out = backbone(input_ids=ids, use_cache=False)
        h = out.last_hidden_state[0]  # [seq, hidden]
        reward = reward_of(r)
        for p in positions:
            feats.append(h[p].float().cpu())
            targets.append(reward)
        n_pos += len(positions)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} rollouts, {n_pos} critic states")

    if not feats:
        raise SystemExit("no critic states collected")
    H = torch.stack(feats).double()  # [N, hidden]
    y = torch.tensor(targets, dtype=torch.float64)  # [N]
    N = H.shape[0]
    print(f"collected {N} critic states; target mean={y.mean():.4f} std={y.std():.4f}")

    # Ridge regression (bias-free, matching nn.Linear(hidden,1,bias=False)):
    #   w = (H^T H + lambda I)^{-1} H^T y
    HtH = H.T @ H
    HtH.diagonal().add_(args.ridge)
    w = torch.linalg.solve(HtH, H.T @ y)  # [hidden]

    pred = H @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"fit: R^2={r2:.4f}  pred[mean={pred.mean():.4f} std={pred.std():.4f}]  ||w||={w.norm():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({"value_head.weight": w.to(torch.float32).reshape(1, hidden_size).contiguous()}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
