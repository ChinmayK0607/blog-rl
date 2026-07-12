#!/usr/bin/env python3
"""Offline gradient-variance / SNR analysis for compaction objectives.

This is the proposal's Phase-1 "variance analysis": given a fixed set of collected
rollouts, estimate the gradient signal-to-noise ratio (SNR) under each training
objective *without* running a model forward. It answers the core question: does
variable rollout segmentation degrade GRPO's group-relative gradient, and does a
per-segment (PPO-style) credit assignment avoid that?

Model-free gradient model
--------------------------
All trainable tokens of a single rollout i push the policy in one shared direction
u_i (they raise the logprob of that rollout's own actions). Under the standard
isotropic-gradient-noise assumption the u_i are treated as orthonormal, so the
per-group policy-gradient estimate is

    g = sum_i  W_i * u_i

where W_i is the *total effective advantage weight* the objective places on rollout
i. The "ideal" update weights each rollout by its true (group-relative) advantage
A_i, i.e. g* = sum_i A_i * u_i. Because the u_i are orthonormal we can compute,
per prompt group, in closed form:

  - bias / baseline residual  B = sum_i W_i         (0 for a valid mean-zero group)
  - gradient alignment cosine cos(g, g*) = <W,A> / (||W|| ||A||)   (1.0 = ideal)
  - variance inflation         ||W||^2 / ||A||^2    (>1 => noisier gradient)

Effective per-rollout weights W_i by objective (S_i = #segments of rollout i)
-----------------------------------------------------------------------------
  full GRPO              W_i = A_i            (intact rollout, one inherited advantage)
  compacted GRPO         W_i = A_i * S_i      (every segment re-applies the advantage)
  segment-normalized     W_i = A_i            (divide segment influence by S_i)
  compacted PPO (pilot)  W_i = A_i * S_i      (rollout-reward critic => no per-segment
                                               differentiation; shares the misweighting)
  compacted PPO (ideal)  W_i = A_i            (a per-segment critic that credits each
                                               segment independently recovers alignment)

We also run a Monte-Carlo cross-check: draw random high-dim unit directions u_i,
bootstrap over groups, and report an empirical SNR = ||E[g]|| / RMS||g - E[g]||.

Usage:
  uv run --no-sync python scripts/analyze_gradient_snr.py \
      /home/ubuntu/semi/artifacts/cmp-compacted-grpo-v1/run_default/rollouts \
      --steps 10-70 --token-budget 384 [--mc-dim 2048 --mc-trials 400] [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np


# ----- segment reconstruction (mirrors the orchestrator / analyze_symbolic_compaction) -----
def segment_groups(nodes: list[dict[str, Any]], token_budget: int) -> list[list[int]]:
    """Split a rollout's sampled nodes into compacted segments by newly-introduced tokens.

    Returns a list of segments, each a list of node indices (sampled nodes only).
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


def action_tokens(nodes: list[dict[str, Any]], node_indices: list[int]) -> int:
    return int(sum(sum(nodes[i].get("mask") or []) for i in node_indices))


def parse_steps(spec: str) -> list[int] | None:
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def load_rollouts(rollouts_dir: str, steps: list[int] | None) -> list[dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(rollouts_dir, "step_*", "train_rollouts.jsonl")))
    rows: list[dict[str, Any]] = []
    for path in files:
        step = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if steps is not None and step not in steps:
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_step"] = step
                    rows.append(r)
    return rows


def group_key(r: dict[str, Any]) -> str:
    return r["task"].get("name") or str(r["task"].get("idx")) or r.get("info", {}).get("task_id", "?")


def reward_of(r: dict[str, Any]) -> float:
    rw = r.get("rewards") or {}
    return float(rw.get("success", next(iter(rw.values()), 0.0)))


# ----- objective effective weights -----
OBJECTIVES = ["full_grpo", "compacted_grpo", "segment_normalized_grpo", "compacted_ppo_pilot", "compacted_ppo_ideal"]


def effective_weight(objective: str, A_i: float, S_i: int) -> float:
    if objective == "full_grpo":
        return A_i
    if objective == "compacted_grpo":
        return A_i * S_i
    if objective == "segment_normalized_grpo":
        return A_i
    if objective == "compacted_ppo_pilot":
        return A_i * S_i
    if objective == "compacted_ppo_ideal":
        return A_i
    raise ValueError(objective)


def analyze(rows: list[dict[str, Any]], token_budget: int, mc_dim: int, mc_trials: int, seed: int) -> dict[str, Any]:
    # group rollouts by prompt
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["_step"], group_key(r))].append(r)

    # Per-group derived quantities
    per_obj_cos: dict[str, list[float]] = {o: [] for o in OBJECTIVES}
    per_obj_bias: dict[str, list[float]] = {o: [] for o in OBJECTIVES}
    per_obj_infl: dict[str, list[float]] = {o: [] for o in OBJECTIVES}
    seg_counts: list[int] = []
    seg_count_cv: list[float] = []
    corr_A_S: list[float] = []
    infl_by_segbucket: dict[str, list[float]] = defaultdict(list)  # influence share vs seg count
    n_mixed = 0
    n_groups = 0

    # Collect (A_i, S_i) and per-objective W vectors for MC
    mc_groups: list[dict[str, np.ndarray]] = []

    rng = np.random.default_rng(seed)

    for _, members in groups.items():
        if len(members) < 2:
            continue
        n_groups += 1
        rewards = np.array([reward_of(r) for r in members], dtype=float)
        if rewards.max() == rewards.min():
            continue  # all-pass or all-fail => zero advantage, no gradient
        n_mixed += 1
        A = rewards - rewards.mean()  # Dr.GRPO: mean-centered, no std norm
        S = np.array([max(1, len(segment_groups(r["nodes"], token_budget))) for r in members], dtype=float)
        seg_counts.extend(int(s) for s in S)
        if S.mean() > 0:
            seg_count_cv.append(float(S.std() / S.mean()))
        if S.std() > 0 and A.std() > 0:
            corr_A_S.append(float(np.corrcoef(A, S)[0, 1]))

        A_norm2 = float((A * A).sum())
        if A_norm2 == 0:
            continue

        W_by_obj: dict[str, np.ndarray] = {}
        for o in OBJECTIVES:
            W = np.array([effective_weight(o, a, int(s)) for a, s in zip(A, S)], dtype=float)
            W_by_obj[o] = W
            dot = float((W * A).sum())
            wn2 = float((W * W).sum())
            cos = dot / math.sqrt(wn2 * A_norm2) if wn2 > 0 else 0.0
            per_obj_cos[o].append(cos)
            per_obj_bias[o].append(abs(float(W.sum())) / (math.sqrt(A_norm2) + 1e-9))
            per_obj_infl[o].append(wn2 / A_norm2)

        # influence share by segment-count for compacted GRPO (does more segments => more pull?)
        infl = np.abs(W_by_obj["compacted_grpo"])
        if infl.sum() > 0:
            share = infl / infl.sum()
            for s_val, sh in zip(S, share):
                bucket = "S=1" if s_val <= 1 else ("S=2-3" if s_val <= 3 else "S>=4")
                infl_by_segbucket[bucket].append(float(sh * len(members)))  # >1 => over-represented

        mc_groups.append({"A": A, **{o: W_by_obj[o] for o in OBJECTIVES}})

    # ----- Monte-Carlo empirical gradient SNR -----
    # For each objective: g = sum_i W_i u_i, g* = sum_i A_i u_i (u_i random unit dirs).
    # SNR = ||mean over bootstrap of proj(g onto ghat*)|| / std, aggregated across groups.
    mc_snr: dict[str, float] = {}
    mc_cos: dict[str, float] = {}
    if mc_groups:
        for o in OBJECTIVES:
            projs = []
            coss = []
            for _ in range(mc_trials):
                g = mc_groups[rng.integers(len(mc_groups))]
                k = len(g["A"])
                U = rng.standard_normal((k, mc_dim))
                U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
                gvec = g[o] @ U
                gstar = g["A"] @ U
                nstar = np.linalg.norm(gstar) + 1e-12
                projs.append(float(gvec @ gstar) / nstar)  # signed projection onto ideal dir
                coss.append(float(gvec @ gstar) / (np.linalg.norm(gvec) * nstar + 1e-12))
            projs = np.array(projs)
            mc_snr[o] = float(abs(projs.mean()) / (projs.std() + 1e-12))
            mc_cos[o] = float(np.mean(coss))

    def stat(d: dict[str, list[float]], o: str) -> float:
        v = d[o]
        return float(np.mean(v)) if v else float("nan")

    return {
        "token_budget": token_budget,
        "n_groups": n_groups,
        "n_mixed_groups": n_mixed,
        "segments_per_rollout_mean": float(np.mean(seg_counts)) if seg_counts else float("nan"),
        "segments_per_rollout_max": int(np.max(seg_counts)) if seg_counts else 0,
        "group_segment_count_cv_mean": float(np.mean(seg_count_cv)) if seg_count_cv else float("nan"),
        "corr_advantage_segmentcount_mean": float(np.mean(corr_A_S)) if corr_A_S else float("nan"),
        "influence_share_by_segbucket": {k: float(np.mean(v)) for k, v in sorted(infl_by_segbucket.items())},
        "objectives": {
            o: {
                "grad_alignment_cos": stat(per_obj_cos, o),
                "baseline_bias": stat(per_obj_bias, o),
                "variance_inflation": stat(per_obj_infl, o),
                "mc_grad_snr": mc_snr.get(o, float("nan")),
                "mc_grad_cos": mc_cos.get(o, float("nan")),
            }
            for o in OBJECTIVES
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts_dir", help="path to <run>/run_default/rollouts")
    ap.add_argument("--steps", default="", help="e.g. '10-70' or '10,20,30' (default: all)")
    ap.add_argument("--token-budget", type=int, default=384)
    ap.add_argument("--mc-dim", type=int, default=2048)
    ap.add_argument("--mc-trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    steps = parse_steps(args.steps)
    rows = load_rollouts(args.rollouts_dir, steps)
    if not rows:
        raise SystemExit(f"no rollouts found under {args.rollouts_dir} for steps={args.steps!r}")
    res = analyze(rows, args.token_budget, args.mc_dim, args.mc_trials, args.seed)

    print(f"\n=== Gradient SNR / variance analysis (token_budget={res['token_budget']}) ===")
    print(f"rollouts={len(rows)}  groups={res['n_groups']}  mixed(usable)={res['n_mixed_groups']}")
    print(f"segments/rollout: mean={res['segments_per_rollout_mean']:.2f} max={res['segments_per_rollout_max']}"
          f"  within-group seg-count CV={res['group_segment_count_cv_mean']:.3f}")
    print(f"corr(advantage, segment_count) = {res['corr_advantage_segmentcount_mean']:.3f}")
    print(f"compacted-GRPO influence share (1.0=fair): {res['influence_share_by_segbucket']}")
    print(f"\n{'objective':<28} {'align_cos':>10} {'bias':>8} {'var_infl':>9} {'mc_snr':>8} {'mc_cos':>8}")
    for o, m in res["objectives"].items():
        print(f"{o:<28} {m['grad_alignment_cos']:>10.4f} {m['baseline_bias']:>8.3f} "
              f"{m['variance_inflation']:>9.3f} {m['mc_grad_snr']:>8.3f} {m['mc_grad_cos']:>8.4f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
