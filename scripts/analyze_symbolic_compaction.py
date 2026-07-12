#!/usr/bin/env python3
"""Offline segment / compaction-imbalance analysis for symbolic RL rollouts.

Consumes verifiers eval ``results.jsonl`` (records carry per-node ``token_ids``,
``sampled`` and ``mask``) and reports the compaction statistics that the
GRPO-vs-PPO-under-compaction hypothesis depends on:

  * segments-per-rollout distribution
  * within-prompt-group coefficient of variation (CV) of segment count
  * within-prompt-group CV of segment length
  * reward / pass-rate by horizon bucket
  * correlation between rollout "influence" (segment count) and reward

No model or GPU is needed: segmentation is recomputed directly from the stored
token ids using the same budget rule as
``prime_rl.orchestrator.trajectories._compaction_node_groups`` (message-aligned,
never cuts a token span). Keep this logic in sync with that function.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any


def _segment_groups(nodes: list[dict[str, Any]], token_budget: int) -> list[list[int]]:
    """Mirror of orchestrator.trajectories._compaction_node_groups for dict nodes."""
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


def _action_tokens(nodes: list[dict[str, Any]], group: list[int]) -> int:
    return sum(sum(1 for m in nodes[i].get("mask") or [] if m) for i in group)


def retokenize_nodes(nodes: list[dict[str, Any]], tokenizer) -> None:
    """Populate empty token_ids/mask on eval traces via the real chat template.

    Eval ``results.jsonl`` records omit token ids (only training rollouts keep
    them). We reconstruct per-node token counts by applying the model chat
    template incrementally, so segmentation matches training-time tokenization
    closely. This is a faithful proxy, not bit-identical to the online renderer.
    """
    if any(n.get("token_ids") for n in nodes):
        return
    per_message_overhead = 4  # role / delimiter tokens per chat message (approx)
    for node in nodes:
        msg = node["message"]
        if node.get("sampled") and msg.get("tool_calls"):
            text = "".join(
                f"{call.get('name', '')} {call.get('arguments', '')}" for call in msg["tool_calls"]
            )
        else:
            text = str(msg.get("content") or "")
        n_tok = len(tokenizer.encode(text, add_special_tokens=False)) + per_message_overhead
        node["token_ids"] = [0] * n_tok
        node["mask"] = [bool(node.get("sampled"))] * n_tok


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = st.mean(values)
    if mean == 0:
        return 0.0
    return st.pstdev(values) / mean


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def analyze(records: list[dict[str, Any]], token_budget: int, tokenizer=None) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rollouts: list[dict[str, Any]] = []

    for rec in records:
        spec = rec["task"]["spec"]
        nodes = rec["nodes"]
        if tokenizer is not None:
            retokenize_nodes(nodes, tokenizer)
        groups = _segment_groups(nodes, token_budget)
        seg_action_tokens = [_action_tokens(nodes, g) for g in groups]
        total_tokens = sum(len(n["token_ids"]) for n in nodes)
        roll = {
            "task_id": spec["task_id"],
            "horizon_bucket": spec["horizon_bucket"],
            "optimal_plan_length": spec["optimal_plan_length"],
            "imbalance_setting": spec["imbalance_setting"],
            "reward": float(rec["rewards"].get("success", 0.0)),
            "num_segments": len(groups),
            "seg_action_tokens": seg_action_tokens,
            "total_tokens": total_tokens,
            "num_turns": float(rec.get("metrics", {}).get("environment_turns", 0.0)),
            "stop_condition": rec.get("stop_condition", ""),
        }
        rollouts.append(roll)
        by_task[spec["task_id"]].append(roll)

    seg_counts = [r["num_segments"] for r in rollouts]
    seg_lengths = [t for r in rollouts for t in r["seg_action_tokens"]]

    # within-group (per prompt/task) statistics
    group_seg_cvs, group_len_cvs, group_seg_means = [], [], []
    for task_id, rs in by_task.items():
        counts = [r["num_segments"] for r in rs]
        group_seg_cvs.append(_cv([float(c) for c in counts]))
        group_seg_means.append(st.mean(counts))
        lens = [t for r in rs for t in r["seg_action_tokens"]]
        group_len_cvs.append(_cv([float(x) for x in lens]))

    # per-horizon breakdown
    horizons: dict[str, Any] = {}
    for h in sorted({r["horizon_bucket"] for r in rollouts}):
        subset = [r for r in rollouts if r["horizon_bucket"] == h]
        tasks_h = {tid: rs for tid, rs in by_task.items() if rs[0]["horizon_bucket"] == h}
        horizons[h] = {
            "num_rollouts": len(subset),
            "num_groups": len(tasks_h),
            "reward_mean": round(st.mean([r["reward"] for r in subset]), 4),
            "segments_per_rollout_mean": round(st.mean([r["num_segments"] for r in subset]), 3),
            "segments_per_rollout_max": max(r["num_segments"] for r in subset),
            "total_tokens_mean": round(st.mean([r["total_tokens"] for r in subset]), 1),
            "num_turns_mean": round(st.mean([r["num_turns"] for r in subset]), 2),
            "group_segment_count_cv_mean": round(
                st.mean([_cv([float(x["num_segments"]) for x in rs]) for rs in tasks_h.values()]), 4
            )
            if tasks_h
            else 0.0,
        }

    # pass@k buckets per group
    def bucket(rs: list[dict]) -> str:
        s = sum(r["reward"] == 1.0 for r in rs)
        return "all_fail" if s == 0 else "all_pass" if s == len(rs) else "mixed"

    buckets = {b: 0 for b in ("all_fail", "mixed", "all_pass")}
    for rs in by_task.values():
        buckets[bucket(rs)] += 1

    summary = {
        "token_budget": token_budget,
        "num_rollouts": len(rollouts),
        "num_groups": len(by_task),
        "reward_mean": round(st.mean([r["reward"] for r in rollouts]), 4) if rollouts else 0.0,
        "segments_per_rollout": {
            "mean": round(st.mean(seg_counts), 3),
            "p50": _pctl([float(x) for x in seg_counts], 0.5),
            "p90": _pctl([float(x) for x in seg_counts], 0.9),
            "max": max(seg_counts) if seg_counts else 0,
        },
        "segment_action_tokens": {
            "mean": round(st.mean(seg_lengths), 1) if seg_lengths else 0.0,
            "p90": round(_pctl([float(x) for x in seg_lengths], 0.9), 1),
            "max": max(seg_lengths) if seg_lengths else 0,
        },
        "within_group_segment_count_cv": {
            "mean": round(st.mean(group_seg_cvs), 4) if group_seg_cvs else 0.0,
            "p90": round(_pctl(group_seg_cvs, 0.9), 4),
            "max": round(max(group_seg_cvs), 4) if group_seg_cvs else 0.0,
            "frac_groups_cv_gt_0": round(
                sum(c > 1e-9 for c in group_seg_cvs) / len(group_seg_cvs), 3
            )
            if group_seg_cvs
            else 0.0,
        },
        "within_group_segment_length_cv_mean": round(st.mean(group_len_cvs), 4) if group_len_cvs else 0.0,
        "pass_at_k_bucket_counts": buckets,
        "mixed_fraction": round(buckets["mixed"] / len(by_task), 3) if by_task else 0.0,
        "corr_segments_vs_reward": round(
            _pearson([float(r["num_segments"]) for r in rollouts], [r["reward"] for r in rollouts]), 4
        ),
        "by_horizon": horizons,
    }
    return summary


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in paths:
        f = p / "eval" / "results.jsonl" if p.is_dir() and (p / "eval" / "results.jsonl").exists() else p
        for line in f.read_text().splitlines():
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="results.jsonl file(s) or run dir(s)")
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF model id to re-tokenize eval traces that lack token_ids (proxy).",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional summary.json output path")
    args = parser.parse_args()
    records = _load(args.results)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    summary = analyze(records, args.token_budget, tokenizer)
    summary["source"] = "retokenized_proxy" if tokenizer is not None else "stored_token_ids"
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
