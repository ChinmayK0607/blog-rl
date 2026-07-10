# W&B run ledger — GRPO vs PPO under rollout compaction

**Project:** [`ChinmayK0604/blog-rl`](https://wandb.ai/ChinmayK0604/blog-rl) · Base model: `Qwen/Qwen3-4B-Instruct-2507` · Env: `symbolic_tool_calling_v1`

All runs share the symbolic tool-calling environment. `Train reward` = `train/agg/effective/reward/mean`
(last logged); `Best val` = max of `eval/<env>/all/rewards/success/mean` over the run. Many runs show
W&B state **crashed** because they were stopped by the tmux supervisor / GPU cleanup / a disk-full event
rather than exiting through W&B's clean shutdown — the logged metrics up to the last step are valid.
`—` means the metric was never logged (run died before its first eval, or in-run eval was disabled).

---

## Phase B — warm-started hard curriculum (group `symbolic-compaction-hard-v1`)

The headline experiment: does PPO's per-segment critic beat GRPO on genuinely hard, long-horizon tasks
(long+xlong, optimal plan 13 & 17) when **every arm is warm-started from the same full-GRPO policy**
(`cmp-full-grpo-v1` step 90) at `token_budget=384`? Plus a **value-head warm-start** ablation for PPO.

| Run | Run ID | Steps | Train reward | Best val | Status / role |
|---|---|---|---|---|---|
| hardb-full-grpo-v1 | `6b9ce13d` | 70 | ~0.90¹ | —² | Full GRPO on hard set (reference). Completed 70 steps. ²In-run val hit the tool-parser bug; scored offline. |
| hardb-compacted-grpo-v1 | `1bfed1ed` | 26+ | 0.543 | 0.500 | Compacted GRPO — **rerun in progress** (after disk fix + eval fix). |
| hardb-compacted-grpo-v1 | `344aa54e` | 21 | 0.543 | — | First attempt; killed by disk-full at step 21. |
| hardb-compacted-ppo-v1 | `ad12a5c5` | — | — | — | Compacted PPO, **cold critic**; crashed at inference start (disk-full). Rerun queued. |
| hardb-compacted-ppo-warmcritic-v1 | `66d9a055` | — | — | — | Compacted PPO, **warm critic** (value head pretrained offline, R²=0.71); crashed at start (disk-full). Rerun queued. |

¹ from the training log (`Step 70 | Reward 0.90`); this run's W&B train-agg summary key was not populated.

---

## Phase A — four regimes on `curriculum-v1` (group `symbolic-compaction-compare-v1`)

Head-to-head of the four training objectives on the medium curriculum, `token_budget=384`. Establishes
that segment-count imbalance degrades naive compacted GRPO and that segment-normalization recovers it;
compacted PPO is the critic-based comparison.

| Run | Run ID | Steps | Train reward | Best val | Status / role |
|---|---|---|---|---|---|
| cmp-full-grpo-v1 | `fe9eb643` | 116 | 0.875 | **1.000** | Full-rollout GRPO baseline (no compaction). Warm-start source (`weights/step_90`). |
| cmp-compacted-grpo-v1 | `1b673bbf` | 115 | 0.875 | 0.986 | Compacted GRPO `tb=384` (canonical run). |
| cmp-compacted-grpo-v1 | `825535b8` | 75 | 0.706 | 0.928 | Earlier attempt; orchestrator rollout-loop stall. |
| cmp-segnorm-grpo-v1 | `fcd9f62e` | 149 | 0.900 | 0.986 | Segment-normalized GRPO (weight 1/#segments). |
| cmp-compacted-ppo-v1 | `49a57276` | 150 | — | — | Compacted PPO, cold critic; trained 150 steps (in-run val not logged). |
| cmp-compacted-ppo-v1 | `7d5aa395` | 3 | 0.643 | — | Crashed: value head not populated in forward → fixed in `lm_head.py`. |
| cmp-compacted-ppo-v1 | `45ae7910` | 3 | 0.623 | — | Crashed: 0-dim PPO scalar metric `cat` → fixed in `train.py`. |

---

## Early exploration & pilots (groups `symbolic-compaction-rl`, `symbolic-compaction-rl-hard`)

Long-horizon exploration on the curated/long datasets that motivated the difficulty ladder and the
`token_budget` lever. Several compacted-GRPO-long runs saturated to val 1.0, which is exactly what drove
the move to a harder curriculum for Phase B.

| Run | Run ID | Steps | Train reward | Best val | Notes |
|---|---|---|---|---|---|
| qwen3-4b-instruct-full-grpo-long-v1 | `eb24ad6b` | 82 | 0.889 | — | Full GRPO, long data |
| qwen3-4b-instruct-full-grpo-long-v1 | `d9520caa` | 86 | 0.875 | — | Full GRPO, long data |
| qwen3-4b-instruct-full-grpo-long-v1 | `ce8a27df` | 31 | 0.809 | — | Full GRPO, long data |
| qwen3-4b-instruct-compacted-grpo-long-v1 | `7218b58e` | 63 | 0.933 | **1.000** | Compacted GRPO long — saturated |
| qwen3-4b-instruct-compacted-grpo-long-v1 | `732010ae` | 62 | 0.909 | **1.000** | Compacted GRPO long — saturated |
| qwen3-4b-instruct-compacted-grpo-long-v1 | `8a03dd2c` | 72 | 0.889 | — | Compacted GRPO long |
| qwen3-4b-instruct-compacted-grpo-long-v1 | `edc6aa67` | 44 | 0.886 | — | Compacted GRPO long |
| qwen3-4b-instruct-compacted-grpo-long-v1 | `535a641e` | — | — | — | Compacted GRPO long (early death) |
| qwen3-4b-instruct-compacted-grpo-long-v1 | `9b34254d` | — | — | — | Compacted GRPO long (early death) |
| qwen3-4b-instruct-segment-normalized-grpo-long-v1 | `a8ac0165` | 100 | — | — | Seg-norm GRPO long |
| qwen3-4b-instruct-segment-normalized-grpo-long-v1 | `6f962fa3` | 25 | 0.804 | — | Seg-norm GRPO long |
| qwen3-4b-instruct-compacted-grpo-pilot-v1 | `33f968be` | 20 | — | — | Pilot |
| qwen3-4b-instruct-segment-normalized-grpo-pilot-v1 | `9a8e088d` | 20 | — | — | Pilot |
| qwen3-4b-instruct-segment-normalized-grpo-pilot-v1 | `f3706b87` | — | — | — | Pilot (early death) |
| qwen3-4b-instruct-compacted-ppo-pilot-v1 | `2fcb59b3` | — | — | — | Pilot |
| qwen3-4b-instruct-compacted-ppo-pilot-v1 | `760a4893` | — | — | — | Pilot |
| qwen3-4b-instruct-hard-segment-normalized-grpo-pilot-v1 | `c0059db1` | — | — | — | Hard pilot |
| qwen3-4b-instruct-hard-compacted-ppo-pilot-v1 | `2a8e44e1` | — | — | — | Hard pilot |

---

### Canonical runs for the blog

| Regime | Run | Run ID |
|---|---|---|
| Full GRPO (baseline / warm-start) | cmp-full-grpo-v1 | `fe9eb643` |
| Compacted GRPO | cmp-compacted-grpo-v1 | `1b673bbf` |
| Segment-normalized GRPO | cmp-segnorm-grpo-v1 | `fcd9f62e` |
| Compacted PPO | cmp-compacted-ppo-v1 | `49a57276` |
| Hard / warm-start (Phase B) | hardb-* | `6b9ce13d`, `1bfed1ed`, + PPO reruns |

_Phase-B PPO rows (cold vs warm critic) will be updated with final step / val once the reruns complete._
