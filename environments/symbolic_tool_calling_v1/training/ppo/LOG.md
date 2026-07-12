# Vanilla PPO on the symbolic compaction benchmark — run log

## What we are trying, in short

Chinmay's study ([blog.md](../../blog.md), HackMD) compared GRPO variants against
"compacted PPO" on the symbolic tool-calling env and concluded PPO loses badly
(hard-val pass@4: full GRPO 0.93 vs compacted PPO 0.25 cold / 0.14 warm-critic).
But his `compacted_ppo` is not real PPO — every compacted segment just gets the
rollout's final reward as a critic target: no GAE, no per-state advantages, and
the policy credit is literally compacted-GRPO credit.

**This experiment runs the missing baseline: actual vanilla PPO** (Schulman 2017)
— clipped surrogate + a learned value head + GAE advantages computed in the
trainer from a per-token reward stream — on the same environment, same model
(Qwen/Qwen3-4B-Instruct-2507), same regime as his Phase-A `cmp-*` runs
(batch 64, group 8, 150 steps, val every 10 steps, temp 0.7, 4096 completion
tokens). Full rollouts, no compaction. If real PPO also fails here, the blog's
"just normalize GRPO by segment count" takeaway gets much stronger; if it works,
the PPO half of the story needs rewriting.

Numbers to beat/compare (his Phase A, medium curriculum, from base model):
| Run | Train reward | Best val |
|---|---|---|
| cmp-full-grpo-v1 | 0.875 | **1.000** |
| cmp-compacted-grpo-v1 | 0.875 | 0.986 |
| cmp-segnorm-grpo-v1 | 0.900 | 0.986 |
| cmp-compacted-ppo-v1 | — (150 steps, no in-run val logged) | — |
| **cmp-vanilla-ppo-v1 (ours)** | ? | ? |

## Setup deltas vs his runs (honest differences)

- **Hardware**: 8× RTX 6000 Ada 48GB on Lium (his: 8× H100 80GB). Same 5
  inference / 3 trainer split. Slower wall-clock, same math.
- **Taskset**: his exact `symbolic-curriculum-v1` (624 train / 69 val) lives on
  his box and its curation depends on stochastic frozen-model rollouts, so we
  rebuilt it the same way as `symbolic-curriculum-v2`: deterministic candidate
  pools (seeds 4001 / 6001) → frozen Qwen3-4B pass@4 sweep on the pod →
  keep pass@4-mixed tasks (1–3 of 4 solved) → ~600 tasks, ~10% val split by
  task_id hash. Same distribution, not the identical task list.
- **Algorithm**: `ppo` (ours) — per-token terminal-reward stream from the
  orchestrator, trainer-side GAE (gamma=1.0, lambda=0.95), clipped surrogate
  (eps=0.2), clipped value loss (c1=0.5), no entropy bonus. Config:
  [rl_qwen3_instruct_cmp_vanilla_ppo.toml](../../rl_qwen3_instruct_cmp_vanilla_ppo.toml).

## Timeline

- **2026-07-12**: Merged Chinmay's latest `synth-env` (Phase-A/B code + results)
  into `feat/vanilla-ppo`; fixed the ppo⇔ppo-loss config validator to accept his
  `compacted_ppo`; all algorithm/PPO unit tests green. Created
  `cmp-vanilla-ppo-v1` config. Rented Lium pod `lunar-matrix-19`
  (8× RTX 6000 Ada, $4.72/hr, 24h TTL). H100s ruled out on cost; no multi-GPU
  A100s were available.

## Results

_(pending — filled in from the wandb run `krishnapg2315/blog-rl/cmp-vanilla-ppo-v1`)_
