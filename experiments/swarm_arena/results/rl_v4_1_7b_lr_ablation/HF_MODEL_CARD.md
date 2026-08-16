---
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
tags:
  - reinforcement-learning
  - multi-agent
  - lora
  - development-checkpoint
---

# Qwen3-1.7B Swarm Arena RL v4 — LR 1e-5 step 8

This is a **public, development-only, non-admitted** bundle of four independent
role LoRA adapters (`blue-0` through `blue-3`) for the Swarm Arena research
environment. It is preserved for reproducibility and follow-up experiments,
not presented as a generally capable or communication-improved model.

## Lineage

- Base: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- SFT initializer:
  `CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible`, revision
  `534522a8f3ff3489b1dd8318dc8e533e51264cde`
- Training source commit:
  `12e0c461a28c3d0311d0353ab1ed45bcffb0b569`
- Evaluation source commit:
  `6d6fe88e`
- Optimizer learning rate: `1e-5`
- Exact update mix: 2 ordinary / 1 communication-critical / 1 matched decoy
- Reward: verified terminal team return only
- Policies: four distinct role adapters, independently optimized

## Development result

The selected step 8 improved ordinary candidate-minus-SFT return by
`+0.0707407` across 18 paired cells in a 198-game non-overlapping development
holdout. Both 256-case frozen non-arena regression suites passed for every
role, with zero arena-trigger leakage. Reference-state candidate-to-SFT KL was
small (mean `0.000782232`, p99 `0.0161926`).

However, normal-minus-dropped, sender-shuffled, delayed, and zero-message-budget
effects were all exactly `0` on the same holdout. The checkpoint therefore
shows an exploratory game-capability gain but **no demonstrated improvement in
causal communication or swarm cooperation**. It must not be cited as an
admitted final result.

Full compact evidence and launchers are public in
`ChinmayK0607/blog-rl`, branch `exp/swarm-arena-4b`, under
`experiments/swarm_arena/results/rl_v4_1_7b_lr_ablation/`.

## Adapter SHA-256

- `blue-0`: `9912e5c6faf412d716527fa9828390f1a9c773bedc87f8e6682cffa5bda742ab`
- `blue-1`: `927b7f32ba875aa9ddbafab7c613dc9d1f45853baf876cfda539d33ff6700237`
- `blue-2`: `8a1aa6de257ce25227653ca0cffb2c22c0ef235290954a8303703474c933b8a8`
- `blue-3`: `845d93ae936f483adfb80577b659612009a26d8f142f946e8d3c3282ef7ee270`
