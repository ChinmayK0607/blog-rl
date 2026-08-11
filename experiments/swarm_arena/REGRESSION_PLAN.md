# Regression plan

The adapter must not be promoted for MARL solely because arena-format metrics
improve. Every selected checkpoint is compared against its untouched base model
on identical frozen cases.

## Model regressions

The repository-owned `swarm-regression-v1` suite contains 256 exact, balanced
cases across:

- arithmetic composition;
- list filtering, deduplication, and sorting;
- instruction/key binding;
- unrelated network/node/agent prompts designed to detect arena-format leakage.

Promotion requires no more than a 2-point overall exactness drop, no more than a
5-point drop in any category, at most 2% arena-key leakage, and no leakage
increase over the base model. Rows are paired by immutable case ID.

Run both models with the same deterministic renderer and decoding settings:

```bash
python experiments/swarm_arena/scripts/score_regressions.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir experiments/swarm_arena/results/regression/base

python experiments/swarm_arena/scripts/score_regressions.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter /path/to/selected/lora_adapters \
  --output-dir experiments/swarm_arena/results/regression/adapter

python -m swarm_ctf_eval.regression_compare \
  --base-rows experiments/swarm_arena/results/regression/base/rows.jsonl \
  --adapter-rows experiments/swarm_arena/results/regression/adapter/rows.jsonl \
  --output experiments/swarm_arena/results/regression/comparison.json
```

Before a public capability claim, also run standard pinned versions of IFEval,
GSM8K, and ARC-Challenge. The repository suite is a sensitive LoRA
overspecialization check, not a replacement for broad benchmarks.

## Arena regressions

Track all of the following against both the untouched base and the selected SFT
warm start:

- strict action and grounded-message validity;
- action-option order consistency;
- terminal return and return by topology/horizon/opponent-switch slice;
- invalid actions, invalid broadcasts, and communication spend;
- generated-minus-dropped, generated-minus-shuffled, and
  generated-minus-one-turn-delayed return;
- zero-budget performance;
- specialization/collision rate and per-agent action entropy.

RL is promoted only if it improves terminal return without failing model or
arena regression gates. A return gain without communication-intervention gains
is capability learning, not evidence of swarm cooperation.
