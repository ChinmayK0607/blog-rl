# Regression-safe RL warm start

The first stage-1 adapter is not an eligible RL checkpoint. Its labels pass the
environment audit, but the training objective is too narrow and too strong for
the intended use as a protocol warm start.

## Diagnosis

The failure is instruction interference, not invalid arena labels or output-key
leakage:

- the 5,508-row source is dominated by 4,608 broadcast rows and includes 464
  identical empty-broadcast targets in the training split;
- the rank-32 adapter updates attention and MLP projections with 66.1 million
  trainable parameters;
- `1e-4` learning rate and 310 packed steps expose the adapter to roughly 20
  million tokens without any general-behavior replay;
- validation loss is already `0.1207` at step 40, where the frozen regression
  suite already loses 6.25 exactness points;
- failed responses remain valid JSON but bind requested values to the wrong
  output keys, which is characteristic of narrow schema imitation rather than
  generic corruption.

No checkpoint in the eight-point stage-1 sweep passes the paired regression
gate. Step 40 is least damaging but still loses 17.19 points on instruction
binding. The original step-240 selection loses 11.33 points overall.

## Replacement objective

`arena-warmstart-v3` teaches only the behavior required to enter RL safely:

- balanced turn-zero `BROADCAST` and `ACT` protocol examples;
- only prompt-visible, local-policy labels—no joint oracle or cooperation
  demonstrations;
- explicit empty, fact-only, fact-plus-intent, and full broadcast modes;
- exact instruction-preservation rehearsal on binding, filtering, arithmetic,
  and selection;
- deterministic base-model behavior replay on filtered UltraChat prompts.

The default train mix is 640 arena-protocol rows, 640 exact preservation rows,
and 1,280 base-behavior replay rows. The held-out validation split contains 96
arena rows and 96 disjoint preservation rows.

The LoRA dose is deliberately small: rank 8, Q/V attention projections only,
`1e-5` peak learning rate, 32 steps, and checkpoints every 8 steps. The training
goal is parser reliability with minimal movement from the base model, not policy
quality.

## Reproduction

All repository commands run through `uv`:

```bash
uv run --project /root/blog-rl --no-sync python -m scripts.extract_replay_prompts \
  --output data/warmstart_v3_work/replay_prompts.jsonl --examples 1800

uv run --project /root/blog-rl --no-sync python -m scripts.generate_base_replay \
  --prompts data/warmstart_v3_work/replay_prompts.jsonl \
  --output data/warmstart_v3_work/base_replay.jsonl

uv run --project /root/blog-rl --no-sync python -m scripts.build_warmstart_v3 \
  --replay-rows data/warmstart_v3_work/base_replay.jsonl \
  --output-dir data/warmstart_v3

uv run --project /root/blog-rl --no-sync python -m scripts.audit_warmstart_v3 \
  --dataset data/warmstart_v3 --output results/warmstart_v3/data_audit.json
```

Run training from the repository root with
`experiments/swarm_arena/configs/sft_warmstart_v3.toml`. Candidate checkpoints
are steps 8, 16, 24, and 32.

## Promotion rule

A candidate must pass all of these before RL:

1. strict arena JSON schema validity and legal actions on the v3 validation
   split;
2. no more than a 2-point overall exactness drop and no more than a 5-point drop
   in any category on each paired frozen regression suite;
3. no new arena-key leakage on unrelated prompts;
4. no catastrophic change in the untouched arena bootstrap evaluation.

Protocol-target exact match chooses among candidates that pass the safety
gates. It is not itself evidence of cooperation. If no adapter passes, the base
model plus constrained parsing remains the correct RL initialization.

## Results

Results are written only after all checkpoint comparisons have completed.
