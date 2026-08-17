# Swarm Arena RL progress evaluation v4

This evaluation measures learning during RL without repeatedly opening the
existing frozen v3 final. It separates gameplay capability from causal use of
communication.

## Capability suites

1. **Legacy ordinary OOD:** 24 unchanged maps with 14/16 nodes and 6/8-turn
   horizons, each run under three legal-option orders for 72 paired cells. This
   preserves comparability while testing option-order robustness; bootstrap
   inference still treats the 24 map seeds as the independent units.
2. **Hard ordinary OOD:** new seed-disjoint 18/20-node maps with 8/10-turn
   horizons. These test longer planning, larger observations, resource reuse,
   and adaptation to model-controlled opponents.

The primary capability endpoint is paired candidate-RL minus SFT terminal
return, side averaged and reported separately on both suites and every opponent.

## Communication suite

The new frozen handoff set contains 24 balanced sender/receiver bundles at
18/20 nodes and 8/10 turns. Each independent unit contains both latent worlds.
The candidate plays both sides against immutable base, SFT, and historical
league opponents under normal, dropped, sender-shuffled, delayed, and
zero-budget message conditions. Matched decoys use normal and dropped messages.

Communication requires all four paired critical intervention intervals to be
positive, positive mean effects against every opponent, and a matched-decoy
normal-minus-dropped interval containing zero. Higher ordinary return alone is
capability learning.

## Three evaluation tiers

- **Online monitor:** four development cases from each family, one SFT opponent,
  and normal/dropped conditions. This is a cheap directional signal and cannot
  support a research claim.
- **Checkpoint selection:** twelve development cases from each family, all
  three opponents, both sides, and all interventions. It selects at most one
  checkpoint.
- **Frozen final:** the full legacy, hard-ordinary, and two-world handoff suites,
  run once for the selected checkpoint.

All summaries bootstrap the seed or two-world bundle—not individual agents,
turns, sides, opponents, or latent worlds—as the independent unit. Existing
non-arena regression and collapse gates remain mandatory.

## Ten-update progress subset

Long runs preserve checkpoints every ten logical updates and compare them on a
fixed, paired development subset large enough to be useful as a learning
curve. It contains six legacy maps, six hard maps, and six two-world
critical/matched-decoy handoff bundles. Every case is played from both sides
against the immutable SFT opponent. Critical and decoy cases use normal and
dropped messaging; capability cases compare candidate RL directly with the SFT
initializer.

Update zero executes 192 complete games and stores 72 immutable SFT baseline
rows. Later checkpoints reuse those exact baseline rows and generate 120 new
candidate games. Each capability and causal-communication endpoint therefore
has six independent units, rather than the previous one-case/16-game smoke
test. The summary exposes legacy, hard, and handoff capability; critical
normal-minus-dropped; RL-specific communication lift; critical-minus-decoy
specificity; protocol validity; and an equal-family overall gameplay delta.
From the same rollouts it also reports active-target fact coverage by the
certified sender, critical capture-rate sensitivity to dropped messages, and
the RL-minus-SFT change in that capture sensitivity. These mechanism metrics
diagnose sender failure versus receiver failure; they do not add reward or
replace the causal return endpoints.

The same subset is intentionally repeated at every checkpoint so changes are
paired rather than confounded by different maps. It is a real development
signal for checkpoint selection, but remains separate from the untouched
multi-opponent selection tier and frozen final OOD evaluation.

## Run it

The runner is resumable and writes a content-bound manifest before its first
request. Replace the placeholder served-model names and immutable revisions in
`configs/progress_eval_v4.example.json`, then run the cheap online tier:

```bash
uv run --with ./experiments/swarm_arena \
  python experiments/swarm_arena/scripts/run_progress_eval_v4.py \
  --tier online \
  --config experiments/swarm_arena/configs/progress_eval_v4.example.json \
  --data-dir experiments/swarm_arena/data/rl_v4 \
  --output-dir /workspace/results/progress-v4-online
```

The compact development runner supports non-overlapping windows with
`--ordinary-offset` and `--curriculum-offset`. For example, after a pulse that
uses index zero, a larger follow-up starts both windows at index one.

The online tier executes 96 complete games: eight ordinary capability cells and
four two-world handoff bundles against the SFT opponent, on both sides, using
normal/dropped communication. Selection executes 1,296 games over 12 cases per
family, all opponents, both sides, and every intervention. Frozen final executes
3,168 games over 24 map seeds per family; it is intentionally expensive and is
run once only after selection.

`--tier frozen` additionally requires `--frozen-confirmation` equal to the exact
canonical SHA-256 printed by the runner when the argument is absent or wrong.
This makes opening the final set an explicit logged action rather than an
accidental default.
