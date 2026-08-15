# Swarm Arena RL v4: 30-update development run

This is the compact record of a fresh 30-update, four-policy Qwen3-1.7B LoRA
run. It began from the pinned step-320 SFT adapter and used the locked 50/25/25
ordinary/communication-critical/matched-decoy schedule. Each BLUE role retained
its own adapter and optimizer state. The only reward was verified terminal team
return; there was no speaking, action, capture, or learned-judge bonus.

## Training

- 30 updates, 120 complete groups, and 480 game replicas were admitted with no
  rejection.
- Mean return was `+0.01735`, mean absolute advantage was `0.09640`, and the
  mean non-zero-advantage rate was `86.25%`. Every update had learning signal.
- All four policy adapters were distinct and all four changed at every update.
- All 120 policy batches passed the locked aggregate actor/trainer envelope.
  Worst observed values were `0.00391` mean log-probability error, `0.12181`
  p99 log-probability error, `0.04886` p99 probability error, `0.00953` tail
  fraction, and `0.000457` mean mismatch KL.

## Checkpoint selection

A fixed 66-game pulse compared an independent earlier step-8 run with fresh
same-lineage steps 18, 20, and 30. The predeclared development score selected
step 20. On the single pulse case, its candidate-minus-SFT capability result
was `+0.1071`; this was treated only as a noisy selection signal.

The selected checkpoint then ran a non-overlapping 198-game development
holdout using three ordinary cases and three critical/decoy pairs, three
opponents, both sides, and all message interventions:

- candidate-minus-SFT ordinary return: `+0.00344` (effectively flat);
- critical normal-minus-dropped, shuffled, delayed, and zero-budget: `+0.04444`
  for each intervention;
- matched-decoy normal-minus-dropped: `-0.05797`;
- action protocol, broadcast protocol, and broadcast grounding: `1.0`.

The holdout therefore contains information-specific communication sensitivity:
real messages help on the critical cases but do not generically help the matched
decoys. However, the SFT baseline's critical normal-minus-dropped effect is also
`+0.04444`. This run preserved useful communication behavior; it did **not**
show that RL improved communication over SFT. Capability was also flat on the
larger holdout, so the 30-update run is a stable systems result rather than a
capability breakthrough.

## Regression and collapse diagnostics

- Both 256-case non-arena regression suites passed for all four policies.
  Regression-v1 deltas ranged from `0` to `-0.00781`; regression-v2 deltas
  ranged from `+0.00781` to `+0.01172`. Arena leakage remained zero.
- Overall constrained candidate-to-SFT KL mean was `0.00137` and p99 was
  `0.02485`. BLUE-3 had a larger role-specific p99 of `0.21686`, below the
  fixed `0.30` collapse limit but worth tracking in longer runs.
- No always/never-speaking, repeated-target, action, excessive-KL,
  single-opponent, or return-without-message collapse flag fired. Per-role
  speaking rate was `0.54545` on the holdout trajectories.

## Verdict

**Mechanically stable and RL-usable; information-specific communication is
present but not improved over SFT; capability improvement is not established.**
Selection-tier and frozen-final evaluation remain unopened. A subsequent run
should optimize for longer before reusing this development-only selection
procedure, and any communication-learning claim must compare the RL
intervention effect directly with the SFT intervention effect.

Public, anonymously checksum-verified four-policy bundle:
<https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-long-development>
at revision `1af877668ee3cdd8dd5ccd4734ce620bbe5e2aa0`. It is explicitly marked
`not-admitted`.
