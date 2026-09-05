# V14: Grounded Multi-Turn Follow-Through

Status: CPU design and schedule audit complete. No GPU is rented. Launch remains
blocked on a fresh runtime certificate, immutable production/trainer configs,
and the smallest zero-update signal/parity preflight.

## The narrow diagnosis

V13 did not fail because agents could not speak, parse messages, or act on the
named target. At update 80:

- sender target-fact rate was `0.9583` and every protocol rate was `1.0`;
- candidate capture normal-minus-dropped was `+0.1667`, with a positive 95%
  interval `[+0.0417,+0.3333]`;
- overall gameplay RL-minus-SFT was `+0.0591`, with a positive interval;
- terminal normal-minus-dropped return was only `+0.0069`;
- specificity was `+0.0135`, and RL-specific lift was effectively zero.

V13 therefore learned a real local mechanism: a receiver's first target action
depends on the factual message. It did not reliably turn that first action into
complementary later actions and terminal team control. This matches the
rollout-level pattern of duplicate targets, generic scans/probes after a correct
capture choice, and effects that also appear on decoys.

The implementation explains part of that gap. V13 scheduled handoff horizons
of two or four remaining turns, but `run_live_rl.py` hard-coded the focused
handoff training span to turn offset `(0,)`. Later receiver actions were in the
trajectory and affected terminal reward, but they never received policy-gradient
credit. V14 fixes that exact mismatch; it does not add a message reward or
change the terminal objective.

## Initializer and claim boundary

V14 continues from the four distinct, anonymously mirrored V13 update-80 LoRA
adapters. Update 80 is used because it has the strongest verified gameplay and
legacy/hard retention, not because V13 passed a communication claim. Adapter
hashes and policy revision are frozen in `data/rl_v14/cpu_bundle.json`.

The V13 update-60 and update-80 public summaries are separately hash-bound in
`data/rl_v14/diagnosis.json`. The frozen held-out suite remains unopened.

## Training-only adaptive curriculum

V14 has a maximum of 40 updates, four logical groups per update, and four
fail-closed 10-update stages. The stage shape, reward, receiver balance,
opponent rotation, adaptation algorithm, and candidate pool are immutable. The
specific training handoff cases after stage one are chosen deterministically at
the preceding stage boundary from training rollouts only:

| Stage | Updates | Ordinary / critical / decoy per update | Remaining turns | Trained receiver ACT offsets |
|---|---:|---:|---:|---|
| `two_turn_mechanism` | 10 | 1 / 2 / 1 | 4 | 0, 1 |
| `three_turn_transfer` | 10 | 1 / 2 / 1 | 4 | 0, 1, 2 |
| `long_horizon_conversion` | 10 | 1 / 2 / 1 | 6 | 0, 1, 2, 3 |
| `specificity_consolidation` | 10 | 2 / 1 / 1 | 6 | 1, 2, 3, 4 |

The final stage intentionally stops reinforcing the already-solved first action
and trains only downstream follow-through. Across the complete maximum schedule
there are 70 factual critical groups, 40 matched decoys, and 50 ordinary
retention groups. Every receiver receives exactly ten decoys; critical counts
are 18/18/17/17. Every decoy is an update-local matched subset of a critical
case.

The V13 training-rollout pass-rate audit found 33 observed critical case/worlds
and 24 observed decoys. Critical cases classified as 18 frontier, 7 mastered,
and 8 stalled; decoys classified as 2 frontier and 22 stalled. Nine candidate
decoy case/worlds in the immutable training pool were unseen and therefore count as
frontier exploration, not successes. Ordinary size/horizon buckets were 13
frontier and one mastered. These classifications use verified training
trajectory outcomes only; they do not open development or frozen evaluation.

Stage one now uses eight frontier/unseen decoys plus one stalled anchor, and
eight frontier critical extras plus one mastered and one stalled anchor. After
each completed ten-update stage, the selector repeats the same analysis on that
stage alone. Eighty percent of the next handoff slots target mixed or unseen
frontier cases; ten percent retain mastered regression anchors; ten percent
retain stalled diagnostic anchors. If a category does not exist for a receiver,
the selector falls back deterministically without changing receiver balance.
All selections are written atomically with the source-evidence hash. Resume
must reproduce the exact artifact or fail closed.

Stalled does not mean impossible. It means no positive verified effect was
observed in the available training replicas. Such cases are mostly removed from
ordinary sampling but kept sparsely so a later capability gain is visible.
Mastered cases are likewise retained sparsely to detect forgetting. Ordinary
retention cases are never adaptively removed, because doing so would make the
curriculum itself erase legacy coverage.

The intervention and reward remain unchanged:

- critical: factual terminal return minus receiver-only target-swapped return;
- decoy: target-swap challenge return minus factual/misleading return;
- ordinary: verified terminal return with leave-one-out shared-return baseline;
- no additive message reward, learned judge, or supervised semantic target.

The existing safety supervisor already verifies exact replay, receiver-only
delivery intervention, decision coverage, terminal return, and trainer/serving
parity for multiple turn offsets. V14 adds the missing curriculum-to-controller
binding and keeps legacy plan hashes unchanged when no multi-turn offsets are
declared. V14 records whether each complete logical batch has any nonzero
advantage, but this is telemetry rather than an admission gate. Zero-advantage
batches are expected near saturation and continue through the ordinary
optimizer path. Adaptation never conditions on a single favorable resample and
never retries a batch until it has gradient; it reweights the next stage using
the complete preceding stage.

## Stage gates

Full 192-row development pulses occur at updates 0, 10, 20, 30, and 40. A
hash-bound gate is evaluated before each non-zero continuation. Every gate also
requires all protocol rates to equal `1.0`.

| Update | Terminal message effect | Specificity | RL-specific lift | Capture mechanism | Hard/legacy | Overall gameplay |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | >= 0.000 | >= -0.010 | >= -0.010 | >= 0.100 | >= -0.020 | >= 0.000 |
| 20 | >= 0.015 | >= 0.000 | >= 0.000 | >= 0.125 | >= -0.020 | >= 0.010 |
| 30 | >= 0.025 | >= 0.010 | >= 0.010 | >= 0.125 | >= -0.010 | >= 0.020 |
| 40 | >= 0.030 | >= 0.020 | >= 0.020 | >= 0.125 | >= 0.000 | >= 0.030 |

These are development control thresholds, not confirmatory claims. If any gate
fails, the pulse daemon writes immutable gate/rejection evidence and withholds
the continuation; no later optimizer update is authorized. Thresholds cannot be
relaxed after results are observed.

## Why no privileged critic yet

A state-privileged value function remains theoretically attractive for reducing
ordinary-trajectory variance without biasing the policy objective. It is not
the smallest fix for V13's demonstrated bottleneck: V13 already produced a
positive overall gameplay interval, while the missing signal is later
message-conditioned follow-through. Adding and validating a learned critic now
would conflate two changes and require a new cross-fitting, lag, replay, and
checkpoint contract. V14 keeps the terminal estimator unchanged. A privileged
baseline should be introduced only if the zero-update V14 ordinary screen or a
stage gate shows renewed ordinary-credit sparsity.

## Compute contract

- target: one 4xL40S node;
- maximum optimizer updates: 40;
- maximum provider spend: `$15` at an assumed `$1.52/hour`;
- wall-time cutoff: 9 hours;
- no rental before all CPU tests and the launch bundle are complete;
- on-node zero-update preflight before update 1: adapter hash/alias checks,
  multi-turn exact replay, parity/KL, ordinary pass@4 signed signal in every
  policy slot, HF compact mirror, W&B, watcher, and recovery supervisor;
- automatic stop on a failed stage gate, hard budget, or completed final sync;
- after final summary, public checkpoint hashes, and W&B sync are verified,
  resolve and decommission the exact provider instance immediately.

The preflight is intentionally bounded. It may reject bad seeds or a broken
runtime before optimizer work, but it must not turn into another broad GPU
search. If it fails, decommission and repair on CPU.

## CPU artifacts

- `data/rl_v14/diagnosis.json`: exact V13 evidence extraction and source hashes;
- `data/rl_v14/curriculum.json`: immutable stage shapes, adaptive selector, and candidate pool;
- `data/rl_v14/curriculum_audit.json`: counts, balance, matching, and schedule hash;
- `data/rl_v14/v13_frontier_analysis.json`: compact training-only pass-rate audit;
- `data/rl_v14/stage_gates.json`: fail-closed checkpoint thresholds;
- `data/rl_v14/cpu_bundle.json`: initializer, budget, and launch blockers.

The checked-in schedule SHA binds the stage-one bootstrap plus the unadapted
later-stage skeleton. Every later stage receives its own evidence-bound
selection SHA before rollout; the production-plan SHA binds the immutable
selector config and implementation.
Current stage-gate body SHA-256:
`27098650c9e6f604e8393a75fc01cb0a0e6c694cf80286ba7d8965de84a1f8c2`.
