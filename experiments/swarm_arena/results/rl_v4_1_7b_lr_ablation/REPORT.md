# RL v4 learning-rate and curriculum ablation

Status: **development-only; capability-positive; communication-negative; not admitted**.

## Question

The 30-update v4 run was mechanically stable but essentially flat against its
SFT initializer. This follow-up changed one factor at a time to test whether
the issue was insufficient optimization pressure or insufficient exposure to
communication-critical cases. The frozen selection/final suites were not
opened.

## Variants

| Variant | Learning rate | Exact update mixture | Valid updates | Result |
| --- | ---: | --- | ---: | --- |
| A | `1e-5` | 2 ordinary / 1 critical / 1 matched decoy | 12 | Stable; selected step 8 improved ordinary return |
| B | `1e-5` | 2 ordinary / 3 critical / 3 matched decoys | 4 | Stopped at update 5 by the unchanged parity ceiling |

Both variants started independently from the pinned SFT step-320 initializer,
trained four distinct BLUE LoRA policies, used only verified terminal team
return, and retained the model-controlled base/SFT/historical opponent pool.
No shaping reward or supervised message target was added.

Variant A completed 12/12 updates (48 groups, 192 replicas). Its mean return
was `+0.00717239`, mean absolute advantage was `0.107124`, and mean nonzero
advantage rate was `0.822917`. Every update had learning signal and all four
role adapters changed.

Variant B completed four valid updates (32 groups, 128 replicas), with mean
return `+0.0149210`, mean absolute advantage `0.0861203`, and mean nonzero
advantage rate `0.7890625`. All eight update-5 rollout groups were admitted,
but the optimizer was not run: one policy exceeded the fixed per-token
rollout/trainer mismatch-KL ceiling (`1.0974264 > 1.0`). The other three
policies passed that update, with mean mismatch KL between `0.000147` and
`0.000584`. The ceiling was not relaxed, so only steps 1--4 are valid evidence.

## Matched checkpoint pulse

A deterministic 66-game development pulse compared Variant A steps 8 and 12
with Variant B step 4 on identical cases (offset 6). It selected Variant A step
8 by ordinary capability:

| Candidate | Ordinary candidate - SFT | Critical message effects |
| --- | ---: | ---: |
| Variant A step 8 | `+0.154762` | all `0` |
| Variant A step 12 | `+0.0833333` | all `0` |
| Variant B step 4 | `+0.0357143` | all `0` |

All action, broadcast-schema, and broadcast-grounding rates were `1.0`. The
critical pulse was uninformative and was not used to claim communication.

## Non-overlapping development holdout

The selected Variant A step 8 then ran 198 games over three disjoint shards
(offsets 9--11). Evaluation IDs were unique (`198/198`).

- Ordinary candidate-minus-SFT: **`+0.0707407`** over 18 paired cells.
- Candidate-normal return by opponent when averaging all three suites
  (ordinary, critical, and decoy): base `+0.0933333`, SFT `+0.0200000`,
  historical league `+0.00555556`.
- Ordinary-only candidate return by opponent: base `+0.280000`, SFT
  `+0.0600000`, historical league `+0.0166667`.
- Candidate critical normal-minus-dropped/shuffled/delayed/zero-budget:
  **exactly `0` for every intervention**.
- SFT critical normal-minus-dropped: `0`.
- Matched-decoy normal-minus-dropped: `0`.
- Action, message-schema, and message-grounding validity: `1.0`.

This is a real game-capability improvement on this development sample, but it
is not evidence that RL improved causal message use. In these holdout cells,
neither the SFT nor RL policy used messages in a way that changed returns.

## Regression, drift, and collapse checks

Both 256-case frozen non-arena regression suites passed for every role. V1
overall deltas versus SFT ranged from `-0.00390625` to `0`; v2 ranged from
`-0.00390625` to `+0.0078125`. Arena-trigger leakage was `0` in all eight
comparisons.

Reference-state constrained candidate-to-SFT KL was small: mean `0.000782232`,
p99 `0.0161926`, and max `0.0924678`. No role exceeded the fixed mean or p99
KL limits.

The collapse audit found no action collapse, speaking extreme, repeated-target
collapse, excessive KL, or single-opponent specialization. It intentionally
failed the overall gate only because capability improved while causal message
gain remained zero. That is the correct stop/inspect verdict for this run.

## Interpretation

Increasing the learning rate was more useful than merely running the original
setting longer: it produced an ordinary-return gain by step 8 without protocol
failure. Increasing communication-case density at the same learning rate was
too aggressive for the fixed bounded-off-policy safety envelope and did not
produce a usable later checkpoint.

The next optimization run should therefore keep the stable `1e-5` ordinary
backbone but change how communication-critical experience enters the
curriculum, rather than blindly adding more of the same cases. In particular,
the training batch should contain cases whose state transition is mechanically
dependent on a private fact reaching a different role, while the reward remains
terminal team return. Progress evaluation must continue to report capability
and causal message-intervention deltas separately.

## Reproducibility and storage

The directory contains only compact configs, progress summaries, evaluation
summaries, hashes, diagnostics, and launchers. Raw trajectories and LoRA
checkpoints were not copied to the Mac. Variant A step 8 is development-only
and is not promoted as an admitted model.
