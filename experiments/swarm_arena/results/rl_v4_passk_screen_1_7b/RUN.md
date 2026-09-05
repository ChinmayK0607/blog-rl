# 1.7B training-only pass@k screen

## Scope

This is an exploratory curriculum-selection diagnostic on 12 role-balanced
bundles from `handoff_train.json`. It does not use development or frozen OOD
cases and cannot promote a checkpoint.

- Focal and opponent: pinned Qwen3-1.7B SFT initializer, model versus model
- Conditions: generated, focal-team messages dropped, and reference fact
- Sampling: temperature 0.7; critical generated K=8; controls and decoys K=4
- Games: 672 complete 4v4 episodes
- Requests: 26,880 independent agent requests
- Completion tokens: 1,008,519
- Corrected rollout wall time: 1,952.76 seconds on one L40S
- Protocol validity: 100%

The reference condition preserves the generated message and inserts the
certified grounded target fact only when it is missing. It does not supervise
an action or any later decision.

## Main result

This slice is useful for receiver/tactical RL, but not for sender learning.

| Metric | Result |
|---|---:|
| Critical target-capture pass@1 | 0.1510 |
| Critical target-capture pass@4 | 0.4161 |
| Critical target-capture pass@8 | 0.5833 |
| Critical return contrast@4 | 0.7655 |
| Generated sender target-fact rate | 1.0000 |

The high return contrast means terminal shared return supplies usable policy
gradient on most sampled worlds. The 15.1% pass@1 and 58.3% pass@8 capture rates
place many worlds in a sensible small-model learning range rather than at a
fully solved ceiling.

Communication is not yet a reliable terminal-return signal. Using each bundle
as the independent bootstrap unit:

| Paired endpoint | Mean | 95% interval |
|---|---:|---:|
| Critical generated minus dropped terminal return | +0.0070 | [-0.0119, +0.0243] |
| Critical-minus-decoy terminal-return specificity | -0.0019 | [-0.0235, +0.0165] |
| Critical generated minus dropped turn-zero capture | +0.0312 | [-0.0208, +0.0938] |
| Critical-minus-decoy turn-zero capture specificity | +0.0729 | [0.0000, +0.1667] |
| Critical-minus-decoy receiver-target-action specificity | +0.0104 | [-0.0417, +0.0625] |

The immediate capture result is suggestive but heterogeneous: only four of 12
bundle effects are positive. By terminal time the effect has disappeared.
Because target-fact emission is already 100%, the reference insertion is an
identity intervention for this slice. Small reference-versus-generated
differences therefore expose the known non-bit-reproducibility of independent
vLLM requests and provide a useful empirical noise floor; they are not evidence
that adding an already-present fact helps.

## Exploratory training bands

The band thresholds were selected after inspecting this training-only screen;
they are data-selection heuristics, not confirmatory gates.

| Band | Definition | Worlds |
|---|---|---:|
| Primary receiver | return contrast@4 >= 0.75 and capture pass@1 in [0.125, 0.5] | 12 |
| Hard reserve | return contrast@4 >= 0.75 and capture pass@1 = 0 | 3 |
| Easy stabilizers | return contrast@4 >= 0.5 and capture pass@1 > 0.5 | 1 |
| Low signal | all remaining screened critical worlds | 8 |

For every retained critical world, retain its matched decoy. Do not train a
critical case without the corresponding null control.

The next curriculum should emphasize the 12-world receiver band, use the one
easy world only as a small stabilizer, and hold the three hard worlds out of the
first stage. It should not allocate sender-BROADCAST updates from this slice:
the sender behavior is saturated. If sender learning remains desired, first
screen later training pairs for actual target-fact omissions.

Most importantly, put early handoffs close to the terminal boundary (or use a
short-horizon stage) so a correct message-conditioned capture survives into
the verified terminal reward. Then extend to full 4/5-turn play. This changes
the curriculum, not the game or reward, and directly addresses the observed
washout.

## Artifacts

- `manifest.json`: `a4b2c8965ad39e4b37172acb6fc19181b17ad8296bd9fca68f02703de83baae0`
- `rows.jsonl`: `c7e4268f54e8153d12e4ca080c1a7d079ea62f5716ad0d13875633b5223c3d36`
- `summary.json`: `780a3f5e5ffacf2ba22d8e395b0ba1116ec350fcbd680d8956ded3dd5d5c2b1b`
- `analysis.json`: `874dec87823fc4ac899c671e767a8563935bb8f9925a87cb9022b32f37240129`

The initial 96-game engineering partial is excluded. It replaced rather than
merged the reference fact and is preserved only on the rented host as rejected
evidence. No model weights changed during either run.
