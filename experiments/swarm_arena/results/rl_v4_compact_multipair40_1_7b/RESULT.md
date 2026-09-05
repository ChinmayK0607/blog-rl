# Compact multi-pair communication learnability result

## Verdict

The 40-update Qwen3-1.7B run completed mechanically and learned more
target-directed tactical behavior, but it did **not** learn causal use of the
teammate message. No checkpoint is eligible for development selection, and the
frozen OOD suite remains unopened.

- Run: `rl-v4-compact-multipair40-3ca20933-l40s-a`
- Source: `3ca20933409f2e02f9dce60a3f295f10d15a0806`
- Production-plan SHA-256:
  `93ae311d15bf57a40d7d4a16298bdb8fe35e28997eb68726bcc69e251ef6d64a`
- Runtime-certificate SHA-256:
  `a3da30e9d65b5ef457058444ac6ccf3dbd448758063c26fc7a9e86c5173a8115`
- Hardware: four NVIDIA L40S 46 GB GPUs; three inference workers and one
  trainer; approximately 72 minutes from launch to the final evaluation.
- Tests: 130 Linux tests passed before launch, with two known third-party SWIG
  deprecation warnings.
- Reward: replay-verified terminal control delta only. No message, target,
  action, or capture bonus was used.

## Checkpoint curve

Each checkpoint used 48 matched training-pair games: pairs 7 and 9, both latent
worlds, critical and matched-decoy cases, and normal, dropped, and shuffled
messages with two repetitions. Receiver target accuracy is over the eight
critical normal or dropped games.

| Update | Normal - dropped return | Normal - shuffled return | Normal target | Dropped target | Critical - decoy lift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.01667 | +0.02778 | 4/8 | 4/8 | -0.02870 |
| 10 | -0.00926 | -0.00926 | 4/8 | 4/8 | -0.07222 |
| 20 | -0.00833 | 0.00000 | 4/8 | 4/8 | -0.00185 |
| 30 | +0.00093 | -0.05093 | 4/8 | 4/8 | 0.00000 |
| 40 | +0.00833 | -0.01759 | 4/8 | 4/8 | +0.00833 |

At update 40, a deterministic 20,000-resample bootstrap over the four complete
pair/world units gives:

- normal minus dropped return: `+0.00833`, 95% interval
  `[-0.05370, +0.07037]`; two units positive and two negative;
- normal minus shuffled return: `-0.01759`, 95% interval
  `[-0.08333, +0.04722]`;
- critical minus matched-decoy specificity: `+0.00833`, 95% interval
  `[0.00000, +0.02500]`, driven by one positive unit while three were zero;
- normal receiver target choice: `4/8`; dropped receiver target choice: `4/8`;
- action validity, broadcast validity, and broadcast grounding: `1.0` at every
  checkpoint.

The qualitative preregistered gate required generated messages to beat both
dropped and shuffled messages in aggregate, a positive effect on both pairs,
better receiver target choice with the message, and critical-over-decoy
specificity. Update 40 fails the shuffled, receiver-choice, and both-pair
requirements. Earlier checkpoints fail as well.

## What the optimizer learned

The focused receiver batches retained usable terminal-return variation. From
the first ten to the last ten updates:

| Receiver pair | Target action | Target capture | Mean return | Non-zero advantage |
| --- | ---: | ---: | ---: | ---: |
| Pair 7, first 10 | 50.63% | 20.63% | -0.00648 | 83.75% |
| Pair 7, last 10 | 52.50% | 26.25% | +0.02222 | 95.00% |
| Pair 9, first 10 | 31.88% | 6.25% | +0.00083 | 50.00% |
| Pair 9, last 10 | 43.13% | 20.63% | -0.00417 | 95.00% |

Pair 9's last-ten target-action rate was 51.25% in the left world but only
35.00% in the right world. Pair 7 was 55.00% left and 50.00% right. The model
therefore strengthened target/capture tactics and a left-target prior rather
than learning to switch its choice from the sender's private fact. This is a
credit-localization/conditional-learning limitation, not missing reward
density, invalid output, numerical rejection, or infrastructure failure.

## Integrity and artifacts

- Final ready-record SHA-256:
  `640f17dba7003298900ae8712acfc13d16ff4780eb5bbcd0ecbf5c4fae074eef`
- Final policy revision:
  `dfc7e695494698ab43bd13ee8c4446f793ae973eddfddee013be75353e23e79a`
- Final adapter SHA-256 values:
  - `blue-0`: `4b50b6819ca3bea2634e1f4409b686e96bb180fe6f523689efc68adfa77a85fb`
  - `blue-1`: `2124b5bb516799ab8ce22785bc47bf4d800a31bb82a24ec3ab5dce5bdec9ed0f`
  - `blue-2`: `9108b0da279cba7613910b81991f8de5078b0398d00ec73301f9d8d32c9cbeac`
  - `blue-3`: `c0dfb9b9ec44dda9eda89e45caa2cc8496a506d4ba562831b7565a8d066d04d1`
- Complete four-adapter checkpoints at updates 10, 20, 30, and 40 were
  uploaded atomically and anonymously downloaded/re-hashed before the mirror
  marked them valid.
- Public recovery artifacts:
  <https://huggingface.co/CK0607/swarm-arena-live-runs/tree/main/runs/rl-v4-compact-multipair40-3ca20933-l40s-a>
- Controller/evaluation W&B:
  <https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v4-compact-multipair40-3ca20933-l40s-a-controller-v1>
- Trainer W&B:
  <https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/ya13ogv3>

The local directory intentionally contains only 1.0 MB of compact evidence:
raw evaluation rows, summaries, bound configs, controller progress, checkpoint
hash barriers, and runtime status. Model/checkpoint payloads remain on the
public Hub and were not copied to the Mac.

## Decision

Do not promote update 40 and do not run development or frozen OOD evaluation.
The next experiment should replace the within-world action baseline with a
prospectively audited paired terminal-return contrast that explicitly centers
receiver credit on the critical normal-versus-message-intervention difference,
while retaining matched decoys as a null control. If that estimator cannot show
balanced non-zero critical-over-decoy signal for both receiver slots before
optimization, switch to the 4B model rather than extending this 1.7B run.
