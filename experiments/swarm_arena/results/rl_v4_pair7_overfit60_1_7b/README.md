# Pair-7 communication-overfit diagnosis

Run `rl-v4-pair7-overfit60-2243447c-5090-20260819` completed all 60
updates. This report diagnoses the transient positive update-50 pulse without
opening the frozen OOD evaluation.

## Verdict

The run learned a generic tactical behavior—capture more often—but did not
learn a useful, information-specific response to the sender's broadcast.
Update 50 was not a superior communication checkpoint. Its original 24-game
pulse contained only four critical normal samples; one right-world sample
flipped to the target action, while the same behavior also appeared in the
decoy and shuffled-message controls.

## Larger matched replay

The SFT initializer and RL checkpoints 50 and 60 were each replayed for 192
games: 16 repetitions of both latent worlds, critical and matched-decoy cases,
and normal, dropped, and sender-shuffled messages. Sampling keys, opponent,
temperature, and environment state were matched.

| Checkpoint | Normal return | Normal − dropped return | Normal target action | Dropped target action | Critical − decoy specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT initializer | -0.0093 | -0.0023 | 56.25% | 50.00% | +0.0023 |
| RL update 50 | +0.0394 | -0.0023 | 59.38% | 50.00% | 0.0000 |
| RL update 60 | +0.0370 | 0.0000 | 59.38% | 46.88% | -0.0023 |

The paired bootstrap 95% intervals for all critical return interventions and
critical-minus-decoy specificity include zero. The increase in normal return
from the initializer is therefore a capability gain, not a causal
communication gain.

## Training-decision audit

All 960 receiver training decisions were decoded from the retained Prime-RL
batches and aligned to their signed controller records.

| Window | Critical target action | Critical target capture | Decoy target action | Decoy target capture |
| --- | ---: | ---: | ---: | ---: |
| Updates 0–9 | 48.75% | 15.00% | 53.75% | 17.50% |
| Updates 50–59 | 52.50% | 33.75% | 52.50% | 32.50% |

The receiver learned to replace probes with captures, but critical and decoy
behavior moved together. Across all critical samples it chose `V13` in 271 of
480 decisions and `V19` in 200, yielding 57.50% accuracy in the left world but
42.92% in the right world. This is a persistent target prior, not reliable
switching from the sender's private fact.

The terminal reward itself was directionally useful: correct-target actions
averaged `+0.0556` return and other actions averaged `-0.0369` in critical
training cases. The failure is not reward inversion. The balanced two-world
setup makes the marginal gradient toward either action ID cancel unless the
small model learns the contextual interaction between the long observation
and the sender fact. With only four samples per world/update and an equally
large decoy stream, generic capture learning was the easier signal.

## Consequence for the next run

Do not extend the same 60-update curriculum or select update 50. Before another
GPU run, use more independent communication-critical pairs per update, shorten
the receiver decision context, retain both latent worlds, and compute a paired
receiver advantage across matched worlds/conditions from the unchanged
terminal return. Keep matched decoys for evaluation and a smaller training
fraction rather than giving them half of every optimizer batch. First verify
on CPU that the resulting batches contain substantially more
critical-context-specific advantage than decoy advantage.

`diagnostics/diagnosis.json` contains the decoded training summary and paired
bootstrap intervals. The three replay directories contain their bound
manifests, all compact rows, and summaries.
