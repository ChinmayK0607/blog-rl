# Joint sender-receiver RL plan

## Why the previous run could not learn the full channel

The focused update-60 run varied one agent across common-random-number
replicas, assigned that agent the terminal-return advantage, and trained only
its `ACT` spans. On every handoff scenario the focused agent was the receiver.
This was a clean way to learn message-conditioned actions, but it provided no
direct optimization path from a sender's broadcast choice to its own policy.
Because each LoRA is shared between broadcast and action prompting, action-only
updates could also change broadcast behavior indirectly. The observed result—
better game tactics and weaker message dependence—is therefore consistent with
the optimization contract, not evidence that terminal reward cannot teach
communication.

## Phase-specific focused credit

The next run retains the verified terminal control delta as its only reward.
Each rollout group samples four replicas. One designated agent and one decision
phase vary independently; every other agent/phase sampling stream is coupled.
The designated policy receives the leave-one-out terminal-return advantage.
Other policy envelopes are retained for atomic four-policy routing but receive
zero advantage.

- **Sender group:** vary the certified sender's turn-zero `BROADCAST`; train
  only that broadcast span.
- **Receiver group:** couple the sender message, vary the certified receiver's
  turn-zero `ACT`; train only that action span.
- **Ordinary group:** vary one rotating agent's `ACT` choices over the ordinary
  episode; train only that agent's action spans.

This is phase-local causal credit, not a communication bonus. A sender earns a
positive update only when its sampled broadcast changes the eventual verified
game return. A message that is empty, irrelevant, or misleading receives the
same zero or negative game consequence it naturally causes.

## Curriculum v3

`data/rl_v4/staged_curriculum_v3_joint_80.json` declares 80 updates and four
groups per update:

| Stage | Updates | Ordinary | Critical | Matched decoy |
| --- | ---: | ---: | ---: | ---: |
| joint channel warmup | 10 | 20 | 10 | 10 |
| joint channel acquisition | 20 | 20 | 30 | 30 |
| joint communication reliance | 30 | 0 | 60 | 60 |
| joint consolidation | 20 | 20 | 30 | 30 |
| **Total** | **80** | **60** | **130** | **130** |

Across the 130 critical groups, 65 focus the sender and 65 the receiver. Each
critical assignment retains its structural decoy with the same focus role.
Ordinary maps remain present before and after the communication-heavy stage to
measure tactical retention. The next run starts from the pinned SFT initializer,
not update 50: update 50 had useful tactical learning but already showed weaker
message dependence and is a poor communication initializer.

## Progress measurement

The frozen 192-game pulse remains byte-identical in cases, sides, baseline,
opponent, and normal/dropped interventions. It still reports:

- overall, legacy, hard, and critical-handoff RL-minus-SFT return;
- critical normal-minus-dropped return;
- RL-specific communication lift;
- critical-minus-decoy specificity;
- protocol validity.

The runner now adds mechanism diagnostics from the same games:

- whether the certified sender broadcast the active-target fact;
- critical capture-rate difference under normal versus dropped messages;
- RL-minus-SFT change in that capture dependence.

These diagnostics explain *where* a run fails without changing reward or
checkpoint selection. W&B also separates sender-broadcast and receiver-action
advantage density. The frozen selection and final OOD suites remain untouched.

## Execution contract

Use `configs/rl_v4_1_7b_joint_80.toml`, build a fresh immutable production plan
from curriculum v3, and bind it to a fresh runtime certificate on the rented
Linux host. Run the full Linux suite before optimizer launch, including the
sender-broadcast focused-return test that requires the GPU dependency stack.
Then launch one 80-update run with evaluation barriers every ten updates.

This run is successful as a development result only if capability does not
collapse and both return-level message dependence and the capture-level
mechanism improve from the exact update-zero anchor. Speaking frequency by
itself is never success.
