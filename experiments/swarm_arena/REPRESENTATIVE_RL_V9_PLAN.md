# Representative RL v9: semantic receiver use

## Decision from v8

V8 established that four independent Qwen3-4B LoRA policies can complete a
60-update full-interface run and become more sensitive to teammate messages on
development cases. It did not establish semantic communication on held-out
cases. The selected update-36 policy had a small critical normal-minus-dropped
effect, but the matched decoy effect was equally large. Rollouts showed correct
facts being broadcast, heavy communication-budget use, and occasional useful
handoffs, alongside redundant targets and map-dependent receiver behavior.

The next run must therefore optimize and measure **whether a receiver uses the
content of a private fact**, not merely whether having any message changes the
trajectory.

## Primary counterfactual

Each critical training unit contains two legal candidate targets and two latent
worlds. The actual branch delivers the sender's grounded fact identifying the
exposed target. Its matched counterfactual delivers a well-formed fact with the
two candidate target identifiers swapped.

Everything else is coupled:

- same initial state, opponent, policies, prompt order, and random-key schedule;
- same sender, receiver, turn, message schema, and message count;
- both referenced targets are legal candidates in both worlds;
- only the private target fact changes;
- reward is the difference in verified terminal team return between the actual
  and target-swapped branches, centered across common-random replicas.

There is no message, target, capture, protocol, or action bonus. This contrast
removes the silence and message-presence shortcut that v8 could exploit through
generic de-duplication.

## Trainable policies and spans

The four agent policies remain separate rank-32 LoRA adapters with independent
optimizers and checkpoints. They start from the same immutable SFT adapter and
are never merged during training.

The first implementation trains only the designated receiver's first `ACT`
span. Sender broadcasts remain frozen because v8 already achieved perfect
sender target-fact accuracy on the small held-out screen. Later receiver turns
or sender spans are admitted only after an earlier stage transfers on a
development split:

1. receiver turn-zero `ACT`, two remaining turns;
2. receiver turn-zero and turn-one `ACT`, three remaining turns;
3. receiver `ACT` spans through turn two, four remaining turns.

Stage transitions are prospective and mechanical. A stage advances only if the
semantic critical contrast improves without an equal decoy contrast and the
ordinary preservation floor holds. Failing a stage stops the run; it does not
relax the gate.

## Data and split discipline

- Balance all 12 directed sender-to-receiver roles, four receiver policy slots,
  both latent worlds, target position, map size, and opponent family.
- Use disjoint pair IDs for training, development selection, and frozen OOD.
- Include matched decoys where the receiver already knows the target. The
  target-swapped intervention should have no information value there.
- Include ordinary games in alternating updates for capability preservation;
  do not add their return to the semantic advantage in one scalar reward.
- Increase map and horizon diversity within each stage instead of training a
  single deterministic opponent or target convention.

## Development evaluation and checkpoint selection

Run the compact development pulse at update 0 and every 10 updates. Report raw
counts and paired intervals for:

1. critical actual-minus-target-swapped terminal return;
2. receiver active-target action rate under actual and target-swapped facts;
3. critical-minus-decoy specificity;
4. normal-minus-dropped and normal-minus-shuffled return as secondary checks;
5. ordinary terminal return versus SFT;
6. duplicate-target-turn rate, communication spend, action/broadcast validity,
   grounding, KL/parity, and opponent/pair decomposition.

Select the earliest checkpoint satisfying all of:

- positive critical target-swapped return effect on a majority of development
  pairs and both latent worlds;
- receiver active-target accuracy improves over SFT and changes in the expected
  direction when the fact is swapped;
- critical-minus-decoy specificity is positive;
- no material ordinary-game regression versus the preregistered floor;
- no action, communication-budget, repeated-target, KL, or opponent collapse.

Only the selected checkpoint is opened on the unchanged frozen OOD suite. A
rising training return or normal-minus-dropped lift alone is not success.

## Required CPU implementation before GPU use

1. Add a target-swapped branch type to rollout evidence, replay verification,
   supervisor admission, and immutable run-lock hashing.
2. Generate the swapped fact structurally from the certified handoff bundle;
   never edit free-form text or expose the active world in the receiver prompt.
3. Prove paired branches differ only in the delivered sender fact and downstream
   model decisions/state transitions.
4. Extend the learnability audit to require balanced non-zero semantic effects
   across receiver slots, worlds, opponents, and matched decoys.
5. Add a compact pulse evaluator and predeclare thresholds before launch.
6. Run the Linux test suite and exact serving/trainer calibration once on the
   eventual GPU host, then launch without additional exploratory detours.

## Interpretation boundary

A positive v9 result would show learned content-conditioned receiver behavior
under decentralized private observations in this simulator. It would be a
credible first MARL communication result, not evidence of general swarm
intelligence or an alignment claim.
