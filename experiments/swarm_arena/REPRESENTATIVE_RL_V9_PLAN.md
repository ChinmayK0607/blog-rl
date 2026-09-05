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

## CPU implementation status

Implemented:

1. `message_swap` is a first-class rollout branch bound into immutable evidence,
   independent replay, delivery-contract verification, supervisor admission,
   common-random validation, and receiver-only terminal-contrast advantage.
2. The trusted transform structurally swaps only the two certified candidate
   identities in facts and candidate-target intents. It preserves the message
   schema, fact count, sender, turn, and all non-candidate content.
3. The compact pair evaluator now reports actual-minus-target-swapped return,
   receiver target-action direction, and critical-minus-decoy semantic
   specificity per pair and to W&B.
4. The checked-in 60-update curriculum contains 216 semantic groups and 24
   ordinary preservation groups. Sender and receiver policy slots each receive
   exactly 54 semantic assignments; both worlds receive 108.
5. The CPU audit verifies private-world indistinguishability, unchanged legal
   actions, positive critical oracle opportunity, zero decoy oracle value, and
   exact role/world balance.
6. All 135 Swarm Arena CPU tests pass. The focused semantic-counterfactual
   matrix accounts for 42 of them and covers structural transformation, live
   rollout generation, independent replay, fail-closed supervisor admission,
   compact evaluation, and pulse validation.

Still required on the eventual GPU host: the full Linux suite, exact
serving/trainer calibration, and a short first-update sampled-advantage audit.
The last item is irreducibly model-dependent; CPU construction checks cannot
honestly guarantee that the SFT policy samples non-zero contrasts.

## Interpretation boundary

A positive v9 result would show learned content-conditioned receiver behavior
under decentralized private observations in this simulator. It would be a
credible first MARL communication result, not evidence of general swarm
intelligence or an alignment claim.
