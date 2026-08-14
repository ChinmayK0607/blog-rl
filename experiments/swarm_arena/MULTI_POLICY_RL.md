# Four-policy RL contract

Status: the four-policy routing and safety contract is mechanically validated.
The original per-sender message-drop credit estimator is scientifically
rejected and must not reach an optimizer. The shared-terminal-return candidate
below is planned but not yet admitted.

One game binds eight agent identities to eight policy IDs. Exactly four policies
on one team are trainable during an update epoch; all opponent policies are
frozen. The four trainable policy IDs must be distinct.

Every model decision records:

- game, team, agent, and policy IDs;
- turn and BROADCAST/ACT phase;
- environment trajectory index;
- independent prompt and completion token spans;
- policy version, adapter revision, sampling key, and dynamic-constraint hash.

The environment emits interleaved decisions, but Prime-RL must reconstruct four
independent training samples without merging across policy IDs. During the
first bounded pilot, the trainable phase set must be frozen explicitly;
opponent and counterfactual spans are always absent from optimizer batches.

`prime_rl_bridge.py` now freezes and validates the rollout-side envelope: the
atomic branch set, agent/policy ownership, token spans, immutable revisions,
sampling keys, dynamic-constraint hashes, per-policy advantages, and
rollout/trainer log-prob comparison. Prime-RL already provides multi-run LoRA:
its trainer keeps a distinct adapter slice, optimizer, scheduler, checkpoint and
`run_idx` for every `run_*` directory. The Swarm supervisor must route each
approved agent envelope into its policy's fixed run directory. This avoids a
new optimizer implementation while still producing four independently updated
policies. It must never send all four samples through one run.

## Rejected estimator

The original bootstrap used one actual game plus four first-message deletion
branches. Policy `i` would have received
`G_actual - G_drop_message_i` on its own broadcast tokens. This estimator was
never optimized. After a prompt repair raised certified target-fact coverage
to 11/12 pairs and made receiver actions message-sensitive in 12/12 critical
versus 3/12 decoy cases, it still produced only 7 positive / 1 negative / 4
zero intended effects, 1.0767x intended/off-role localization, and off-role
effects in 8/12 pairs. The frozen gates reject it: a two-turn terminal outcome
is too coarse for clean sender-local credit.

The failed estimator remains useful as an evaluation intervention. It must not
be revived by lowering thresholds, adding a communication bonus, or treating
receiver target matching as reward.

## Candidate shared-return update

The least hackable next candidate keeps the only reward equal to the verified
terminal normalized control-margin change. For one fixed initial state,
opponent snapshot, side assignment, and policy revision, sample `K` independent
joint trajectories. For joint trajectory `k`:

`A_k = G_k - mean(G_j for j != k)`

The leave-one-out baseline is scenario- and opponent-matched but independent of
trajectory `k`'s sampled actions. Route `A_k` to each of the four trainable
policies' own token spans from that trajectory. The scalar is shared; the
policies, private contexts, sampled tokens, gradients, optimizer states, and
checkpoints are not. This remains multi-agent RL with four independently
updated policies, rather than one policy pretending to be four agents.

This design adds no shaped reward and exposes no oracle action or supervised
message target. It accepts higher variance in exchange for preserving the real
team objective. Communication-critical curriculum states make useful messaging
necessary without directly paying agents to speak. Generated-message versus
dropped, shuffled, and delayed evaluations determine whether return gains came
from communication rather than action-only improvement.

Before implementing the pilot, freeze `K`, trainable phases, centering and
normalization, opponent sampling, KL ceiling, and stop conditions. A
broadcast-only pilot is easier to interpret; a later all-phase pilot is more
capable but requires stronger intervention evidence. No choice may be made
after looking at final/OOD outcomes.

Exact four-message Shapley attribution is a possible research audit (16
delivery coalitions per state), not the current training plan. It partitions
message interactions without inventing a bonus, but costs 3.2x more branches
than singleton deletion and can still be high variance. Receiver-action target
matching is capability evidence only and is too directly gameable to use as
reward.

The first systems gate requires exact agreement between rollout and trainer
token IDs, masks and policy routing. Constrained-policy distributions must pass
the certified probability, tail and mismatch-KL envelope on the actual serving
and FSDP trainer stacks. No optimizer step is allowed when any gate fails. The
certificate also performs a real update and proves that only the selected
policy slot changes.

During training, `collapse_audit.py` reports—but never rewards—per-policy
speaking extremes, repeated message targets, action concentration, KL mean/p99,
opponent-specific return, and return gains that disappear under message
interventions. A raised flag pauses promotion for inspection; it does not add a
shaping penalty to the objective.

`safety_supervisor.py` is the only admission path into training. It binds the
source, manifests, base, adapters, opponent and allowed dynamic constraints in
an immutable run lock; independently reconstructs every agent's private
observation and inbox; proves that policies are unchanged across the actual and
four delivery-intervention branches; verifies that only the named sender is
dropped; replays and recomputes reward; and writes tamper-evident
approval/rejection records.
Rollout workers are untrusted producers and cannot enqueue gradients directly.
