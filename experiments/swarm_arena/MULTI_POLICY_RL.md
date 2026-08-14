# Four-policy RL contract

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
message-credit bootstrap, only actual BROADCAST spans are trainable; action,
opponent, and counterfactual spans are absent from optimizer batches.

`prime_rl_bridge.py` now freezes and validates the rollout-side envelope: the
atomic branch set, agent/policy ownership, token spans, immutable revisions,
sampling keys, dynamic-constraint hashes, per-policy advantages, and
rollout/trainer log-prob comparison. Prime-RL already provides multi-run LoRA:
its trainer keeps a distinct adapter slice, optimizer, scheduler, checkpoint and
`run_idx` for every `run_*` directory. The Swarm supervisor must route each
approved agent envelope into its policy's fixed run directory. This avoids a
new optimizer implementation while still producing four independently updated
policies. It must never send all four samples through one run.

One actual game and its four sender-message-drop branches form an atomic credit
group. All five branches use the same policies, revisions, map, opponent
snapshot, and per-decision random-key schedule. Drop branch `i` changes only
delivery of sender `i`'s first-turn broadcast; downstream decisions respond normally to
the changed inbox. A partial or failed group is discarded. Policy `i` receives
`G_actual - G_drop_message_i` on that actual broadcast span only; the learner
averages that policy's loss over games, not over all eight agents.

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
