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
independent training samples without merging across policy IDs. Opponent and
replacement-policy spans are masked from loss.

`prime_rl_bridge.py` now freezes and validates the rollout-side envelope: the
atomic branch set, agent/policy ownership, token spans, immutable revisions,
sampling keys, dynamic-constraint hashes, per-policy advantages, and
rollout/trainer log-prob comparison. Prime-RL's current interleaver can separate
multiple agent contexts, but its trainer still has one optimizer route. The GPU
systems step must connect these four envelopes to four LoRA optimizer routes;
the bridge deliberately fails closed instead of silently training a shared
policy.

One actual game and its four replacement branches form an atomic credit group.
All five branches share the map, opponent snapshot, and per-decision random-key
schedule. A partial or failed group is discarded. Each trainable policy receives
only `G_actual - G_replace_i`; the learner averages that policy's loss over
games, not over all eight agents.

The first systems gate requires exact agreement between rollout and trainer
token IDs, masks, policy routing, and constrained-policy log probabilities.
No optimizer step is allowed when any of these checks fails.

During training, `collapse_audit.py` reports—but never rewards—per-policy
speaking extremes, repeated message targets, action concentration, KL mean/p99,
opponent-specific return, and return gains that disappear under message
interventions. A raised flag pauses promotion for inspection; it does not add a
shaping penalty to the objective.

`safety_supervisor.py` is the only admission path into training. It binds the
source, manifests, base, adapters, opponent and allowed dynamic constraints in
an immutable run lock; independently reconstructs every agent's private
observation and inbox; replays and recomputes reward for the actual and all four
replacement branches; and writes tamper-evident approval/rejection records.
Rollout workers are untrusted producers and cannot enqueue gradients directly.
