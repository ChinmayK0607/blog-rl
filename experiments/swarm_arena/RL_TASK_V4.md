# Swarm Arena RL task v4

RL v4 preserves the existing 4v4 simulator and terminal control objective. Its
new scenario family changes the *distribution of private information*, not the
game's reward or action set.

## Two-world information handoff

Each handoff bundle contains two equally likely latent worlds. The same receiver
is adjacent to two high-value neutral nodes and can legally attempt to capture
either one in both worlds. Exactly one target is exposed:

- in `left_exposed`, the left candidate is exposed;
- in `right_exposed`, the right candidate is exposed.

The receiver has the same stale observation and the same complete legal-action
set in both worlds. A dynamically assigned remote sender has a fresh grounded
observation identifying the exposed target but cannot act on either candidate.
Delivering that fact changes no legal action; it only makes the worlds
distinguishable to the receiver.

The certificate computes an exact decentralized information value. Without the
message, the receiver must use one action for its single information set across
both worlds. With the message, it may choose a different action for the two
distinct information sets. Other teammates wait, and the opponent follows each
of the frozen balanced, aggressive, and defensive policies. The critical
scenario is retained only when the informed optimal action captures the active
target and the expected terminal-control advantage is positive against all
three opponents.

Every critical bundle has a structural decoy. In the decoy, only the receiver's
private memory changes: it already has the fresh target observations, so the
two worlds are distinguishable without a message and the certified message
advantage is exactly zero.

## Unchanged mechanics and reward

- `SCAN`, `PROBE`, `CAPTURE`, `FORTIFY`, `RECOVER`, `TRANSFER`, and `WAIT` are
  unchanged.
- Broadcasts remain grounded, private, budgeted facts and intents.
- Actions remain simultaneous and model controlled for all eight agents.
- Reward remains the zero-sum normalized terminal control-margin delta.
- There is no message, capture, sender, silence, curriculum, or learned-judge
  bonus.

Ordinary procedural maps remain the majority of the curriculum. Handoff
critical and matched-decoy states together occupy 30–50% depending on stage;
roles rotate evenly across every ordered sender/receiver pair.
