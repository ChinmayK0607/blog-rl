# Swarm Arena RL task v3

RL v3 keeps the v2 simulator mechanics and freezes a stricter learning
objective. It does not modify or replace the existing v2 frozen OOD cases.

## Objective

The only environment reward is the normalized change in terminal control
margin:

`(BLUE-owned node weight - RED-owned node weight) / total node weight`

where a node's weight is `value + int(critical)`. The seed-specific initial
margin is subtracted as a variance-reducing constant. The resulting reward is
zero-sum. There are no intermediate rewards, scan/capture/defence bonuses,
collision penalties, message costs, invalid-output penalties, or learned
judges.

Invalid broadcasts and actions are infrastructure contract violations. They
fail the rollout rather than becoming a reward feature. Communication is scarce
through a hard episodic budget, not an additive penalty.

## Policies and credit

The primary experiment has four separately optimized LoRA policies over one
frozen 1.7B backbone. Each policy owns one stable teammate identity and receives
only that agent's private context and completion tokens. The opponent team is
model-controlled but frozen during an update epoch.

For every actual game, four common-random-number replacement branches rerun the
game with exactly one trainable agent replaced by the frozen SFT policy. Agent
`i` receives:

`A_i = actual terminal return - replace-agent-i terminal return`

Only the actual game's trainable-agent tokens receive gradients. Replacement
and opponent tokens never do. This is a counterfactual baseline for the single
terminal objective, not an additional reward. It is an exact difference for
that replacement branch under the frozen opponent and shared random-key
schedule; it is not a Shapley value or a claim that interactions have a unique
causal owner.

## Curriculum

Stage-one examples are exact one-turn causal probes embedded at turn one:

- one sender privately knows an exposed high-value neutral node;
- one receiver is adjacent but does not know the node state;
- the sender cannot act on the target;
- the receiver cannot legally capture it without the fact;
- the broadcast makes capture legal in the same turn;
- exact joint-action enumeration certifies a positive terminal-control
  advantage against balanced, aggressive, and defensive frozen opponents.

Every critical example has a matched decoy with identical public state and
sender observation, but the receiver already knows the fact. The decoy's exact
communication advantage is zero. All twelve ordered sender/receiver identity
pairs occur equally often in every split.

The checked-in schedule marks only stage one as ready. Stage two mixes the
certified probes with two/three-turn ordinary maps; stage three increases the
ordinary-map share and horizon to four/six turns. Promotion uses held-out
evaluation; training reward alone never advances a stage.

## Immutable evaluation boundary

The original 72-case v2 OOD suite remains unchanged. RL v3 adds a disjoint
frozen OOD set of certified critical/decoy pairs. Training and development
manifests use graph sizes 12/13; the added frozen set uses sizes 14/16.
