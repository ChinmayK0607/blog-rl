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

For every actual game, four common-random-number message-drop branches rerun the
game with the same eight policies, revisions, sampling keys, and initial state.
Branch `i` changes only the environment delivery of sender `i`'s first-turn
broadcast to the empty message; every other message is delivered normally.
Downstream actions are regenerated from their changed private inboxes. Sender
`i` receives:

`A_i = actual terminal return - drop-sender-i-message terminal return`

Only sender `i`'s corresponding first-turn BROADCAST tokens receive that advantage. Action,
opponent, and counterfactual-branch tokens never receive gradients in the
bootstrap stage. This is a causal communication-channel intervention for the
single terminal objective, not an additional reward. It is not a Shapley value
or a claim that interactions have a unique causal owner.

The earlier whole-policy replacement estimator remains reproducible for audit
history but is not the default training path: paired diagnostics found diffuse
off-role effects. A run lock names the estimator, and evidence from one mode
cannot be admitted under the other.

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
