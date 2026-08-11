# Swarm Arena episode v2: RL-native task

`arena-episode-v2` preserves the published v1 mechanics but changes the learning
problem. V1 remains frozen and reproducible.

## Why v1 is not the final RL task

V1 is a one-turn protocol/mechanics gate with dense reward and exact oracle
demonstrations. SFT can learn both the canonical broadcast and action mapping.
That is desirable for a warm start but makes near-saturated SFT imitation
unsurprising and does not establish causal message use.

## V2 learning problem

- 4-8 simultaneous turns rather than one;
- zero intermediate task reward and a terminal zero-sum team return;
- private partial observations and only locally visible event feedback;
- a two-phase broadcast/action turn;
- grounded facts update teammate knowledge before action selection, so a message
  can change both the receiver's legal options and its choice;
- at most two facts per message;
- a private episodic communication budget for every agent;
- a terminal communication cost and penalties for invalid messages/actions;
- periodic resource regeneration so early actions do not exhaust the episode;
- training on graph sizes 12-13 and horizons 4-6;
- frozen OOD evaluation on graph sizes 14/16 and horizons 6/8;
- opponent-policy switches halfway through evaluation episodes.

There is deliberately no target broadcast and no exact long-horizon action
label. The only optimization signal for message content, timing, specialization,
and adaptation is delayed team return. This is the part assigned to MARL.

SFT is limited to a protocol warm start: valid JSON, grounded facts, and legal
action selection. It is not trained on v2 trajectories, preferred messages, or
long-horizon oracle actions. Reusing optimized v2 trajectories as SFT labels
would collapse the intended RL question back into imitation.

The available v1 SFT adapters are not used as the v2 initialization: all eight
evaluated checkpoints fail the frozen general-instruction regression gate. V2
starts from the untouched instruct model. Strict parsers, legal-action masks,
and invalid-action costs provide protocol scaffolding without teaching a
supervised coordination policy.

## Required controls

Every selected policy is evaluated on identical episodes under:

1. generated messages;
2. dropped messages;
3. within-team sender-shuffled messages;
4. messages delayed by one turn;
5. a zero communication budget;
6. action-option permutations.

The primary endpoint is terminal team return against the frozen opponent
schedule. The mechanism endpoint is the paired generated-minus-dropped return.
Generated-minus-shuffled and generated-minus-delayed must also be positive to
claim semantic, timely message use.

## Model-free task audit

On the 72 frozen OOD episodes, a centralized collision-avoiding heuristic beats
independent local heuristics by `+0.540` terminal return with a normal 95%
interval of `[+0.059, +1.020]`. Always waiting scores `-15.505`, compared with
`-1.399` for independent agents. The task is therefore neither solved by a
trivial passive policy nor saturated by independent one-step behavior. The
centralized advantage is modest, so publication claims still depend on the
stronger learned-policy intervention results rather than this audit alone.

## Promotion gates

- beat the SFT warm start on paired terminal return with a 95% interval excluding
  zero;
- generated messages beat dropped, shuffled, and delayed controls with paired
  intervals excluding zero;
- at least 95% strict action validity;
- at least 99% grounded broadcast validity;
- at least 95% action-option order consistency;
- no material regression on the frozen non-arena regression suite;
- improvements persist on held-out graph sizes, horizons, and opponent switches.

If only terminal return improves, report capability learning. If the
communication interventions do not pass, do not call it swarm cooperation.
