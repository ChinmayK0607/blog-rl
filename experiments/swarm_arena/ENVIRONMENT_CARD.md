# Swarm Arena environment card

## Research question

Can small language-model agents learn decentralized communication and joint
allocation policies that close the gap to a centralized oracle under partial
observation? The benchmark isolates that question in a fast symbolic 4v4 game;
it does not claim to simulate real network security.

## Game

- Two teams of four agents act simultaneously on a connected graph.
- Node identifiers are randomly assigned and do not reveal ownership.
- Nodes have an owner, value, critical flag, fortification level, exposure state,
  compromise state, and symmetric adjacency.
- Each agent has a position, a private timestamped knowledge map, and 0–4 resource.
- Remote observations become stale. Adjacent node identifiers are known, but an
  unseen neighbor must be scanned before its state is visible.
- Episodes use a fixed horizon (eight turns by default) or end when a team controls
  no nodes.

### Actions

`SCAN`, `PROBE`, `CAPTURE`, `FORTIFY`, `RECOVER`, `TRANSFER`, and `WAIT` are the
only actions. The environment enumerates the complete legal action set for each
agent every turn.

Resolution order is fixed and independent of dictionary or worker order:

1. validate all actions against the pre-turn state;
2. transfer resources (available to the receiver next turn because legality is
   fixed before resolution);
3. recover and fortify;
4. scan and probe;
5. capture;
6. refresh local observations and advance the clock.

This creates genuine simultaneous coordination. Multiple probes can remove
multiple shield levels, a probe can enable a teammate's same-turn capture, and a
simultaneous fortification can block insufficient probing. Opposing captures of
a neutral node are contested rather than resolved by iteration order.

## Reward

The legacy one-turn solver score is zero-sum and includes symmetric control,
resource opportunity, information gain, and invalid-action terms. The main
multi-turn RL task uses only the normalized terminal control-margin delta. It
has no additive message, capture, silence, sender, curriculum, or learned-judge
bonus. Invalid structured output fails admission instead of becoming a
tradeable negative reward. No model-written text is passed to a learned reward
model.

Action redundancy is not inferred from matching target strings. Evaluation uses
a leave-one-agent-out counterfactual: an action is redundant only if replacing it
with `WAIT` does not reduce team reward. This avoids falsely penalizing useful
multi-probe or multi-fortify coordination.

## Communication protocol

The primary benchmark uses machine-checkable semantic broadcasts:

```json
{
  "facts": [
    {"node": "V42", "owner": "RED", "status": "EXPOSED", "value": 3, "critical": true, "observed_turn": 3}
  ],
  "intent": {"type": "CAPTURE", "target": "V42"},
  "request_resource": 0
}
```

Facts must exactly match the sender's timestamped observation, intentions must be
legal for the sender, and the entire response must match the schema. Extra prose,
extra keys, fabricated state, future timestamps, and out-of-range action IDs fail.
Natural-language communication can be added as a separate experiment, but it is
not mixed into the primary score because semantic text grading would introduce
false positives and negatives.

## Evaluation design

The frozen manifest contains 60 cases:

- 20 graphs with 12 nodes;
- 20 topology-OOD graphs with 13 nodes;
- 20 topology-OOD graphs with 14 nodes;
- balanced coverage of balanced, aggressive, and defensive opponent policies.

Every case is tested with generated, dropped, reference, and shuffled messages.
The generated condition is repeated under three action-order permutations. The
report records strict protocol rates, exact oracle regret, optimal-outcome rate,
counterfactual redundancy, communication effects, topology/policy slices, and
95% intervals. The manifest hash is included in every result.

The oracle enumerates every legal four-agent joint action against the specified
opponent. A rollout is optimal when its realized reward matches the best reward;
it is not required to reproduce one arbitrary canonical assignment. This avoids
false negatives when several joint actions are equally good.

The RL progress evaluation is separate from this legacy SFT/protocol suite. It
preserves the existing 14/16-node, 6/8-turn ordinary OOD baseline and adds
seed-disjoint 18/20-node, 8/10-turn ordinary maps. Its communication suite uses
paired latent worlds: the receiver can legally capture either candidate in both
worlds, but only a remote sender knows which target is exposed. Matched decoys
give that fact directly to the receiver and therefore have exact zero message
value. Candidate-minus-SFT return measures game capability; communication is a
separate causal endpoint requiring normal messages to beat dropped, shuffled,
delayed, and zero-budget conditions without a decoy effect.

## Data isolation and reproducibility

- Evaluation seeds are reserved in code and rejected by the SFT splitter.
- SFT rows are split by whole procedural seed, never by agent-level row.
- Node names, action order, topology, state, resources, and observations vary by
  deterministic seed.
- Dataset rows carry generator, prompt, and dataset versions plus a content hash.
- An independent audit reconstructs states and rechecks every protocol target,
  solver label, split, action position, duplicate ID, and leakage sentinel.

## Required baselines before research claims

Report random-legal, local deterministic, no-message, shuffled-message,
reference-message, base-model, SFT, and MARL policies. The centralized oracle is
an upper bound, not a deployable policy. A claim about learned coordination
requires improvement over the local and no-message baselines, degradation under
shuffled messages, and consistency across topology and opponent-policy slices.
