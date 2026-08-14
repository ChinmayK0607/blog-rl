# Fresh 1.7B broadcast-priority confirmation

- Status: completed; capability replication only, not RL admission
- Source: `a1c3740aec24809d9a3a9f2042005f36261f3ab9`
- Policy: `capability-sft-1.7b`
- Policy revision: `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Data: previously untouched training-manifest pairs 12--23
- Samples: 120 constrained broadcasts (60 per prompt variant)
- Runtime: 2026-08-14 15:06:40--15:08:41 UTC (about 121 seconds)
- Hardware: one NVIDIA RTX A6000 48 GB
- Optimizer steps: zero

The preselected `actionable_priority` instruction passed every frozen
confirmation gate: 60/60 protocol-valid, 58/60 target facts (96.67%), and
target-fact majority on 12/12 pairs. The current prompt produced 43/60 target
facts (71.67%) and majority on 9/12 pairs. The improvement was 25 percentage
points, above the predeclared 20-point threshold.

This disjoint replication permits a versioned production-prompt update and a
new causal message-credit audit. It does not itself validate the estimator or
authorize optimizer steps.

Artifact:

- `probe.json`: 147,767 bytes; SHA-256
  `c2da6a11a4cd4cd5118881ea9a3e84d817936eacafd3c963d82e3b492f3e41f7`

