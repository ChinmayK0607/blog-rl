# 1.7B broadcast-priority capability probe

- Status: completed; capability probe only, not RL admission
- Source: `e68c826aab419d3e93fe5f492faccb356e420211`
- Policy: `capability-sft-1.7b`
- Policy revision: `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Data: training-manifest pairs 0--11, five common-random repetitions each
- Samples: 180 constrained broadcasts (60 per prompt variant)
- Runtime: 2026-08-14 15:00:51--15:03:10 UTC (about 139 seconds)
- Hardware: one NVIDIA RTX A6000 48 GB
- Optimizer steps: zero

The generic `actionable_priority` instruction passed every predeclared
capability gate: 60/60 protocol-valid, 57/60 target facts (95%), and target-fact
majority on 12/12 pairs. The current prompt produced 44/60 target facts
(73.33%) and majority on 10/12 pairs, so the absolute improvement was 21.67
percentage points. `recency_scoped` matched the current prompt's target rate.

This result only establishes that a non-leaking instruction can improve the
warm start's selection of useful private facts on already-inspected training
pairs. It does not validate causal credit, communication gains, or RL readiness.
A fresh-pair confirmation is required before changing the production prompt.

Artifact:

- `probe.json`: 222,527 bytes; SHA-256
  `1f127c43d10358ffcc11bfe78ade5e2999ab5f4d4783b398d67f3e8e82d6e952`

