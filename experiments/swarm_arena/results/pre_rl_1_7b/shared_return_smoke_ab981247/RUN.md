# Sample-bound shared-return rollout-only smoke

This is the authoritative compact evidence for supervisor/router v4. It closes
the post-approval sample-substitution gap found during adversarial review of the
v3 smoke. No trainer or optimizer process ran.

## Identity and contract

- Source: `ab981247772c66cff5f5b00922fc1fa9c8f1aea0`
- Supervisor: `arena-fail-closed-supervisor-v4-sample-bound-shared-return`
- Model: `/workspace/models/qwen3-1.7b-70d244c`
- Base-config SHA-256:
  `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197`
- Initial and opponent adapter SHA-256:
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Scenario: frozen development curriculum pair 0, critical member, seed
  `4000008`, state SHA-256
  `a2f694eec9b7e69ac14d8b6242ba58950aa8647a8cf60f2c57d886f9ca743124`
- Credit: four independent replicas, verified terminal BLUE return,
  leave-one-out mean, first-turn `BROADCAST` spans only.
- Sample commitment: prompt IDs/masks, completion IDs/masks, rollout log
  probabilities, temperatures, every dynamic allowed-token row, environment,
  and training mode.

## Result

All four replicas replayed and were approved. The initial state starts at turn
1 and reached the horizon after one transition, so each complete replica has 16
decisions and one replay turn.

| Replica | Return | Leave-one-out advantage |
| --- | ---: | ---: |
| 0 | -0.0833333 | -0.1111111 |
| 1 | 0.0000000 | approximately 0 |
| 2 | +0.0833333 | +0.1111111 |
| 3 | 0.0000000 | approximately 0 |

The advantages sum to zero within floating-point tolerance. Every decision has
one allowed-token row per completion token. Each of the four approvals contains
four non-empty sample commitments, one for each independently owned policy
span, and all approvals commit to the same complete-evidence hash. The
controller successfully recomputed every sample hash, constructed four
isolated policy batches per replica, and merged them. `--rollout-only`
prevented queue admission.

The adversarial test replaces prompt token IDs with a same-length array after
approval; the router rejects it with `committed training-sample payload
mismatch`. The focused Linux suite passed 4/4, and the complete suite passed
65/65 in 41.07 seconds. Both audit hash chains verify. The post-run A6000 state
was healthy at 41,032 MiB VRAM, 0% idle utilization, and 31 C.

## Artifacts

- `live_rl_diagnostic.json`: 1,476 bytes, SHA-256
  `729bb640696f924bdfe2eb6ea7ffab7ec83e4cd03c1307479e5b51439182bb9e`
- `shared_return_evidence.jsonl`: 474,579 bytes, SHA-256
  `6fae21eba1438282023df10ef808abe26b24ac75364ca9acd940f89c03c0c0d0`
- `admission.jsonl`: 12,026 bytes, SHA-256
  `1c2115b593556ef75cf08b7785d50b00cd123671d7b3da7634c2e49ac93dfd58`

The supervisor signing key is intentionally excluded. The first v4 launcher
used a mistyped full source SHA and was rejected by the immutable commit guard
before any vLLM request; the corrected source identity completed cleanly.

This is a systems/mechanical pass only. Trainer-side numerical parity and the
scientific opponent-pool, intervention, KL, and collapse gates remain pending.
