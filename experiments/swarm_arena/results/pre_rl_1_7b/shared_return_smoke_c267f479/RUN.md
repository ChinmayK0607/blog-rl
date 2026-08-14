# Shared-return rollout-only smoke

> Superseded for queue-admission purposes by
> `../shared_return_smoke_ab981247/`. This v3 run remains valid replay and
> routing evidence, but its approval did not bind the exact in-memory
> `TrainingSample` payload. No optimizer used this evidence.

This is compact, complete evidence for the first live execution of the
fail-closed shared-terminal-return path. No trainer or optimizer process ran.

## Identity

- Source: `c267f479badac8735c1241c826cb7286ea84c401`
- Model: `/workspace/models/qwen3-1.7b-70d244c`
- Base-config SHA-256:
  `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197`
- Initial and opponent adapter SHA-256:
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Scenario: frozen development curriculum pair 0, critical member, seed
  `4000008`, state SHA-256
  `a2f694eec9b7e69ac14d8b6242ba58950aa8647a8cf60f2c57d886f9ca743124`
- Contract: four independent replicas, verified terminal BLUE return,
  leave-one-out mean baseline, first-turn `BROADCAST` spans only.

## Result

The supervisor independently replayed and approved all four replicas. The
initial curriculum state begins at turn 1 and reached the configured horizon
after one transition, so each complete replica contains 16 decisions and one
replay turn. Returns were:

| Replica | Return | Leave-one-out advantage |
| --- | ---: | ---: |
| 0 | +0.1666667 | +0.1944444 |
| 1 | -0.0833333 | -0.1388889 |
| 2 | 0.0000000 | -0.0277778 |
| 3 | 0.0000000 | -0.0277778 |

The advantages sum to zero. Each replica produced four signed policy
envelopes with the same scalar but distinct policy IDs and owned token spans.
All approvals commit to the same complete-evidence hash. The controller also
constructed and merged the four per-replica routing groups; `--rollout-only`
prevented queue admission and any optimizer step.

Hash-chain verification reported one complete-evidence record, one admission
record, four replicas, four approvals, 16 decisions and one replay turn per
replica. The full Linux Swarm Arena suite passed 65 tests in 44.27 seconds
before this run.

## Artifacts

- `live_rl_diagnostic.json`: 1,476 bytes, SHA-256
  `21225cbd502d4b3624527b9688fe8b567d9917484211955352b63e461084bf28`
- `shared_return_evidence.jsonl`: 441,259 bytes, SHA-256
  `84cf512166c23262645c555e73f473df5e0eec69b73b358a7e4d23dbf061b304`
- `admission.jsonl`: 10,566 bytes, SHA-256
  `8c85cf8533de0236934199b11021bf62cb869228956bd4cc5c14ad414497a70b`

The supervisor signing key is intentionally excluded.

## Launcher notes

The first Linux pytest invocation imported an old cached local-package wheel;
rerunning with `--with-editable ./experiments/swarm_arena` bound tests to the
checked-out commit and passed 4/4 focused tests, then 65/65 full-suite tests.
The first smoke launcher omitted the same editable-package flag and stopped at
module import before contacting vLLM. The corrected invocation changed only
that environment flag and completed cleanly. Neither failed command produced a
rollout, trainer, or optimizer step.

This is a systems/mechanical pass only. Trainer-side log-probability parity and
scientific communication/collapse admission remain pending.
