# V14.4: policy-local parity recovery

Status: CPU-frozen; publication, anonymous verification, credentials, and a
fresh exact-runtime certificate remain required before renting another GPU
pod. No GPU is currently allocated.

## Decision

V14.4 repeats the V14.3 policy-routed curriculum from the V13 update-80 parent
initializer. It does not resume the seven accepted V14.3 updates because doing
so would bind one optimizer trajectory across two different runtime-parity
contracts. The reward, credit estimator, opponent rotation, case pool, policy
routing, optimizer, DPPO mask, dtypes, stage gates, and 40-update maximum are
unchanged.

The only planned runtime difference is the numerical contract:

- trainer mean mismatch KL is explicitly `0.002`, the pre-existing 4B
  async-admission ceiling, rather than the certifier's generic historical
  default or the old loose `0.15` template value;
- one fresh 32-sample probe must contain exactly eight samples for each of
  `blue-0` through `blue-3` and must exercise three distinct rollout servers;
- the pooled probe and every policy-local subset must independently pass the
  exact trainer-declared thresholds;
- the certificate retains its full threshold body and policy-local metrics,
  and paid-run preflight rehashes both before update 1;
- local token export is enabled for post-failure diagnosis and is excluded
  from the compact public mirror.

This ceiling is prospective, not fitted to the rejected V14.3 value. A fresh
certificate can still reject it before optimizer work.

## Preventive launch fixes

The CPU audit found and repaired four recurrent sources of wasted compute:

1. Supplying `--trainer-config` no longer inherits the certifier's hidden
   `0.0005` CLI default. Thresholds come from the resolved trainer config;
   explicitly conflicting CLI values fail closed.
2. A pooled pass cannot hide a failing policy lane. Empty, missing,
   imbalanced, zero-token, or failed policy-local probe coverage is rejected.
3. The serving smoke probe and parity probe must bind the same three distinct
   URLs. Repeating one server three times no longer satisfies the certificate.
4. Per-policy orchestrator LoRA rank and alpha now derive from the trainer
   instead of being hard-coded to rank 16 / alpha 32. Preflight independently
   checks alias, rank, and alpha before training, preventing metadata-invalid
   rank-32 exports.
5. The parity certifier's temporary four-policy run configs now use those same
   trainer-derived aliases, ranks, and alphas. Certification therefore cannot
   test a rank-16 orchestration surface and then approve a rank-32 paid run.

The existing 10,800-second pulse wait, append-only incident policy, compact HF
mirror, W&B isolation, `$15` cap, nine-hour TTL, and immediate exact-pod
teardown remain mandatory. A watcher should additionally alert on no durable
progress, missing service sessions, disk growth from local token exports, a
ready barrier without a pulse process, and idle GPUs after a terminal state.
It may perform only the already-defined narrow operational recoveries; it may
not change scientific settings or resample until favorable.

## Frozen identities

- Parent V14.3 CPU-bundle body:
  `2741872a8a4d9f632752c56a7f0c58537155812679427ea5c355d5806401ea32`
- V14.4 trainer config file:
  `efd3cb87221e7a0dafa055ce67959e6b8a16962264a9e7ff14bdc1c94ebc83c9`
- V14.4 CPU-bundle body:
  `ef4c9c614856edbf23b525724e3cc9524a8fe749e6e7e5fc2e6f4e6dd887aef3`
- Unchanged curriculum body:
  `197e50e9253798f75a83d29e306c321618c7b9633d3b86b4133e54bb0bb8e0e5`
- Unchanged stage-gate body:
  `27098650c9e6f604e8393a75fc01cb0a0e6c694cf80286ba7d8965de84a1f8c2`

These identities are local until the final source commit is intentionally
published and independently verified. Renting before that verification is
prohibited.
