# Mask-audited four-group parity certificate

Verdict: **rejected for RL admission**. The fail-closed gate worked; thresholds
were not changed and no training checkpoint was produced or promoted.

## Immutable inputs

- Source commit: `5e0e87ed0aaf8de74c638422b246afe22da4a733`
- Base revision: `1ddb5b89`
- Initial/opponent adapter SHA-256:
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Serving config: `configs/inference_1_7b_parity_strict.toml`
- Trainer config: `configs/rl_1_7b_multirun.toml`
- Curriculum: two development pairs, each evaluated as one critical and one
  matched decoy group; four independent replicas per group.

## Rollout checks

- Four groups, 16 trajectories, 256 total model decisions.
- All 256 decisions completed on transport attempt one.
- At every sampled token whose legal set fit in the requested top 20, the
  server's finite top-logprob token set exactly equalled the independently
  reconstructed xgrammar set. Masked vLLM `-9999` sentinels were excluded.
- Every state transition and terminal return was replayed; private contexts,
  outputs, revisions, namespaces, exact sample content, policy routing, signed
  approvals, and both hash chains passed.
- The probe contains 64 first-turn broadcast samples, 4,305 completion tokens,
  and 3,672 branching tokens.

## Frozen parity result

Five aggregate/tail components and four-policy isolation passed:

- mean absolute log-probability error: `0.002640` (limit `0.005`)
- p99 absolute log-probability error: `0.069428` (limit `0.12`)
- p99 probability error: `0.029992` (limit `0.05`)
- probability-error tail fraction: `0.003949` (limit `0.005`)
- mean mismatch-KL: `0.0001818` (limit `0.0005`)
- optimizer parameter sets were disjoint; only `run_blue_0` changed in the
  disposable isolation step.

Two worst-case components failed:

- maximum probability error: `0.140238` (limit `0.10`)
- maximum mismatch-KL: `0.100478` (limit `0.08`)

Because the exact server mask matched, this is not a hidden legal-token or
normalization-set bug. It is rare numerical drift between vLLM sampling
log-probabilities and the Prime teacher-forced trainer path. The trainer's
pre-step parity gate must remain a hard stop until the serving and training
forward paths are made numerically compatible on a broad probe.

## Artifacts

Raw hashes are recorded before gzip compression:

- raw probe: `fe0ae52d78c3e85607bd1c74a265a7f7721df917fedb407c1d67d75b28d3162d`
- raw evidence: `79660dcdfad0580c529bd7d19b43d77274b89b9f87aca252bf3bab7d80790d2b`
- admission: `9e6acc50cbde2284663f19d7385c88985a7aff4d2bb1a2906ab93f19859b948f`
- parity report: `00b64f0be0f58a0e6ced7d228b7c72d1512b14e25d6f9af01cfef651e121ea2a`
- diagnostic: `0d366c918005b185b73c72ebbff9e8896445eb6440c731c0fb352b7f9e504d6f`
- compressed probe: `d4abdb1e9e1e586053ce906a0dc8c89672cba0bbd29d9f94530295e0b5f7e385`
- compressed evidence: `7cc5cb66b70e6b05e356d1a248c6aaaafd46cba20d1657ef249c75874f69e72f`

The supervisor signing key and all checkpoints are intentionally excluded.

