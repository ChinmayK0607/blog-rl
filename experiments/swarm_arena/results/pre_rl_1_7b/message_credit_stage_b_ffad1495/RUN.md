# Stage B — message-credit diagnostic

Status: completed; hard rejection. No optimizer, Stage C, or RL pilot ran.

- Rollout source: `ffad14954f4f7b8e695bc394a79ae3f4f5b39ffb`
- Analyzer source: `19fd2c68a946c440b2b168ac44ecea5150ca1e92`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Initial adapter / opponent revision:
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Fixed audit: 12 role-balanced pairs / 24 alternating critical-decoy
  scenarios, message-drop, horizon two, rollout-only.
- Timing: started 2026-08-14 14:36:47 UTC; completed 14:38:57 UTC
  (about 130 seconds). A6000 peak reservation was 41.0/49.1 GiB; observed
  full-load temperature was 51 C.

## Frozen gates

- Sender message identical: 12/12 — pass.
- Certified target fact present: 5/12 — fail (threshold 8/12).
- Intended sender D: mean +0.0342928735; 4 positive, 1 negative, 7 zero —
  sign-count gate fails.
- Mean |intended D|: 0.0581023973; mean |off-role D|: 0.0409699855;
  ratio 1.41817 — fail (threshold 2).
- Pairs with nonzero off-role D: 5/12 — fail (maximum 4).
- Receiver target effects: critical 7/12; decoy 7/12 — fail.

The estimator is rejected for RL because both localization and off-role gates
fail. Positive mean alone is not sufficient.

## Artifact integrity

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `message_credit_evidence.jsonl` | 2,422,092 | `9827b6578cf8b6e2e6a5320777e0aebd4d00d347f1b254553656b82bb87addef` |
| `admission.jsonl` | 65,594 | `d5ac8c4f3a1c2422b13820978415a4ec256eb91f869fe075d39d6bd32df5d2e1` |
| `live_rl_diagnostic.json` | 20,493 | `e40e37512e8ff2704ee7386f68b46d2bdf141020890ef55cab2d3d141693ce87` |
| `stage_b_paired_summary.json` | 16,692 | `bb55c977697a1419118531888afd20bcb8953346a461be32b3ab4996d65fcdc4` |

The evidence JSONL has 24 hash-chained records; each contains five replayed
branches, raw emitted and delivered broadcasts, legal and chosen actions,
events, target transitions, credits, and request/output hashes.

## Invocation-only detours

- An initial `--input` analyzer invocation was unsupported; the evidence path
  is positional.
- The first public analyzer revision missed a `canonical_sha256` import; its
  focused test failed before analysis. The fixed public analyzer above passed
  the focused test and generated this summary.

No supervisor key, token, model, adapter, cache, or verbose server log is
included. vLLM was stopped after completion; the A6000 was at 1 MiB / 0% /
32 C when packaging began.
