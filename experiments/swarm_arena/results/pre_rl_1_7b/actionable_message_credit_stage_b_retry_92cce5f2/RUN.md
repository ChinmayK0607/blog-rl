# Actionable-prompt Stage B causal re-audit

Status: completed; rejected under the frozen gates. No optimizer, Stage C, or
RL pilot ran.

- Source: `92cce5f227f6aac13d90a9e8cf6218cb967156b2`
- Prompt: `arena-episode-v5-actionable-broadcast-priority`
- Policy/opponent revision: `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Data: development pairs 12--23, 24 alternating critical/decoy scenarios
- Runtime: 2026-08-14 15:27:41--15:29:25 UTC (104.26 seconds launcher to
  close; 80.88 seconds from run-directory creation to evidence close)
- Hardware: one NVIDIA RTX A6000 48 GB
- Optimizer steps: zero

## Frozen result

Passed: identical sender message 12/12; certified target fact 11/12; intended
mean paired effect +0.07050; receiver target effect critical 12 versus decoy 3.

Failed: intended sign count 7 positive / 1 negative / 4 zero (requires at
least 8 positive); intended/off-role localization 1.0767x (requires at least
2x); nonzero off-role effect 8/12 (maximum 4). Verdict: rejected.

The prompt repaired fact selection and revealed a real immediate communication
channel. The two-turn terminal-return deletion estimator remains too coarse to
localize that channel to one sender reliably. This is not an RL-ready credit
signal.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `message_credit_evidence.jsonl` | 2,480,555 | `812d2e838130f07544e6c412aa2e1bf481b749d03b32528fd4fecabd265bfad3` |
| `admission.jsonl` | 68,575 | `efafad1f7584013a7b14d66b41f679ab2ee91421ed4644182a87f98d064c182c` |
| `live_rl_diagnostic.json` | 21,481 | `45d3862f795b35142f135beb2997b6a201759589469ad331f48618ccf0635e24` |
| `stage_b_paired_summary.json` | 17,154 | `79d9eccb627288b15d5adafa9b2ed2686e35591c69ce27d139a0551f680ef798` |

