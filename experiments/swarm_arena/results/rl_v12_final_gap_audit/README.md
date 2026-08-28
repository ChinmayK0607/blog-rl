# V12 final rollout-gap audit

This directory contains compact, training-only evidence derived from the complete
160-update V12 run. It does not contain raw rollouts, model weights, held-out
cases, or a formal checkpoint-selection result.

## Main findings

- Critical receivers chose the fact-supported target in 95.91% of 880
  replicas.
- Decoy receivers chose private evidence in 70.00% of 640 replicas. In the
  final 20 updates, misleading-target selection was 20.00%, 6.25%, 37.50%,
  and 5.00% for blue-0 through blue-3.
- Only 16.63% of ordinary focused replicas had nonzero advantage. The final
  window rates were 10.00%, 12.50%, 0%, and 0% by policy slot.
- Active/private-supported target actions had mean paired effect +0.04587;
  alternate/message-following target actions had mean effect -0.06786.

These results support a role-adaptive V13 continuation and a mandatory
ordinary pass@4 learnability screen before any optimizer update. They do not
support a held-out communication claim.

## Files

- `compact_rollout_gap_audit.json`: compact complete-run aggregates plus hashes
  of the public progress and development rows used.
- `v13_repair_case_selection.json`: deterministic training-case ranking and
  V13 role bands. Its scope is explicitly training-only.

Large public checkpoints and rollout rows remain off-device in the Hugging
Face mirror. The untouched frozen evaluation was not opened.
