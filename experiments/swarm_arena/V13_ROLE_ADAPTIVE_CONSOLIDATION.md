# V13: Role-Adaptive Consolidation

Status: CPU design and interim schedule audit complete. GPU launch is deliberately blocked until V12 finishes and a V12 checkpoint is formally selected.

## What the first 100 V12 updates say

This diagnosis uses only V12 training rollouts through update 99 and compact development evaluations through update 80. It is not a held-out result.

- Factual following is nearly saturated. Critical receivers select the fact-supported target in 95.8% of 520 replicas, and the average factual counterfactual effect is +0.0984.
- Misleading-message robustness is improving but asymmetric. Across decoys, selection of the receiver's privately supported target rose from 56.3% at updates 0–19 to 80.0% at updates 80–99.
- Blue-0 is the remaining clear failure slot. At updates 80–99 it still follows the misleading alternate target 60% of the time; blue-3 is at 4.2% and the other slots are materially better over the broader repair window.
- Ordinary-map credit remains sparse. Only 14.4% of focused ordinary replicas have nonzero advantage overall; in updates 80–99 the rate is 0% for blue-0 and 11.1% for blue-3. This is the main retention/optimality bottleneck.
- The communication actions themselves are capture-heavy, but ordinary actions remain diverse. This is task structure, not enough evidence to call action collapse.
- The training signal points in the intended direction: active/private-supported target actions have mean effect +0.0674, while alternate/message-following target actions have mean effect -0.0910. V13 therefore does not need a new communication reward.

The update-80 compact development pulse is promising but not decisive: normal-minus-dropped return is +0.0315, critical-minus-decoy specificity is +0.0468, and RL-specific communication lift is +0.0480, but all three confidence intervals still include zero. Legacy and hard ordinary return are approximately retained at that checkpoint (-0.0058 and +0.000003 versus SFT), also with wide intervals.

## V13 objective

Turn V12's fragile, role-asymmetric robustness into a general behavior while preserving ordinary gameplay. V13 is not a restart and does not relearn basic factual broadcasting.

The initializer is the earliest formally selected V12 checkpoint. If V12 selects no checkpoint, V13 does not launch automatically.

## Frozen training shape

V13 is an 80-update continuation with four groups per update:

- 140 fresh ordinary groups for capability retention and broader map diversity;
- 80 misleading-message decoy challenges;
- 80 factual critical controls matched to every decoy topology and world;
- 20 additional hard factual rehearsals.

The 80 decoy challenges are role-adaptive but retain coverage of every policy:

| Receiver | Decoy groups | Reason |
| --- | ---: | --- |
| blue-0 | 32 | dominant residual misleading-message obedience |
| blue-3 | 20 | second repair allocation and weak ordinary credit |
| blue-1 | 14 | maintenance floor |
| blue-2 | 14 | maintenance floor |

Every decoy remains a matched subset of critical cases. The schedule interleaves roles and rotates base, SFT, historical, and current opponents. All four LoRA policies remain separate.

## Reward and credit

The reward remains verified terminal team return. There is no additive message reward and no supervised message target.

- Critical receiver advantage: factual message return minus receiver-only target-swapped return.
- Decoy receiver advantage: misleading target-swapped return minus factual/private-evidence return.
- Ordinary advantage: terminal-return leave-one-out baseline in the current implementation.

The ordinary path is the weak point. Before V13 GPU launch, fresh ordinary candidates must pass a training-only pass@4 screen proving nonzero credit in every policy slot. A privileged state-conditioned critic is the preferred later variance-reduction upgrade for ordinary trajectories, but it must remain a baseline only: terminal reward and the policy objective cannot change, and rollout/trainer log-probability checks must still pass.

## Three stages

1. `role_adaptive_conflict_repair`, 20 updates: two ordinary, one critical, one matched decoy per update; two remaining handoff turns.
2. `hard_negative_transfer`, 40 updates: alternate between one/two and two/one ordinary/critical groups, always with one matched decoy; four remaining turns.
3. `capability_consolidation`, 20 updates: return to two ordinary plus one matched critical/decoy pair on larger maps and longer horizons.

Development pulses run at updates 0, 20, 40, 60, and 80. Selection is the earliest checkpoint with positive semantic return and specificity, both ordinary clustered lower bounds at least -0.02, and no receiver-level challenge regression. Stop after two consecutive checkpoints with no blue-0 robustness improvement and any ordinary regression.

## What must happen before GPU use

1. Refresh the compact gap screen from the complete V12 progress artifact. The current case selection is explicitly interim.
2. Bind the formally selected V12 four-policy checkpoint and verify all adapter hashes anonymously.
3. Run the fresh ordinary pass@4 screen and prove nonzero signal across all four slots.
4. Rebuild and freeze the V13 curriculum, schedule audit, production plan, development thresholds, and hashes.
5. Run Linux tests and one bounded live-rollout preflight. Do not repeat the former certificate bureaucracy; the launch check is limited to imports, adapter binding, one parity batch, signal coverage, and mirror/W&B health.

The untouched frozen evaluation remains unopened until a V13 development checkpoint is selected.
