# V13: Role-Adaptive Consolidation

Status: completed-run CPU design and schedule audit complete. GPU launch is blocked only on the ordinary pass@4 signal screen.

## What the complete 160-update V12 run says

This diagnosis uses the complete V12 training rollouts and compact development evaluations through update 160. It is not a held-out result.

- Factual following is nearly saturated. Critical receivers select the fact-supported target in 95.9% of 880 replicas.
- Misleading-message robustness improved but remains asymmetric. Across all 640 decoy replicas, the receiver follows private evidence 70.0% of the time. In updates 140–159, misleading-target obedience is 20.0% for blue-0, 6.25% for blue-1, 37.5% for blue-2, and 5.0% for blue-3.
- The final hard-case ranking therefore shifts repair weight from the update-100 blue-0 diagnosis toward blue-2 while retaining coverage of every slot.
- Ordinary-map credit remains sparse. Only 16.6% of focused ordinary replicas have nonzero advantage overall; in updates 140–159 the rate is 10.0%, 12.5%, 0%, and 0% for blue-0 through blue-3. This is the main retention/optimality bottleneck.
- The communication actions themselves are capture-heavy, but ordinary actions remain diverse. This is task structure, not enough evidence to call action collapse.
- The training signal points in the intended direction: active/private-supported target actions have mean effect +0.0459, while alternate/message-following target actions have mean effect -0.0679. V13 therefore does not need a new communication reward.

The update-160 compact development pulse is the strongest V12 checkpoint: normal-minus-dropped return is +0.0564, critical-minus-decoy specificity is +0.0578 with a fully positive interval `[+0.0101,+0.1032]`, and RL-specific communication lift is +0.0728. Hard and legacy ordinary point effects are +0.0632 and +0.0012, but the legacy interval remains wide. This is strong development evidence, not a held-out claim.

## V13 objective

Turn V12's fragile, role-asymmetric robustness into a general behavior while preserving ordinary gameplay. V13 is not a restart and does not relearn basic factual broadcasting.

The initializer is the four distinct, anonymously hash-verified V12 update-160 adapters. This is explicitly a non-admitted continuation warm start, not a retroactive claim that V12 passed formal selection. V12 itself used the same scientifically valid distinction when it continued from non-selected V11-u180.

## Frozen training shape

V13 is an 80-update continuation with four groups per update:

- 140 fresh ordinary groups for capability retention and broader map diversity;
- 80 misleading-message decoy challenges;
- 80 factual critical controls matched to every decoy topology and world;
- 20 additional hard factual rehearsals.

The 80 decoy challenges are role-adaptive but retain coverage of every policy:

| Receiver | Decoy groups | Reason |
| --- | ---: | --- |
| blue-0 | 23 | persistent hard-case message obedience |
| blue-1 | 13 | maintenance floor |
| blue-2 | 25 | largest final-window residual failure |
| blue-3 | 19 | hard-case maintenance and zero final-window ordinary credit |

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

Development pulses run at updates 0, 20, 40, 60, and 80. Selection is the earliest checkpoint with positive semantic return and specificity, both ordinary clustered lower bounds at least -0.02, and no receiver-level challenge regression. Stop after two consecutive checkpoints with no improvement in the worst receiver's challenge robustness and any ordinary regression.

## What must happen before GPU use

1. Run the frozen 256-game ordinary pass@4 screen and prove nonzero signal across all four slots and opponent families.
2. Bind the already verified V12-u160 four-policy warm start and recheck adapter hashes anonymously on the GPU host.
3. Build the final trainer/production configs from the frozen V13 curriculum and screen result.
4. Run Linux tests and one bounded live-rollout preflight. Do not repeat the former certificate bureaucracy; the launch check is limited to imports, adapter binding, one parity batch, signal coverage, and mirror/W&B health.

The untouched frozen evaluation remains unopened until a V13 development checkpoint is selected.
