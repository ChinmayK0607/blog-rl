# Message-credit audit plan

Frozen before the first live result from the message-edge estimator. Source
implementation: `567bc1393d101ca9f4a9613cabececece09a2399`.

## Question

Does dropping one sender's first-turn delivered broadcast produce terminal
credit that is specific to the certified private-information edge, rather than
generic rollout variation or whole-policy behavioral differences?

No optimizer is allowed during these audits.

## Estimand

For pair `p`, agent `i`, and scenario kind `k` in `{critical, decoy}`:

`C[p,i,k] = G_actual[p,k] - G_drop_first_message[p,i,k]`

The paired information-specific effect is:

`D[p,i] = C[p,i,critical] - C[p,i,decoy]`

Critical and decoy scenarios share the sender observation, structural state,
policies, revisions, constraints, and sampling namespace. They differ only in
whether the intended receiver already knows the target fact.

`D[p,intended_sender]` is the primary diagnostic. The two agents that are
neither the certified sender nor receiver are the off-role controls. The
receiver is reported separately because its private observation intentionally
differs across the critical/decoy pair.

## Stage A — two-pair mechanical smoke

Run two critical/decoy pairs. This stage makes no scientific claim. It passes
only if all of the following are exact:

- all actual and four drop branches complete and independently replay;
- run lock, policy revisions, constraints, private contexts, and output hashes
  verify;
- actual and counterfactual emitted first-turn broadcasts are identical before
  delivery intervention;
- exactly one named sender delivery is empty on exactly the intervention turn;
- all other deliveries are unchanged;
- an already-empty message produces an identical trajectory and zero credit;
- each routed BLUE batch owns exactly one actual first-turn BROADCAST span;
- zero action, later-message, opponent, or counterfactual spans enter a batch;
- no NaN, OOM, HTTP, adapter-registry, or structured-generation failure.

Any failure stops before the diagnostic.

## Stage B — 12-pair / 24-scenario diagnostic

Use 12 role-balanced certified pairs, one for every ordered sender/receiver role.
This is an exploratory admission diagnostic, not confirmatory evidence.

Mechanical gates remain 100% exact. In addition, call the result **promising**
only if all of these predeclared conditions hold:

1. The intended sender's first-turn emitted broadcast is identical between the
   matched critical and decoy scenario in 12/12 pairs.
2. The intended sender includes the certified target fact in at least 8/12
   pairs. A run with fewer learning opportunities is capability-limited, even
   if its few nonzero returns look favorable.
3. `mean(D[p,intended_sender]) > 0`.
4. At least 8/12 intended-sender paired effects are positive and no more than
   2/12 are negative. Exact zeros are reported, not discarded.
5. The mean absolute intended-sender paired effect is at least twice the mean
   absolute paired effect over the two off-role controls.
6. At most 4/12 pairs have any nonzero paired effect on an off-role control.
7. The receiver's target action/capture changes more often under critical
   message deletion than matched-decoy deletion; the per-pair counts and exact
   transition evidence must be reported.
8. Raw critical and decoy return, message content, action choice, and capture
   rates are reported alongside credits so a return delta cannot be
   misdescribed as communication use.

Failure of conditions 3, 5, or 6 rejects this estimator for RL. Failure of only
conditions 2, 4, or 7 is inconclusive/capability-limited and requires task or
warm-start improvement before RL. Thresholds are not relaxed after inspection.

## Stage C — 52-pair / 104-scenario confirmation

Run only if Stage B is promising. Use a fresh, role-balanced, seed-disjoint
development subset; do not use frozen final evaluation. The seed/pair is the
independent unit.

Confirmatory gates:

- the paired bootstrap 95% interval for intended-sender `D` is strictly above
  zero;
- at least 75% of intended-sender broadcasts contain the certified target fact;
- the intended-sender mean absolute paired effect is at least twice the
  off-role mean absolute effect;
- the off-role paired mean has its complete 95% interval within the frozen
  equivalence band `[-0.02, +0.02]`;
- the matched-decoy intended-sender mean effect has its complete 95% interval
  within `[-0.02, +0.02]`;
- results persist across all ordered role pairs and are not driven by one
  opponent, side, node size, or action-option ordering;
- every mechanical and safety invariant remains exact.

Only a Stage C pass permits broadcast-only rollout/trainer parity
recertification and a short RL pilot. It does not itself establish learned swarm
cooperation.

## Required outputs

- immutable run identity and public source/model/data revisions;
- raw per-branch replay evidence and compact per-pair table;
- first-turn emitted and delivered message hashes/content;
- legal action sets, chosen actions, target capture, terminal return, and credit;
- intended-sender, receiver, and off-role summaries;
- exact sign counts and deterministic paired bootstrap intervals;
- hash-chained supervisor admissions/rejections;
- wall time, throughput, peak VRAM, failures/retries, and estimated cost;
- explicit verdict and decommission status.
