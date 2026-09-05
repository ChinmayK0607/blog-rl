# Prospective measurement and reliability repair

Status: implementation and CPU validation, **not GPU-admitted or a frozen new training contract**.
Historical rows, gates, curriculum, dtypes, optimizers, rewards, and quarantine limits remain unchanged.

## What each comparison means

1. An SFT-vs-SFT harness null checks the evaluation path; it is not a warm-start measurement.
2. The actual step-zero four-policy roster is evaluated against the immutable SFT reference.
3. Later checkpoints retain the established SFT comparisons and separately report matched
   checkpoint-minus-initializer returns and communication-effect changes, grouped by independent bundle.
4. Each staged pulse verifies every registered adapter's on-disk weight hash against the ready
   checkpoint or frozen opponent snapshot on every server. This binds registry paths to artifacts;
   it does not independently prove GPU-resident tensor contents. Exact-host runtime parity remains required.
5. Old SFT-harness rows cannot seed the corrected initializer comparison. Use a new run identity.

## Semantic development probe

`run_progress_eval_v4.py --tier semantic_pulse --rl-specific-communication` adds a separate
development-only probe: six ordinary legacy cases, six hard cases, and six independent
two-world critical/decoy handoff bundles, both sides, normal/drop/receiver-only content swap.
It has **264 rows**, not 192 independent observations. Later reuse of 96 SFT rows leaves
168 fresh games. It must have its own prospective budget and gate declaration before being
connected to a staged run; the old 192-row gates cannot admit it.

Primary communication estimand: intention-to-treat normal-minus-receiver-only-target-swap
return, with critical-minus-matched-decoy specificity. Ineligible sender messages remain in
the denominator; do not retry or discard them. Report absolute normal and swapped returns,
eligibility, initial receiver action, and ordinary capability alongside sensitivity. Report
both improvement over SFT and change from the actual initializer. Drop/shuffle/delay remain
supporting diagnostics, not substitutes for semantic evidence. No frozen held-out files are opened.

## Controlled follow-through experiment

Preregister **first-action-only versus multi-turn receiver credit**. Use identical distinct
initial policies, predetermined fixed curriculum and opponent schedule, equal logical-slot and
sampling budgets, and paired training seeds. Do not combine this ablation with adaptive
scheduling. Freeze sender *behavior* through a separate immutable sender-policy route, not
merely a zero BROADCAST loss mask on a shared trainable adapter. Before launch, a regression
must prove sender requests still use the frozen weights after receiver checkpoint reloads.
That serving/training route is not yet implemented or admitted in this patch.

Measure initial correct action, later target retention, redundant teammate actions, and
terminal return. The only arm difference may be the declared receiver ACT credit offsets.
No message-obedience reward or favorable resampling. Replicate the chosen fixed recipe over
at least three prospectively declared training seeds before a final communication claim.

## Independent evidence and stopping

Choose the minimum worthwhile effect *before new outcomes*. Proposed planning target: 0.02
terminal-return units, two-sided alpha 0.05, power 0.8. Estimate bundle-level variance from
an explicitly identified development pilot; never count world/side/intervention rows as
independent units. `required_independent_units` provides normal-approximation planning only;
confirm with simulation/cluster resampling before freezing the sample count. For illustration,
SD 0.10 needs 197 bundles, not 197 episode rows. No measured variance or powered claim is asserted here.
Use a fixed sample/seed count and no early success peeking; interim gates remain training controls.

Distinguish: (a) claim not established because its interval spans zero, (b) evidence ruling
out a prospectively meaningful benefit, and (c) operational interruption before the hypothesis
was tested. A negative point estimate alone does not establish (b).

## Credit-estimator audit

`audit_paired_credit_toy.py` exactly enumerates two Bernoulli replicas and invokes the actual
`paired_terminal_contrast_advantages` function. With a fixed counterfactual policy, independent
baseline randomness recovers the objective gradient; shared randomness is generally correlated
with the factual action. Replica centering need not remove that term. This is a counterexample
to an automatic unbiasedness claim, not a complete diagnosis of the multi-turn/shared-parameter
training estimator. Define the intended objective and audit both branch score terms and
multi-turn coupling before changing the production estimator or introducing a critic.

## Operational admission

- Freeze profile evidence, exact trainer/inference config hashes, topology, all evaluation games,
  training time, checkpoint overhead, setup remainder, safety factor, and final-sync reserve.
- The old full schedule has 672 fresh games. At 1.3 games/minute, evaluation alone takes
  about 8.6 hours *without* safety margin. A nine-hour launch is rejected, not silently shortened.
- `preflight_staged_budget.py` supports offline `--available-seconds` admission before rental
  and live `--deadline-epoch` admission before trainer startup. The same report supplies controller
  barrier and pulse wait timeouts. Estimates are not completion guarantees.
- Higher game concurrency is not enabled by this patch. Benchmark fixed predetermined games
  and production-shaped parity on separately budgeted hardware before binding a faster profile.
  Never infer throughput from rewards or select favorable parity samples.
- Expand a fixed parity regression corpus to long contexts, mixed adapters, concurrent requests,
  and checkpoint reloads; preserve prior failures with their hashes. Do not substitute trainer
  logprobs for serving logprobs or loosen the numerical gates.
- Controller/pulse failures now produce durable terminal markers even if the trainer remains
  resident. Install the exact-pod teardown supervisor before GPU setup; preserve/sync/hash-check
  evidence before authorized deletion. No rental is performed by these repair tools.

## Remaining launch blockers

New source publication and CPU-bundle binding, truthful feasible operational measurements,
exact-host runtime certification, and a separately frozen scientific plan for any new semantic
gates or sender-frozen ablation. Do not reuse the V14.7 CPU bundle to claim these changed files are certified.
