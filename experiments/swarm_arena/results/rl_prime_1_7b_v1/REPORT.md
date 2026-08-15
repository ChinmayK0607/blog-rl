# Qwen3-1.7B four-policy RL v1

**Verdict: mechanically validated, not admitted as an improved RL or communication checkpoint.**

This bundle is the compact, reproducible record of a four-policy LoRA run from
the pinned Swarm Arena SFT warm start. It is published for inspection, not as a
claimed swarm-capable model.

## What did work

- The exact Prime actor and trainer were freshly recertified on four balanced
  groups (64 samples; 3,954 completion tokens). The full numerical parity and
  adapter-isolation gate passed.
- Three full logical updates completed. Each had four separately optimized BLUE
  policies, shared terminal team reward, 16 independent joint-trajectory
  replicas, nonzero advantages, and four distinct adapter changes.
- The mean absolute advantage was `0.0713141`, mean return was `0.0361149`, and
  the non-arena regression gates passed. The collapse audit found neither
  always/never-speaking nor action/target collapse.

## What did not work

- The fourth proposed update failed the fixed pre-optimizer parity gate
  (`p99_probability_error=0.062207937 > 0.05`; tail fraction
  `0.01183432 > 0.005`). It was rejected before optimization; no failed adapter
  was exported or selected.
- On the development-only arena suite, the candidate was not better than the
  SFT start on ordinary cases (mean paired return difference `-0.00610169`).
- All critical generated-vs-dropped, shuffled, delayed, and zero-budget message
  contrasts were exactly zero. This is not evidence of learned communication.
- The horizon-2 sender-credit audit had zero intended sender effects. A larger,
  predeclared horizon-4 audit did show more message-sensitive receiver actions
  in critical than decoy cases (`10` vs `2`), but missed its sign-consistency,
  localization, and off-role gates. The credit estimator is therefore rejected
  for per-agent message training at this scale.

## Claim boundary

The implementation establishes a fail-closed path for independently optimized
agent policies with a shared terminal team return. It does **not** establish
return improvement, causal communication use, adversarial robustness, or
general swarm intelligence. The frozen OOD final suite remains unopened.

The next justified step is environment/estimator redesign: increase the rate of
private facts that another agent can uniquely exploit, then recertify the
message-credit signal before spending more RL compute.

## Files

- `prime_parity_certificate.json`: exact actor/trainer numerical certificate.
- `training_summary.json`: only the three valid updates.
- `step4_rejected_parity_failure.json`: excluded failed fourth batch.
- `development_summary.json`: development-only model-pool and intervention
  results.
- `step3_policy_kl.json`, `regression_summary.json`, and `collapse_audit.json`:
  stability safeguards.
- `message_credit_horizon2_summary.json` and
  `message_credit_horizon4_stage_b_summary.json`: predeclared credit audits.
