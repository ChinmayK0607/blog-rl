# V14.3: Policy-routed adaptive curriculum

Status: CPU implementation, audit, tests, and reproducible bundle complete. No
GPU is rented and no V14.3 optimizer update has run.

## Decision

V14.2 showed that one global ordinary-signal gate is the wrong control
structure. Its complete pilot found different regimes for the four policies:

- blue-0: `expand` — retain its observed frontier while testing unseen cases;
- blue-1: `consolidate` — prefer its observed frontier with fixed anchors;
- blue-2: `consolidate` — prefer its observed frontier with fixed anchors;
- blue-3: `discover` — rotate unseen cases, then exploit a bounded fraction
  only after complete training-stage evidence finds a frontier.

A weak policy no longer vetoes fresh, useful training for the others. Every
scheduled policy slot still runs exactly once. A flat realization contributes
zero gradient naturally and is recorded as telemetry; it is never resampled
until favorable.

## Routing contract

The pool contains 128 training-only cases, eight per policy/opponent cell. The
24 pilot-observed identities are classified from the complete V14.2 result:
eight frontier, six mastered, and ten stalled. The remaining 104 are unseen.
No pilot trajectory is retained. V14.3 uses fresh sampling namespaces and
fresh rollouts only.

At each ten-update boundary, the selector analyzes the entire preceding
training stage. It may update case classifications and bind the next stage's
exact case identities. It cannot change the group mix, opponent rotation,
policy balance, reward, credit estimator, counterfactual, stage shape, or stage
gate. Development and frozen outcomes are forbidden selector inputs. Every
selection is atomically written and must reproduce exactly on resume.

`consolidate` routes frontier requests to observed frontier cases before unseen
fallbacks. `expand` assigns half of frontier requests to observed frontier and
half to unseen discovery. `discover` initially assigns all frontier requests
to unseen cases; after at least one frontier is observed, it reserves 25% for
that frontier and keeps exploring the remainder. Mastered and stalled anchors
retain the existing fixed fractions.

## Compute contract

The existing four-stage, 40-update maximum and update-10/20/30/40 behavioral
gates remain unchanged. Zero-advantage logical updates are telemetry and do not
halt or trigger resampling. Before renting, publish and anonymously verify the
exact source and confirm HF/W&B credentials. On the paid host, only the runtime
certificate, mirror preflight, update-0 baseline, and operational watcher are
required before update 1; there is no additional broad ordinary screen.

The target remains 4xL40S at an assumed `$1.52/hour`, with a `$15` hard cap and
nine-hour TTL. Stop and immediately decommission the exact pod after the first
failed stage gate, hard budget/TTL, unrecoverable operational fault, or verified
completion and artifact sync.

## Frozen identities

- V14.2 assessment body:
  `ba8abe53e1d973600f6c9c9c1690354cb9009d122586883e265fc98857bc3087`
- Ordinary pool body:
  `4d39e513aeb5a83b1cd836f3ea1a6e591fa8a312ffa868131aed429016b44999`
- Curriculum body:
  `197e50e9253798f75a83d29e306c321618c7b9633d3b86b4133e54bb0bb8e0e5`
- Finalization audit body:
  `ecffdd096e0e0aa2caec524931e3070bcc3c869b16e8f2370cffb96bf812abcd`
- CPU bundle body:
  `1dbf1e12a86575ce174301411640a0dc98cb0b151411c00fd777afed0390c37e`
- Unchanged stage-gate body:
  `27098650c9e6f604e8393a75fc01cb0a0e6c694cf80286ba7d8965de84a1f8c2`
