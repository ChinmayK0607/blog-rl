# V14.3: Policy-routed adaptive curriculum

Status: exact 4xL40S pod allocated and frozen setup complete. No V14.3
optimizer update has run. Runtime certification is intentionally pending the
corrected public launch identity below.

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
  `2741872a8a4d9f632752c56a7f0c58537155812679427ea5c355d5806401ea32`
- Unchanged stage-gate body:
  `27098650c9e6f604e8393a75fc01cb0a0e6c694cf80286ba7d8965de84a1f8c2`

The first published bundle was `1dbf1e12...c37e`. Before any optimizer work,
launch review found that the generic launcher did not forward the curriculum's
already-frozen target-swap sender retry count or the mandatory stage-gate
artifact. The in-progress runtime certificate was preserved and stopped as
stale. The narrow wiring correction adds no new gate or scientific behavior:
it forwards the existing values and uses the established 10,800-second
operational ready-file timeout. The corrected bundle above supersedes the
first identity; certification and training must bind only the corrected public
source and bundle.
