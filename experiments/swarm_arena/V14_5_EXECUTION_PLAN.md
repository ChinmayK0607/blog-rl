# V14.5 reliable execution plan

Status: CPU implementation and validation only. No GPU rental is authorized by
this document.

## Why V14.5 exists

V14.1 and V14.2 spent their runs in zero-update screening. V14.3 applied seven
optimizer updates before a policy-local vLLM/trainer numerical tail exceeded a
newly tightened threshold. V14.4 fixed the certificate mismatch and passed its
192-row update-zero evaluation, but applied only two optimizer updates before
one policy-local logical batch exceeded the unchanged `0.002` mismatch-KL
threshold. None of those outcomes tested the V14 curriculum at its first
ten-update behavioral gate.

V14.5 is therefore an execution repair of V14, not a new curriculum. It starts
again from the four distinct public V13 update-80 policies. It keeps V14.4's
curriculum, reward, optimizer, learning rate, loss, DPPO masking, model dtypes,
serving stack, numerical thresholds, update-zero evaluation, and frozen
update-10/20/30/40 behavioral gates.

## Numerical-parity handling

The numerical gate remains active before every optimizer step. A passing
logical four-policy batch updates normally. When a batch fails:

1. no optimizer or scheduler step is applied to any policy;
2. every accumulated gradient for that atomic batch is discarded;
3. no replacement batch is sampled and no seed is retried;
4. the failure, all policy-local metrics, thresholds, and action are appended
   and fsynced before continuation;
5. the exact unchanged four policy weights are published for the next
   predetermined curriculum slot; and
6. at most one such quarantine is allowed in each ten-update stage. A second
   failure in the same stage aborts the run before an optimizer step.

This does not claim that a quarantined batch trained the model. Public progress
and each stage-gate artifact separately report scheduled logical updates,
optimizer-applied updates, and parity-quarantined updates. A ten-slot stage with
one quarantine is reported as nine optimizer steps, not ten.

The bounded quarantine avoids both failure modes we care about: it does not
apply numerically untrusted gradients, and it does not condition replacement
sampling on a favorable parity outcome. The one-per-stage ceiling prevents a
systematically incompatible serving/trainer stack from silently turning into
an under-trained run.

## Permanent launch repairs

- The staged preflight reads the serialized Prime orchestrator's current
  `student.model.lora` schema. It no longer uses the obsolete top-level
  `model.lora` path.
- Every staged subprocess uses the same `uv run --frozen --extra flash-attn`
  runtime. A generic `uv run` can no longer silently select an environment
  without vLLM.
- The launcher requires the exact public base and adapter repository IDs and
  preflight compares them to `PREPARE.json` before optimizer work.
- The compact public mirror includes the small parity-quarantine ledger but
  continues to exclude local token exports.
- Quarantine counts are restored from the append-only ledger on trainer resume;
  a rejected or inconsistent ledger cannot receive a fresh allowance.

## Spend contract

No pod may be rented until the source is clean, the V14.5 CPU bundle reproduces,
focused tests and lint pass, the exact source and bundle are public and
anonymously verified, and the complete launch environment has been rendered.
On a pod, setup/certification/update-zero remains bounded and update 1 is locked
until mirror and W&B preflights pass. The first spend decision remains the
update-10 behavioral gate. Completion, rejection, or an unrecoverable
operational fault requires final compact sync followed by immediate exact-pod
decommissioning.

## Interpretation boundary

V14.5 can tell us whether the grounded V14 curriculum improves communication
and retention through a complete stage. It cannot turn a failed confidence
interval into a claim, and a quarantine is not evidence of learning. Frozen
held-out data remains unopened until the existing formal selection rule permits
it.
