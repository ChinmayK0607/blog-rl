# V14.6 parity-stable execution plan

Status: CPU validation only. No GPU rental is authorized by this document.

## Why V14.6 exists

V14.5 passed the exact-host runtime certificate and the unchanged 192-row
update-zero evaluation. Its first and fourth logical batches nevertheless
exceeded the unchanged `0.002` mean mismatch-KL gate. The first batch was
quarantined with no optimizer or scheduler step; two subsequent batches
produced two durable optimizer updates; the second failure aborted stage zero
before optimization. The exact pod was synced, hash-verified, and removed at a
final provider cost of `$5.36`.

The runtime certificate captured 32 requests sequentially. Production rollout
generation is concurrent, and prior audits showed that vLLM batching can expose
rare numerical tails that a sequential probe misses. Repeating V14.5 unchanged
would therefore resample a rejected execution path rather than repair it.

## Prospective execution repair

V14.6 retains V14.5's four distinct V13 initializers, predetermined curriculum,
reward, optimizer, scheduler, learning rate, loss, DPPO masks, training dtypes,
parity thresholds, quarantine rule, update-zero evaluation, and frozen
update-10/20/30/40 behavioral gates.

Only the serving and runtime-certification path changes:

- vLLM runs eagerly with asynchronous scheduling disabled;
- generation uses vLLM's own neutral generation config;
- RMSNorm and fused add-RMSNorm use the native implementation;
- each server admits at most four concurrent sequences; and
- exact-host parity certification captures 128 predetermined decisions with
  four concurrent requests per server, retaining 32 samples per policy and
  exercising every declared rollout server.

The `0.002` mismatch-KL ceiling remains unchanged. A failed certificate blocks
evaluation and optimization. During training, at most one complete atomic
logical batch per ten-slot stage may still be quarantined with no optimizer or
scheduler step and no replacement; a second failure aborts.

## Spend and topology contract

The preferred target is one 4x NVIDIA L40/L40S host using trainer GPU 0 and
three isolated rollout servers on GPUs 1--3. A separately certified 8-GPU host
may use trainer GPUs 0--1 plus six isolated rollout servers on GPUs 2--7. Every
GPU must be assigned exactly once. The total rental ceiling remains `$60` with
a nine-hour TTL.

Before GPU-heavy setup, install the exact-pod recovery, idle, terminal, budget,
and teardown supervisors. Public source and CPU-bundle hashes, credentials,
HF/W&B preflights, runtime topology, strict inference config, parity evidence,
and the unchanged update-zero result must all be bound before update 1.
Rejection, verified completion, terminal fault, or the deadline safety margin
requires compact sync and hash verification followed by immediate deletion of
only the exact resolved pod.

## Interpretation boundary

This is a preventive numerical-execution change, not a scientific rescue.
Frozen held-out data remains unopened. No reward, gate, threshold, schedule,
case, or replacement sample may be changed after observing a result.
