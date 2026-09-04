# V14.7 A6000 execution plan

Status: CPU validation only. Rental requires explicit approval of the exact
provider node, configuration, and hourly rate.

## Purpose

V14.7 rebinds the CPU-validated V14.6 parity-stable run to one four-GPU
NVIDIA RTX A6000 host. It exists only to use an available 48 GB Ampere node
without pretending that the prior L40/L40S hardware declaration covered it.

The initializer, predetermined curriculum, reward, optimizer, scheduler,
learning rate, loss, DPPO masks, training dtypes, numerical thresholds,
quarantine semantics, 192-row update-zero evaluation, and frozen behavioral
gates at updates 10/20/30/40 are unchanged.

## Fixed topology and admission

GPU 0 is the trainer. GPUs 1, 2, and 3 each host one isolated rollout server.
Every GPU is assigned exactly once. The node must report exactly four NVIDIA
RTX A6000 GPUs with at least 48 GB VRAM each and at least four exposed ports.

The V14.6 correctness-first inference settings remain byte-equivalent after
TOML parsing: eager execution, synchronous scheduling, vLLM's neutral
generation config, native RMSNorm kernels, and at most four concurrent
sequences. The exact host must pass 128 predetermined parity decisions at four
concurrent requests per server, pooled and policy-local numerical gates, and
the unchanged 192-row update-zero evaluation before optimizer update 1.

## Spend and stop contract

The rental ceiling is $60 with a nine-hour TTL. Recovery, idle, terminal,
budget, and exact-pod teardown supervisors must be active before GPU-heavy
setup. A failed admission gate, second parity failure within a ten-slot stage,
failed behavioral gate, terminal fault, budget/deadline trigger, or verified
completion requires compact artifact synchronization and immediate deletion of
only the exact resolved pod.

No gate may be loosened, no batch may be resampled or replaced, and frozen
held-out data remains unopened.
