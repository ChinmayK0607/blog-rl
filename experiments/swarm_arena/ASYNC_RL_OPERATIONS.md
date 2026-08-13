# Fail-closed asynchronous RL operation

No model process can write directly to a trainer queue. Rollout workers write
complete credit-group evidence to an untrusted staging queue. A separate CPU
supervisor independently replays the game, recomputes terminal reward, validates
the immutable run lock, verifies all five branches and sampling keys, checks
agent/policy/token ownership, and compares constrained rollout/trainer
log-probabilities. Only signed approval envelopes enter the trainer queue.
Approvals use HMAC-SHA256 with a key mounted only into the supervisor and
trainer-admission process. `prime_multi_run_router.py` verifies the signature,
token ownership and fixed policy-to-`run_*` mapping before writing four separate
Prime-RL `TrainingBatch` objects. In production, rollout and supervisor/trainer
processes must use different Unix users or containers; rollout workers receive
write access only to staging, never the signing key or trainer run directories.

The environment exposes only structured broadcast and enumerated-action JSON.
Agents receive no shell, network, filesystem, code execution, arbitrary tool,
or external-system interface. Invalid or unsupported output fails the group; it
does not become a negative reward that the policy can learn to trade off.

Every supervisor decision is appended to a hash-chained JSONL audit log and
periodically uploaded with immutable checkpoints. The frozen manifests, source
commit, base revision, four adapter revisions, opponent revision, and allowed
dynamic-constraint hashes are bound into a run-lock digest. Resuming with a
different value creates a new run; it cannot append to the old trace.

## Stop and promotion policy

Infrastructure failures stop optimizer input immediately: incomplete branch
groups, replay/state/reward mismatch, policy or token misrouting, unexpected
constraint hashes, non-finite values, stale policy revisions, and log-prob
parity failure. Training diagnostics pause checkpoint promotion on speaking,
target, action or KL collapse; gains against only one opponent; regression; or
return gains without message-intervention gains. Diagnostic flags never alter
reward.

Promotion requires the development curriculum gates and both frozen regression
suites. Final evaluation is run once on the selected checkpoint and cannot be
used for checkpoint selection.

## GPU topology

The validation machine uses one RTX 6000 Ada 48 GB: one shared 1.7B backbone,
four trainable LoRA adapters, the frozen SFT replacement adapter, and small
rollout batches. This validates correctness, not throughput.

The full asynchronous run should use four RTX 4090 GPUs:

- GPU 0: Prime-RL multi-run trainer with four LoRA adapter slices and four
  independent optimizers over one frozen backbone;
- GPUs 1–3: rollout workers serving base, SFT, trainable and historical
  adapters with continuous batching;
- CPU supervisor: replay, credit-group validation, hash-chain audit and queue
  admission;
- bounded queues apply backpressure, and policy revisions switch only between
  complete update epochs.

Actual and four replacement branches share deterministic random-key schedules
and can be distributed across rollout workers. A branch from a stale policy or
opponent epoch is discarded rather than mixed into training.
