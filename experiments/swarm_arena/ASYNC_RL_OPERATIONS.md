# Fail-closed asynchronous RL operation

No model process can write directly to a trainer queue. Rollout workers write
complete credit-group evidence to an untrusted staging queue. A separate CPU
supervisor independently replays the game, recomputes terminal reward, validates
the immutable run lock, verifies the actual plus four message-drop branches and
sampling keys, checks agent/policy/token ownership, and compares constrained rollout/trainer
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

The live controller is `scripts/run_live_rl.py`. It requests each of the eight
agents independently for broadcast and action phases, records exact prompt and
completion token spans, constructs one actual branch plus one branch that drops
each trainable BLUE sender's first-turn delivered broadcast, then hands the evidence to
the supervisor. All branches retain the same policies and shared random-key
schedule. Only actual BROADCAST spans can enter training. The controller cannot
enqueue samples unless replay, delivery, ownership, reward, and constraint
checks all pass. Multiple independently approved games may be merged only at
the fixed four-run routing boundary.

For the information-handoff curriculum, launch the controller with
`--task-data-version v4 --data-dir experiments/swarm_arena/data/rl_v4` and do
not pass `--horizon` unless the run is an explicitly labelled ablation. The
controller then binds composite hashes for the v4 curriculum/train,
development, and frozen-evaluation inputs in the run lock and uses each
selected scenario's certified horizon. The default
remains v3 for replaying old evidence; a v3 run cannot silently read v4 split
names, and a v4 run cannot silently fall back to the old two-turn horizon.

## Public-artifact launch gate

Paid GPU work must not start from private or authentication-only inputs.
`scripts/prepare_live_rl_run.py` requires exact public base-model and adapter
repository revisions plus a commit-pinned public source URL. It checks them
without a Hugging Face token, anonymously downloads the adapter, and requires
its bytes to match the local pinned SHA-256 before creating a run directory.
The publish scripts explicitly create public repositories and repeat their
verification without credentials. A missing, private, mutable, or mismatched
artifact fails before launch; logging in is not an acceptable substitute.

## Pre-RL calibration contract

Calibration is evaluated over the model's constrained choice distribution, not
the unrestricted vocabulary. A fixed probe uses the production actor backend,
pinned base revision and pinned adapter, then rescores those same tokenized
prompts with the actual Prime FSDP trainer. It checks token IDs, dynamic legal
sets, completion masks, policy ownership and four-run adapter initialization.
A slower exact Prime actor remains a reference implementation, not a required
production rollout path. A real single-policy optimizer step must change only
its assigned `run_*` slot.

Different CUDA kernels do not produce bitwise-identical logits. Admission is
therefore fail-closed on a checked-in numerical envelope: mean and p99 absolute
log-probability drift, max and p99 probability drift, probability-tail
frequency, and mean/max mismatch KL. The raw maximum log-probability error and
importance-ratio error remain diagnostic fields, not sole gates, because an
almost-zero-probability alternative can have a large log-space delta without a
material policy-distribution delta. The certified 1.7B values and thresholds
are stored under `results/pre_rl_1_7b/`. Any change to the model, adapter,
tokenizer, structured constraints, precision path or inference/trainer kernels
invalidates the certificate and requires a rerun.

Named LoRA updates use an explicit unload, load, and `/v1/models` registry-path
verification transaction on every rollout server. Reusing a name with a plain
load call is forbidden: vLLM may return success while retaining the existing
registration, which can silently mix stale policy bytes into counterfactual
branches. Initial trainable-policy and opponent revisions are separate required
controller inputs. A replacement revision is required only when explicitly
running the retained `policy_replacement` audit mode.

Serving constraints are reconstructed with the same installed xgrammar
`choice` grammar and tokenizer vocabulary used by vLLM. A decoded canonical
choice can have multiple valid tokenizations, so a trie built only from
`tokenizer.encode(choice)` is not equivalent to the serving distribution. The
controller records xgrammar's exact allowed-token bitmask at every sampled
token; any token rejected by that grammar or any unterminated completion fails
the group. This constraint implementation has its own protocol version and
therefore invalidates older parity certificates.

## Bounded asynchronous production admission

`swarm_ctf_eval.async_admission` separates two effects that must not be
conflated:

- **backend mismatch:** the fixed no-update calibration measures numerical
  drift between the optimized serving kernels and trainer implementation;
- **policy staleness:** every rollout records its behavior adapter SHA-256,
  immutable revision and update index, while trainer rescoring uses the current
  revision and update index.

Every run must explicitly precommit its maximum policy-update lag, mean and p99
absolute log-ratio limits, symmetric importance-ratio limit, p99 probability
error, and probability-tail limit. These values have no hidden library
defaults; their serialized SHA-256 enters the run lock. The CPU admission
function validates exact constraint rows, constraint and calibration allowlists,
policy rosters, immutable behavior revisions, complete current-policy
log-probabilities, and per-policy lag. A well-formed rollout outside any bound
is discarded whole. It is never clipped, relabelled, or partly routed to an
optimizer. Malformed or incomplete evidence fails the supervisor.

This permits vLLM-class continuous batching and community-optimized attention
kernels while bounding off-policyness. It does not retroactively relax the v1
parity gate or turn its rejected step four into evidence.

The production controller accepts `--production-plan`. The immutable plan
binds trainable phases/turns, an exact ordinary/critical/matched-decoy mixture,
a base/SFT/historical/current model-opponent rotation, the serving backend and
kernel calibration, queue capacity, and every bounded-off-policy limit into
each run lock.

The first optimized-backend pilot intentionally uses lag zero. The separate
filesystem rescore worker verifies that all four current adapter hashes and
update indices still exactly equal their behavior snapshots, then returns the
same backend's recorded token log-probabilities. This is valid only at lag zero;
`run_lag_zero_rescore_worker.py` rejects a stale adapter. Backend/trainer drift
is checked independently by the bound no-update calibration and Prime's
pre-optimizer gate. Lag one or two requires a real current-policy constrained
rescorer; changing only the limit is insufficient.

`AtomicAsyncTrainingQueue` accepts or rejects one complete four-policy group.
It cannot route a subset, reuse a rollout ID, mix trainer steps, or clip a
divergent group. Production co-packing is allowed under the calibrated
off-policy envelope, but every complete trajectory must still fit `seq_len`.

Every supervisor decision is appended to a hash-chained JSONL audit log and
periodically uploaded with immutable checkpoints. The frozen manifests, source
commit, base revision, four adapter revisions, opponent revision, and allowed
dynamic-constraint hashes are bound into a run-lock digest. Resuming with a
different value creates a new run; it cannot append to the old trace.

`run_checkpoint_monitor.py` is the resume-safe promotion sidecar. Its immutable
plan must contain exactly one export, online 96-game evaluation, both regression
suites, policy-KL check, collapse audit, and public checkpoint publication task.
It refuses selection or frozen evaluation during training and records every
argv, return code, and log. These diagnostics can pause promotion; none changes
reward.

## Stop and promotion policy

Infrastructure failures stop optimizer input immediately: incomplete branch
groups, replay/state/reward mismatch, policy or token misrouting, unexpected
constraint hashes, non-finite values, policy lag or log-ratio bounds, and
uncertified backend calibration. Training diagnostics pause checkpoint promotion on speaking,
target, action or KL collapse; gains against only one opponent; regression; or
return gains without message-intervention gains. Diagnostic flags never alter
reward.

Dropping an already-empty message is an invariance check and must reproduce the
exact trajectory and zero advantage. Matched decoy scenarios must also remain
near zero. The retained whole-policy replacement mode is diagnostic only unless
it separately passes localization gates.

Promotion requires the development curriculum gates and both frozen regression
suites. Final evaluation is run once on the selected checkpoint and cannot be
used for checkpoint selection.

## GPU topology

The validation machine uses one RTX 6000 Ada 48 GB: one shared 1.7B backbone,
four trainable LoRA adapters, the frozen SFT opponent adapter, and small rollout
batches. This validates correctness, not throughput.

The full asynchronous run should use four RTX 4090 GPUs:

- GPU 0: Prime-RL multi-run trainer with four LoRA adapter slices and four
  independent optimizers over one frozen backbone;
- GPUs 1–3: rollout workers serving base, SFT, trainable and historical
  adapters with continuous batching;
- CPU supervisor: replay, credit-group validation, hash-chain audit and queue
  admission;
- bounded queues apply backpressure; every rollout is pinned to one complete
  behavior-policy snapshot and admitted only within the precommitted lag bound.

Actual and four message-drop branches share deterministic random-key schedules
and can be distributed across rollout workers. A branch beyond the lag bound,
or with a changed frozen opponent, is discarded rather than mixed into
training.

For remote runs, make the inference command the `tmux` pane's `exec` process
and use `tmux pipe-pane` for persistent logs. Do not use a shell pipeline ending
in `tee` as the pane's lifecycle owner: a logger exit can close the pane even
when the server itself was healthy.

Run the complete Linux pytest gate before launching any live `torchrun` on the
host. The repository-level `cleanup_zombies` fixture executes
`pkill -f torchrun`, so running pytest concurrently is destructive even when
the selected tests are otherwise read-only.

When launching multiple independent vLLM processes on one host, give every
process unique `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `data_parallel_rpc_port`, and API port values.
Concurrent first-start compilation against one shared cache can race and leave
another process trying to load a not-yet-created Triton shared object. Before
starting the trainer, run `scripts/probe_live_rollout.py` against every rollout
URL; it loads the pinned adapter and must pass both an exact broadcast choice
and an exact legal-action choice through the production structured client.
