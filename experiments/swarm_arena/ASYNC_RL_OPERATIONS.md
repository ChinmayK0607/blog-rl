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

## Public-artifact launch gate

Paid GPU work must not start from private or authentication-only inputs.
`scripts/prepare_live_rl_run.py` requires exact public base-model and adapter
repository revisions plus a commit-pinned public source URL. It checks them
without a Hugging Face token, anonymously downloads the adapter, and requires
its bytes to match the local pinned SHA-256 before creating a run directory.
The publish scripts explicitly create public repositories and repeat their
verification without credentials. A missing, private, mutable, or mismatched
artifact fails before launch; logging in is not an acceptable substitute.

## Pre-RL parity contract

Parity is evaluated over the model's constrained choice distribution, not the
unrestricted vocabulary. The rollout certificate uses the exact vLLM server,
pinned base revision and pinned adapter, then rescores those same tokenized
prompts with the actual Prime FSDP trainer. It checks token IDs, dynamic legal
sets, completion masks, policy ownership and four-run adapter initialization.
A real single-policy optimizer step must change only its assigned `run_*` slot.

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

For live asynchronous rollouts, the supervisor may defer the numerical portion
of parity to the trainer only when the signed approval and immutable run lock
bind the exact trainer parity-gate digest. The trainer gathers the exact
trainable token set across ranks, recomputes all probability-drift and
mismatch-KL metrics, and raises before `optimizer.step()` if any threshold is
exceeded. Admission rejects a missing or mismatched gate digest. This keeps
rollout workers outside the trusted boundary without allowing a failed serving
versus trainer comparison to update weights.

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
- bounded queues apply backpressure, and policy revisions switch only between
  complete update epochs.

Actual and four message-drop branches share deterministic random-key schedules
and can be distributed across rollout workers. A branch from a stale policy or
opponent epoch is discarded rather than mixed into training.

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
