# Staged 1.7B RL run

This is the next long-run recipe. It changes the **training schedule**, not the
reward or frozen evaluation. All four BLUE roles retain separate LoRA adapters,
optimizers, private contexts, and token routes. Each role receives the same
exact terminal team return for its own trajectory; there is no message bonus,
judge reward, action shaping, or supervised broadcast target.

## Curriculum

The immutable specification is
`data/rl_v4/staged_curriculum_v1.json`. It declares 120 logical updates with
four complete games per update:

| Stage | Updates | Aggregate ordinary / critical / decoy | Ordinary difficulty |
|---|---:|---:|---:|
| tactical stabilization | 20 | 50 / 25 / 25 | 12–13 nodes, 4–5 turns |
| communication introduction | 40 | 40 / 30 / 30 | 12–13 nodes, 4–5 turns |
| communication heavy | 40 | 30 / 35 / 35 | 12–13 nodes, 4–5 turns |
| adaptive self-play | 20 | 30 / 35 / 35 | 12–13 nodes, 4–5 turns |

Every individual update contains equal numbers of critical and matched-decoy
games. A communication-only update is exactly two critical plus the two matched
decoys, never four unpaired critical games. Across the run this produces 176
ordinary games, 152 critical games, and 152 matched decoys using handoff pair
indices 0–151 from the existing oracle-certified v4 training manifest.

The opponent schedule remains an exact per-update rotation through base, SFT,
historical, and current-policy opponents. This prevents an update from being
identified with one opponent family and keeps the current-policy opponent
adaptive without letting RED share BLUE's private state.

Training difficulty increases through communication density and the adaptive
opponent—not longer episodes. Larger 18/20-node, 8/10-turn games remain in the
development/final evaluations. Keeping training at 12/13 nodes and 4/5 turns
removes an unnecessary sequence-length and runtime risk from the long run.

## Optimization choice

`configs/rl_v4_1_7b_staged.toml` uses LoRA rank 16 and a constant learning rate
of `7.5e-6`. The previous `5e-6` run was stable but weak; `1e-5` learned tactical
return quickly, while the abrupt communication-heavy variant failed its
predeclared mismatch-KL bound. Prime's current live controller does not change
learning rate at stage boundaries, so a single intermediate rate is more
honest than silently restarting optimizers and describing the result as one
continuous run.

The run starts from the common public SFT adapter, not the four role-specific
step-8 development adapters. The current initializer interface accepts one
common adapter. Adding an unaudited per-role resume path would confound this
curriculum test and could mix optimizer state.

The online trainer/serving parity gate is deliberately minimal. Exact token
spans, legal-choice rows, adapter identity, finite values, and four-policy
optimizer isolation are structural checks. Mean mismatch-KL (`0.002`) is the
primary online distribution-drift bound, with mean absolute log-probability
error (`0.05`) as a broad catastrophic sanity check. P99, maximum, probability
tail, and per-token mismatch statistics remain logged diagnostics rather than
abort conditions; DPPO masking handles isolated importance-ratio outliers.
Learning quality is decided by the paired ten-update evaluation, not by a
proliferation of serving/trainer numerical thresholds.

## Paid-run order and fail-closed preflight

Do not launch the 120-update controller from an old plan or parity report. The
required order on the exact four-GPU host is:

1. check out a clean, public, pinned source commit and run the full Linux test
   suite;
2. create a fresh run directory with `prepare_live_rl_run.py --policy-steps
   120 --checkpoint-interval 10`;
3. start three independent rollout servers on GPUs 1--3 with unique vLLM,
   Triton, TorchInductor, RPC and API ports;
4. capture a fresh 32-decision constrained probe across all three servers;
5. run `certify_prime_parity.py` on GPU 0 with the resolved trainer TOML;
6. run the live broadcast/action probe against all three servers;
7. bind those results into one runtime certificate, then build the production
   plan from that certificate;
8. run `preflight_staged_rl.py`; only a v2 `PREFLIGHT.json` with `status =
   passed` authorizes the launcher.

The runtime certificate binds the source commit, base revision, adapter bytes,
resolved trainer TOML, inference TOML, vLLM version, GPU names/VRAM/driver,
three-server structured probe, and passing numerical-parity/four-policy-
isolation report. A GPU, driver, config, source, adapter, trainer, or vLLM
change therefore requires recertification. The preflight also rejects a dirty
tree, missing public inputs, an existing progress file, insufficient disk,
wrong checkpoint retention, the wrong opponent rotation, unavailable servers,
or a non-idle trainer GPU.

The two runtime probes are deliberately bounded. The standard profile uses 32
sequential decisions. After a production-only numerical tail escapes that
probe, the parity-stable profile uses 128 predetermined decisions with four
concurrent requests per server and the preflight requires those exact capture settings. Both
are insurance against wasting a roughly half-day training run; neither is an
evaluation result.

## Build the immutable runtime plan

The checked-in curriculum contains no machine paths or opponent checkpoint
locations. After creating the runtime certificate, bind it to the verified
four-opponent production plan:

```bash
uv run --with ./experiments/swarm_arena \
  python experiments/swarm_arena/scripts/build_staged_rl_plan.py \
  --base-plan /workspace/run-inputs/verified-v4-production-plan.json \
  --curriculum experiments/swarm_arena/data/rl_v4/staged_curriculum_v1.json \
  --handoff-manifest experiments/swarm_arena/data/rl_v4/handoff_train.json \
  --runtime-certificate /workspace/run-inputs/staged-120-runtime-certificate.json \
  --admission-limits experiments/swarm_arena/configs/async_admission_minimal_v1.json \
  --output /workspace/run-inputs/staged-120-production-plan.json
```

The builder reloads the result through the production-plan validator and emits
an audit containing the exact per-stage counts, maximum handoff pair index,
unique ordinary seeds, runtime-certificate hash, plan hash, and full schedule
hash. The controller rejects any `--steps` value other than 120 for this plan.

The complete certificate sequence, after the run directory and servers exist,
is:

```bash
uv run python experiments/swarm_arena/scripts/capture_runtime_parity_probe.py \
  --tokenizer "$SWARM_MODEL" \
  --adapter "$SWARM_INITIAL_ADAPTER" \
  --adapter-sha256 "$SWARM_ADAPTER_SHA256" \
  --base-url http://127.0.0.1:8001 \
  --base-url http://127.0.0.1:8002 \
  --base-url http://127.0.0.1:8003 \
  --samples 32 \
  --output /workspace/run-inputs/staged-120-parity-probe.json

CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nproc-per-node=1 \
  experiments/swarm_arena/scripts/certify_prime_parity.py \
  --model "$SWARM_MODEL" \
  --adapter "$SWARM_INITIAL_ADAPTER" \
  --adapter-sha256 "$SWARM_ADAPTER_SHA256" \
  --trainer-config "$SWARM_RUN_DIR/trainer.toml" \
  --probe /workspace/run-inputs/staged-120-parity-probe.json \
  --output-dir /workspace/run-inputs/staged-120-parity-work \
  --report /workspace/run-inputs/staged-120-parity-report.json

uv run python experiments/swarm_arena/scripts/probe_live_rollout.py \
  --tokenizer "$SWARM_MODEL" \
  --adapter "$SWARM_INITIAL_ADAPTER" \
  --base-url http://127.0.0.1:8001 \
  --base-url http://127.0.0.1:8002 \
  --base-url http://127.0.0.1:8003 \
  > /workspace/run-inputs/staged-120-serving-probe.json

uv run python experiments/swarm_arena/scripts/bind_runtime_certificate.py \
  --repo-root "$SWARM_REPO_ROOT" \
  --source-commit "$(git rev-parse HEAD)" \
  --base-revision "$SWARM_BASE_REVISION" \
  --adapter "$SWARM_INITIAL_ADAPTER" \
  --trainer-config "$SWARM_RUN_DIR/trainer.toml" \
  --inference-config experiments/swarm_arena/configs/inference_1_7b_l40s.toml \
  --serving-probe /workspace/run-inputs/staged-120-serving-probe.json \
  --parity-probe /workspace/run-inputs/staged-120-parity-probe.json \
  --parity-report /workspace/run-inputs/staged-120-parity-report.json \
  --trainer-gpu-id 0 \
  --inference-gpu-id 1 \
  --inference-gpu-id 2 \
  --inference-gpu-id 3 \
  --output /workspace/run-inputs/staged-120-runtime-certificate.json
```

For an eight-GPU V14.5 profile, use trainer IDs 0 and 1, inference IDs
2 through 7, and six distinct rollout URLs consistently in the serving probe,
parity probe, certificate, preflight, pulse evaluator, and controller. Run the
trainer-side parity command and the staged trainer with
`CUDA_VISIBLE_DEVICES=0,1 --nproc-per-node=2`. The runtime certificate binds
the complete partition; a different split requires a fresh certificate.

For the V14.6 parity-stable profile, use
`configs/inference_4b_l40s_parity_strict.toml` and capture the parity probe with
`--samples 128 --concurrency 4`. The certificate records the sample count and
per-server concurrency, and staged
preflight rejects a strict serving certificate produced with any other capture
shape.

Set `SWARM_RUNTIME_CERTIFICATE` to that final file before invoking
`scripts/launch_staged_rl.sh`. The launcher runs the full preflight itself and
will not start the trainer/controller if it fails.

## Evaluation and W&B

Trainer internals are written to an offline W&B run on the GPU host and synced
after training. This prevents a W&B outage from killing an optimizer process.
A separate online sidecar logs controller-level terminal return, advantage
density, curriculum fractions, stage name, and return by scenario/opponent. It
also ingests compact evaluation summaries. The sidecar is intentionally a
separate W&B run in the same group: a W&B/network failure cannot terminate or
alter optimizer input.

```bash
uv run --with wandb \
  python experiments/swarm_arena/scripts/log_live_rl_wandb.py \
  --progress /workspace/runs/staged-120/live_rl_progress.json \
  --eval-root /workspace/results/staged-120-evals \
  --expected-updates 120 \
  --finish-marker /workspace/results/staged-120-evals/COMPLETE \
  --project swarm-arena-rl \
  --group qwen3-1.7b-staged-120 \
  --run-name qwen3-1.7b-staged-120-controller \
  --run-id qwen3-1.7b-staged-120-controller-v1 \
  --tag 1.7b --tag causal-communication --tag development \
  --compact-artifact /workspace/run-inputs/staged-120-production-plan.json \
  --compact-artifact experiments/swarm_arena/data/rl_v4/staged_curriculum_v1.json
```

At updates 0, 10, ..., 120, use the 16-game, BLUE-side-only `pulse` development tier with
`--rl-specific-communication`. The added primary diagnostics are:

1. ordinary candidate-RL minus SFT return;
2. candidate critical normal minus dropped-message return;
3. **RL-specific communication lift**: the critical message effect for RL minus
   the same effect for SFT;
4. **critical-minus-decoy specificity**: candidate message effect on critical
   cases minus its effect on matched decoys.

The pulse is a learning-curve diagnostic. It cannot establish a
paper claim or select repeatedly against the frozen final. After update 120,
run the larger `online` tier first, then one development selection, the frozen
non-arena regression suites,
policy KL, and collapse audit; select at most one candidate before any final
evaluation.

The controller stops at a content-hashed barrier before update 0 and after
every tenth update. The pulse process must finish and write a matching
continuation before training can proceed. This guarantees that each pulse sees
the exact retained checkpoint rather than an adapter name that was already
overwritten. At update 0, SFT-vs-SFT capability and RL-specific communication
differences must be exactly zero; failure blocks the first optimizer step.

Every tenth checkpoint is permanently retained. Only the last two intervening
checkpoints are kept. Trainer W&B runs offline; after successful completion,
sync its local W&B directory explicitly, then upload only compact summaries
and selected adapters. Network logging cannot stop training.

On the previously measured 4x L40S path, 30 updates took about three hours.
With the short-horizon curriculum, 120 updates plus thirteen 16-game pulses is
budgeted at roughly **12--14 hours**, subject to a fresh five-update throughput
measurement. Treat a materially slower first five updates as a stop-and-debug
condition rather than allowing an unattended cost overrun.

## Storage and publication

The GPU host holds transient caches and checkpoints. W&B receives scalar curves
and compact JSON summaries. Git receives source, plans, hashes, audits, and the
research log. Only explicitly retained adapters are pushed to a public
Hugging Face repository; no checkpoint or raw rollout corpus is copied to the
Mac. Credentials are supplied only through environment variables and never
written into plans, logs, commands committed to Git, or W&B config.
