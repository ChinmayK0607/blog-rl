# V10 clean held-out evaluation

This is the once-only evaluation of the development-selected update-40 v10
checkpoint. It compares the four independent RL adapters with the exact SFT
initializer on capability, ordinary communication interventions, and the
receiver-only semantic target-swap intervention.

## Why this is a clean remainder rather than the original full frozen suite

The v8 exploratory screen previously opened frozen handoff source pairs 4 and
17, hard cases `ordinary-hard-003` and `ordinary-hard-018`, and legacy seed
`3000003`. The v10 lock excludes those units completely, including every option
ordering of the exposed legacy seed. No result from those units may enter a v10
confirmatory endpoint.

The remaining fixed matrix contains:

- 22 two-world handoff bundles;
- 22 hard ordinary maps;
- 23 legacy seeds under three option orders;
- base, SFT, and historical-league opponent labels;
- both focal sides;
- 4,260 deterministic games in total.

The lock is
`data/rl_v4/v10_clean_holdout_lock.json`, canonical body SHA-256
`1a5bb75f165cbf320e9f9761064d2baa9d000533c537ad367e45a7518a9ffb32`.
Do not edit it after observing any held-out outcome.

## Statistical interpretation

The checkpoint was selected once from the v10 development curve before this
lock. Frozen results cannot select another checkpoint.

The target-swap return estimand is intention-to-treat. If the generated sender
message omits the active fact, the intervention is recorded as ineligible and
contributes zero intervention strength; it is not retried or discarded. This
keeps sender quality inside the end-to-end measurement and avoids conditioning
the held-out result on successful communication.

Primary semantic endpoints are:

1. candidate critical normal minus receiver-only target-swapped return;
2. that effect minus the SFT initializer's effect;
3. critical effect minus matched-decoy effect;
4. normal minus swapped receiver target-action rate;
5. per-opponent effects and target-swap eligibility.

The existing v4 capability, dropped, shuffled, delayed, zero-budget,
matched-decoy, protocol, regression, and collapse endpoints remain mandatory.

## GPU request

Use one 4xL40S node. All four GPUs serve inference; no trainer is needed. Based
on the measured v8/v10 evaluator rate, budget 9 hours for 4,260 rows plus up to
one hour for setup and final publication. Set provider auto-termination to 11
hours. At `$2/hour`, the expected cost is about `$18–22`.

## Host preparation

Pull the public experiment branch and run the artifact preparer on the GPU
host, never on the developer Mac:

```bash
cd /workspace/blog-rl
git switch exp/swarm-arena-4b
git pull --ff-only
export PYTHONPATH=/workspace/blog-rl/experiments/swarm_arena
export UV_CACHE_DIR=/workspace/.uv-cache

/workspace/uv-bootstrap/bin/uv run python \
  experiments/swarm_arena/scripts/prepare_v10_clean_holdout_artifacts.py \
  --lock experiments/swarm_arena/data/rl_v4/v10_clean_holdout_lock.json \
  --workspace /workspace
```

The preparer downloads only public artifacts and verifies the SFT plus all four
candidate adapter SHA-256 values. Its final status is
`/workspace/run-inputs/v10-clean-holdout-artifacts.json`.

## Serving

Start four isolated Prime-RL inference servers using
`configs/inference_4b_l40s.toml`, ports 8000–8003, one visible GPU each, and
separate TorchInductor/Triton/vLLM caches. Always use `uv run inference`; do not
replace it with `vllm serve`.

On server `i`, load:

- `v10-u40-blue-i` from
  `/workspace/artifacts/swarm-arena-live-runs/runs/rl-v10-receiver-isolated4b60-d25505dc/checkpoints/step-40/policy-blue-i`;
- `sft-opponent` from `/workspace/artifacts/qwen3-4b-sft-v2-d1a55d55`;
- `historical-opponent` from the same frozen adapter path, matching the
  preregistered v10 opponent label.

Use `POST /v1/load_lora_adapter` with `lora_name` and `lora_path`, then verify
all aliases through `GET /v1/models` and one structured completion per alias.

## Fail-closed CPU audit

Before the first model request:

```bash
/workspace/uv-bootstrap/bin/uv run python \
  experiments/swarm_arena/scripts/run_v10_clean_holdout.py \
  --config experiments/swarm_arena/configs/v10_clean_holdout_4b.json \
  --lock experiments/swarm_arena/data/rl_v4/v10_clean_holdout_lock.json \
  --data-dir experiments/swarm_arena/data/rl_v4 \
  --output-dir /workspace/runs/v10-clean-holdout-u40 \
  --confirmation 1a5bb75f165cbf320e9f9761064d2baa9d000533c537ad367e45a7518a9ffb32 \
  --audit-only
```

Expected audit values are 22 clean handoff units, 22 hard units, 23 legacy
units, and 4,260 rows. Any mismatch blocks execution.

## Evaluation and off-node preservation

Launch the evaluator and mirror in separate tmux windows. The evaluator is
resumable and refuses a mismatched output manifest.

```bash
/workspace/uv-bootstrap/bin/uv run python \
  experiments/swarm_arena/scripts/run_v10_clean_holdout.py \
  --config experiments/swarm_arena/configs/v10_clean_holdout_4b.json \
  --lock experiments/swarm_arena/data/rl_v4/v10_clean_holdout_lock.json \
  --data-dir experiments/swarm_arena/data/rl_v4 \
  --output-dir /workspace/runs/v10-clean-holdout-u40 \
  --confirmation 1a5bb75f165cbf320e9f9761064d2baa9d000533c537ad367e45a7518a9ffb32 \
  --resume
```

```bash
/workspace/uv-bootstrap/bin/uv run python \
  experiments/swarm_arena/scripts/run_v10_holdout_mirror.py \
  --output-dir /workspace/runs/v10-clean-holdout-u40 \
  --lock experiments/swarm_arena/data/rl_v4/v10_clean_holdout_lock.json \
  --config experiments/swarm_arena/configs/v10_clean_holdout_4b.json \
  --repo-id CK0607/swarm-arena-live-runs \
  --repo-path runs/rl-v10-clean-holdout-u40 \
  --chunk-rows 100 \
  --interval-seconds 300
```

The mirror uploads compact rows every 100 completed games and immutable gzip
raw shards. Every commit is anonymously downloaded and SHA-256 checked before
local mirror state advances. It uploads no logs, credentials, caches, or model
weights.

## Post-evaluation KL and collapse audit

While all four servers are still running, capture a balanced 32-decision legal
choice probe from the frozen SFT alias:

```bash
/workspace/uv-bootstrap/bin/uv run python \
  experiments/swarm_arena/scripts/probe_constrained_rollout.py \
  --base-url http://127.0.0.1:8000 \
  --model sft-opponent \
  --tokenizer /workspace/models/qwen3-4b-cdbee75f \
  --adapter /workspace/artifacts/qwen3-4b-sft-v2-d1a55d55 \
  --adapter-sha256 168c9f9cdd0537660b664e9863ec9e351faf5e84d85ffbc77e95501fe1d903d2 \
  --output /workspace/runs/v10-clean-holdout-u40/policy_kl_probe.json \
  --samples 32
```

After the 4,260-row evaluation is complete, stop only the GPU-0 inference
server and measure the four selected adapters against the SFT initializer on
that frozen probe. The public update-40 export has an adapter-metadata defect:
its immutable tensors are rank 32 and the frozen trainer used rank 32 / alpha
64, but each exported `adapter_config.json` declares rank 16 / alpha 32. Create
metadata-corrected views first. The utility fails closed unless all four tensor
ranks and target modules agree with the frozen trainer config; it symlinks the
unchanged weight files and records every source/config/weight hash.

```bash
/workspace/uv-bootstrap/bin/uv run \
  experiments/swarm_arena/scripts/repair_v10_kl_adapter_metadata.py \
  --trainer-config experiments/swarm_arena/configs/rl_v10_4b_receiver_isolated_60.toml \
  --adapter blue-0=/workspace/artifacts/swarm-arena-live-runs/runs/rl-v10-receiver-isolated4b60-d25505dc/checkpoints/step-40/policy-blue-0 \
  --adapter blue-1=/workspace/artifacts/swarm-arena-live-runs/runs/rl-v10-receiver-isolated4b60-d25505dc/checkpoints/step-40/policy-blue-1 \
  --adapter blue-2=/workspace/artifacts/swarm-arena-live-runs/runs/rl-v10-receiver-isolated4b60-d25505dc/checkpoints/step-40/policy-blue-2 \
  --adapter blue-3=/workspace/artifacts/swarm-arena-live-runs/runs/rl-v10-receiver-isolated4b60-d25505dc/checkpoints/step-40/policy-blue-3 \
  --output-root /workspace/runs/v10-clean-holdout-u40/repaired_adapters
```

The frozen project environment does not include PEFT because inference is
served by vLLM. Install the exact audit-only packages into an isolated overlay
without resolving or replacing the project's Torch/CUDA stack:

```bash
/workspace/uv-bootstrap/bin/uv pip install \
  --target /workspace/audit-overlay \
  --no-deps \
  peft==0.20.0 accelerate==1.13.0
```

Then run the audit through the frozen project environment plus that overlay:

```bash
PYTHONPATH=/workspace/audit-overlay CUDA_VISIBLE_DEVICES=0 \
/workspace/uv-bootstrap/bin/uv run --no-sync python \
  experiments/swarm_arena/scripts/measure_constrained_policy_kl.py \
  --model /workspace/models/qwen3-4b-cdbee75f \
  --baseline-adapter /workspace/artifacts/qwen3-4b-sft-v2-d1a55d55 \
  --candidate-adapter blue-0=/workspace/runs/v10-clean-holdout-u40/repaired_adapters/blue-0 \
  --candidate-adapter blue-1=/workspace/runs/v10-clean-holdout-u40/repaired_adapters/blue-1 \
  --candidate-adapter blue-2=/workspace/runs/v10-clean-holdout-u40/repaired_adapters/blue-2 \
  --candidate-adapter blue-3=/workspace/runs/v10-clean-holdout-u40/repaired_adapters/blue-3 \
  --probe /workspace/runs/v10-clean-holdout-u40/policy_kl_probe.json \
  --output /workspace/runs/v10-clean-holdout-u40/policy_kl.json
```

Then run the raw-trajectory collapse audit:

```bash
/workspace/uv-bootstrap/bin/uv run python \
  experiments/swarm_arena/scripts/audit_final_eval_collapse.py \
  --eval-dir /workspace/runs/v10-clean-holdout-u40 \
  --policy-kl /workspace/runs/v10-clean-holdout-u40/policy_kl.json \
  --output /workspace/runs/v10-clean-holdout-u40/collapse_audit.json \
  --policy-alias v10-u40-blue-0=blue-0 \
  --policy-alias v10-u40-blue-1=blue-1 \
  --policy-alias v10-u40-blue-2=blue-2 \
  --policy-alias v10-u40-blue-3=blue-3
```

The frozen collapse limits are: speaking rate must stay strictly between .02
and .98; no action or message target may exceed .95 concentration once it has
at least 20 samples; constrained candidate-to-SFT KL must have mean at most .08
and p99 at most .30. These are diagnostics, never rewards.

The mirror waits for and publishes the probe, KL report, and collapse report
before exiting. The final decommission gate is:

- evaluator `COMPLETE` exists;
- `summary.json` exists and has 4,260 rows;
- mirror status is healthy, complete, and has 4,260 mirrored raw rows;
- `policy_kl.json` and a passing `collapse_audit.json` are publicly mirrored;
- final public revision passes anonymous verification.
