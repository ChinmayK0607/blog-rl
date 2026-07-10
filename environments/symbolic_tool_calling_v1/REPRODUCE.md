# Reproducing the blog: GRPO vs PPO under rollout compaction

End-to-end steps to reproduce the experiments behind the writeup (`blog.md`). Everything runs from the
Prime-RL repo root on an 8×GPU node. Base model: `Qwen/Qwen3-4B-Instruct-2507`. Experiment tracking:
W&B project `blog-rl` (see `table.md` for the full run ledger).

All the pieces added for this study:

| Area | Path |
|---|---|
| Environment | `environments/symbolic_tool_calling_v1/` |
| Compaction objectives | `src/prime_rl/orchestrator/algo/compacted_grpo.py`, `.../algo/routing.py` |
| PPO value head (+ warm-start) | `src/prime_rl/trainer/model.py`, `.../models/layers/lm_head.py`, `.../trainer/rl/train.py` |
| Configs | `environments/symbolic_tool_calling_v1/rl_qwen3_instruct_*.toml`, `eval_*.toml` |
| Taskset generation | `scripts/generate_symbolic_taskset.py` |
| Offline analysis | `scripts/analyze_symbolic_compaction.py`, `scripts/analyze_gradient_snr.py` |
| Value-head pretraining | `scripts/pretrain_ppo_value_head.py` |
| Supervisors / eval | `scripts/supervise_symbolic_*`, `scripts/run_symbolic_hard_eval_checkpoints.sh` |
| Checkpoint hygiene | `scripts/prune_symbolic_checkpoints.py`, `scripts/finalize_hf_checkpoint.py` |

## 0. Setup

```bash
git clone git@github.com:ChinmayK0607/blog-rl.git && cd blog-rl   # branch: synth-env
export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local
uv sync
uv run --no-sync pytest environments/symbolic_tool_calling_v1/tests -q   # deterministic env checks
export WANDB_PROJECT=blog-rl          # or WANDB_MODE=offline to skip logging
```

## 1. Tasksets

The generator is deterministic (task identity = hash of the normalized spec), so a given seed reproduces
the exact set. Key knobs: `--horizons` (depth), `--distractor-ratio`, `--recovery-cost`, `--verbosity`,
`--imbalance`.

**Hard curriculum (Phase B)** — single command:

```bash
uv run --no-sync python scripts/generate_symbolic_taskset.py \
  artifacts/symbolic-hard-curriculum-v1 \
  --num-tasks 320 --val-tasks 40 --seed 72000 --horizons long,xlong \
  --branching-factor 2 --distractor-ratio 0.4 --recovery-cost 3 \
  --verbosity low --imbalance high
```

**Medium curriculum (Phase A, `symbolic-curriculum-v1`)** — built by selecting *pass@4-mixed* tasks
(solved on 1–3 of 4 rollouts by the frozen model) from short(depth 3)+medium(depth 5) generations, so
every prompt group keeps learning signal. Regenerate the candidate pools with the generator (e.g.
`--horizons short,medium --distractor-ratio 0.4 --verbosity low`) then keep the mixed groups from a
frozen-model pass@4 sweep (see `eval_qwen3_instruct_*_pass4.toml` + `symbolic-pass-at-k`).

**Graded hard eval set** — `eval_graded_hard_{medium,long,xlong}_pass4.toml`.

## 2. Phase A — four regimes on the medium curriculum

`token_budget=384` for the compacted arms; all on `symbolic-curriculum-v1`.

```bash
bash scripts/supervise_symbolic_compaction_compare.sh
```

Runs, one 8-GPU job at a time (config → W&B run):

| Objective | Config | W&B run |
|---|---|---|
| Full-rollout GRPO (baseline) | `rl_qwen3_instruct_cmp_full_grpo.toml` | `cmp-full-grpo-v1` |
| Compacted GRPO | `rl_qwen3_instruct_cmp_compacted_grpo.toml` | `cmp-compacted-grpo-v1` |
| Segment-normalized GRPO | `rl_qwen3_instruct_cmp_segnorm_grpo.toml` | `cmp-segnorm-grpo-v1` |
| Compacted PPO | `rl_qwen3_instruct_cmp_compacted_ppo.toml` | `cmp-compacted-ppo-v1` |

## 3. Offline analysis (no training)

Over the collected rollouts of any run (`<run>/run_default/rollouts`):

```bash
# segment-count / compaction-imbalance distribution
uv run --no-sync python scripts/analyze_symbolic_compaction.py \
  artifacts/cmp-compacted-grpo-v1/run_default/rollouts --token-budget 384

# gradient signal-to-noise per objective (the proposal's variance analysis)
uv run --no-sync python scripts/analyze_gradient_snr.py \
  artifacts/cmp-compacted-grpo-v1/run_default/rollouts --steps 10-70 --token-budget 384
```

## 4. Hard-task pass@k of the Phase-A checkpoints

Serves each regime's checkpoint (hermes tool parser) and runs pass@4 on the depth-graded ladder:

```bash
bash scripts/run_symbolic_hard_eval_checkpoints.sh      # writes artifacts/hard-eval-phaseA-v1/summary.log
```

## 5. Phase B — warm-started hard training + PPO value-head warm-start

All arms warm-start from the **same** full-GRPO policy (`cmp-full-grpo-v1/weights/step_90`) and train on
the hard curriculum at `token_budget=384`. Eval uses `tool_call_parser = "hermes"` (required for
local-path models, which don't auto-resolve the parser).

**Value-head warm-start (kills the PPO critic cold-start).** Under the exact PPO-init policy, regress the
hidden state at each segment-start critic position onto the rollout return (closed-form ridge; backbone
frozen). Load it via `trainer.model.ppo_value_head_init`:

```bash
uv run --no-sync python scripts/pretrain_ppo_value_head.py \
  --model artifacts/cmp-full-grpo-v1/weights/step_90 \
  --rollouts artifacts/hardb-full-grpo-v1/run_default/rollouts \
  --steps 1-25 --token-budget 384 --max-rollouts 400 --max-seq-len 16384 \
  --out artifacts/hardb-ppo-warmcritic/value_head.safetensors
# -> reports fit R^2 (warm critic explains ~0.7 of return variance vs ~0 for zero-init)
```

Run the sequential queue (compacted GRPO → cold PPO → warm-critic PPO), disk-bounded (`keep_last=2` +
per-arm transient cleanup):

```bash
bash scripts/supervise_symbolic_hard_phaseb_rerun.sh
```

| Arm | Config | W&B run |
|---|---|---|
| Full GRPO (reference) | `rl_qwen3_instruct_hardb_full_grpo.toml` | `hardb-full-grpo-v1` |
| Compacted GRPO | `rl_qwen3_instruct_hardb_compacted_grpo.toml` | `hardb-compacted-grpo-v1` |
| Compacted PPO (cold critic) | `rl_qwen3_instruct_hardb_compacted_ppo.toml` | `hardb-compacted-ppo-v1` |
| Compacted PPO (warm critic) | `rl_qwen3_instruct_hardb_compacted_ppo_warmcritic.toml` | `hardb-compacted-ppo-warmcritic-v1` |

## 6. Checkpoint hygiene

Prime-RL keeps bulky DCP trainer checkpoints (`checkpoints/`, optimizer state) and lean HF exports
(`weights/step_N`). For eval/serving keep only the HF export and strip the trainer-only PPO value head:

```bash
# collapse to a single vLLM-loadable model.safetensors per step, dropping value_head.*
uv run --no-sync python scripts/finalize_hf_checkpoint.py --run-root artifacts/<run>
```

The Phase-B supervisor + `scripts/finalize_phaseb_storage.sh` do this automatically on completion (and
drop `checkpoints/` + `broadcasts/`).

## Notes / gotchas

- **Local-path models need `tool_call_parser="hermes"`** in the inference config; the `"auto"` default
  resolves the parser from the model *name*, which only matches hub ids (`Qwen/Qwen3-...`).
- **Disk**: a full 70-step run produces ~180 GB of DCP checkpoints + ~40 GB of weight broadcasts. Use
  `keep_last` and the finalize step, or the disk will fill mid-queue.
- PPO requires `trainer.model.ppo_value_head=true` and `impl in {custom, auto}` (custom Qwen3 only).
