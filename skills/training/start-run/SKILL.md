---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, and `inference` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbConfig | None`): bare `--wandb` enables defaults; `--wandb @ wandb.toml` enables from a file; `--no-wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Validation aliases let renamed fields keep working; legacy keys can be remapped in a `model_validator(mode="before")`.
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/reverse_text/rl.toml
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml   # SLURM
uv run rl @ examples/reverse_text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Experiment-local Python packages are not installed by the root project. For
  launchers under `experiments/<name>/`, invoke script files by their absolute
  repository path and prepend the experiment directory to `PYTHONPATH`, or run
  from that experiment's own `--project`. Do not rely on `python -m scripts.*`
  from the repository root: the root `scripts/` namespace can shadow the
  experiment-local one. Do not add PEFT with an unpinned `uv run --with peft`
  overlay: it can re-resolve Torch and break binary compatibility with the
  already-validated CUDA stack. The Swarm Arena scoring launchers require
  `SWARM_EVAL_RUNTIME`, an isolated `pip --target` directory containing only
  `peft==0.19.1` and `accelerate==1.13.0`, both installed with `--no-deps` and
  prepended to `PYTHONPATH`. This leaves the Prime-RL environment unchanged.
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('rlm_swe'))"`).
  If a local env exists under `deps/research-environments/environments/` but does not
  import, add it to the root `pyproject.toml` env extra, workspace members, and
  `[tool.uv.sources]`, then run `uv sync --all-extras`.
- Swarm Arena uses Prime-RL multi-run LoRA: create four fixed `run_*`
  directories so `MultiRunManager`, `MultiLoRAOptimizer`, and the packer assign
  one adapter slice and optimizer route to each agent policy. Before any
  optimizer step, run the RL-v3 global audit, build one actual plus four complete
  replacement branches, route each approved actual-agent sample only to its
  fixed run, and require constrained rollout/trainer log-prob parity. Never send
  all four agents through one run or assign one rollout-level advantage to all
  four.
- Seed all four Swarm Arena policy slots from one immutable warm start with
  trainer-side `model.lora.initial_adapter_path` and
  `model.lora.initial_adapter_sha256`. The trainer reads PEFT safetensors only,
  checks the digest before and after loading, requires an exact tensor-key and
  shape match, and initializes every new run before its optimizer is created.
  A resumed native run checkpoint takes precedence over this common seed.
- Swarm Arena rollout workers never write to the trainer queue. Route complete
  actual-plus-four-replacement evidence through `safety_supervisor.py`; require
  independent state replay, terminal-reward recomputation, exact private-context
  hashes, immutable run-lock revisions, constraint allowlisting, policy routing,
  and log-prob parity. Append approvals/rejections to the hash-chained trace.
  Any mismatch fails closed and blocks optimizer input rather than becoming a
  reward penalty.
- Certify constrained-policy parity on the exact serving and trainer stacks
  before admitting a run. Require identical token IDs, allowed-token masks and
  policy routing, then gate the unavoidable vLLM/FSDP kernel drift with the
  checked-in probability, tail and mismatch-KL thresholds. Record raw maximum
  log-probability error for diagnosis, but do not use it alone: large log-space
  errors on negligible-probability tokens can coexist with distributional
  agreement. Re-run the certificate after any model, adapter, tokenizer,
  precision, constraint or serving-kernel change.
- Launch long-lived inference inside `tmux` with the command as the pane's
  `exec` target and attach logging with `tmux pipe-pane`. Avoid making a shell
  pipeline ending in `tee` the pane's main process; a logging-process exit can
  otherwise tear down a healthy server and look like a model failure.

## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/reverse_text/sft.toml
uv run sft @ examples/reverse_text/sft.toml --slurm
uv run sft @ examples/reverse_text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- Entrypoint: `src/prime_rl/entrypoints/sft.py`
- SLURM: single- and multi-node

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly.

```bash
uv run inference @ configs/debug/infer.toml
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Smoke checks:

```bash
curl http://<host>:<port>/health
curl http://<host>:<port>/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- Config: `InferenceConfig` (`packages/prime-rl-configs/src/prime_rl/configs/inference.py`)
- Entrypoint: `src/prime_rl/entrypoints/inference.py`
- SLURM: single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |

## Key paths

- `src/prime_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator` for direct launches)
- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse_text/`)
