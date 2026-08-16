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
  one adapter slice and optimizer route to each agent policy. The admitted
  candidate samples four independent joint trajectories from one scenario and
  opponent snapshot, computes the scenario-matched leave-one-out terminal-return
  advantage, and shares each trajectory's scalar across its four agents while
  routing every agent's own tokens only to its fixed policy run. Never send all
  four agents through one run or merge their private contexts, gradients,
  optimizer states, or checkpoints.
- Seed all four Swarm Arena policy slots from one immutable warm start with
  trainer-side `model.lora.initial_adapter_path` and
  `model.lora.initial_adapter_sha256`. The trainer reads PEFT safetensors only,
  checks the digest before and after loading, requires an exact tensor-key and
  shape match, and initializes every new run before its optimizer is created.
  A resumed native run checkpoint takes precedence over this common seed.
- Swarm Arena rollout workers never write to the trainer queue. Route complete
  shared-return evidence through `safety_supervisor.py`; require independent
  state replay, terminal-reward recomputation, exact private-context hashes,
  unique sampling namespaces, immutable run-lock revisions, exact sample-content
  commitments, constraint allowlisting, four-policy routing, and log-prob
  parity. Append approvals/rejections to the hash-chained trace. Any mismatch
  fails closed and blocks optimizer input rather than becoming a reward penalty.
- Certify constrained-policy parity on the exact serving and trainer stacks
  before admitting a run. Require identical token IDs, allowed-token masks and
  policy routing, then gate the unavoidable vLLM/FSDP kernel drift with the
  checked-in probability, tail and mismatch-KL thresholds. Record raw maximum
  log-probability error for diagnosis, but do not use it alone: large log-space
  errors on negligible-probability tokens can coexist with distributional
  agreement. Re-run the certificate after any model, adapter, tokenizer,
  precision, constraint or serving-kernel change.
- Swarm Arena serving evidence records the finite top-logprob distribution for
  every constrained token row, excluding only vLLM's `-9999` masked sentinels.
  For rows that fit within the requested top-k, require the server token set to
  exactly equal the independently reconstructed xgrammar set. The parity
  certificate reports full-row normalization error, total variation and both
  KL directions in addition to the frozen sampled-token gates.
- When diagnosing a parity failure, run
  `experiments/swarm_arena/scripts/certify_prime_parity_matrix.py` with
  `configs/parity_matrix_1_7b.json`. It allocates one declared trainer variant
  per visible GPU, uses independent rendezvous endpoints, records every config
  hash and command, and selects the first passing variant in the checked-in
  order. A matrix result is diagnostic: regenerate evidence with the selected
  trainer config bound into the run lock and recertify before RL. Never replace
  vLLM behavior logprobs with trainer logprobs merely to make parity pass.
- A Swarm Arena live rollout can use deferred trainer parity only when its
  signed supervisor approval and immutable run lock contain the SHA-256 of the
  active `rollout_parity_gate`. The router must compare that digest before queue
  admission, and the trainer must gather errors over the exact completion-token
  set and validate every bound threshold before `optimizer.step()`. A missing,
  changed or failed gate is a hard stop, never a diagnostic-only warning.
- Launch long-lived inference inside `tmux` with the command as the pane's
  `exec` target and attach logging with `tmux pipe-pane`. Avoid making a shell
  pipeline ending in `tee` the pane's main process; a logging-process exit can
  otherwise tear down a healthy server and look like a model failure.
- Complete the project pytest gate before starting any live `torchrun`. The
  repository-level `cleanup_zombies` fixture invokes `pkill -f torchrun` at
  module setup, so even a read-only test invocation can terminate unrelated
  trainers or rollout actors owned by the same user. Never run pytest beside a
  live training, certification, or rollout process on the same host.
- For the staged Swarm Arena run, first bind
  `data/rl_v4/staged_curriculum_v1.json` to the verified opponent/runtime plan
  with `scripts/build_staged_rl_plan.py`. The staged plan declares its exact
  update count; never shorten or extend it under the same run identity.
  `scripts/log_live_rl_wandb.py` is a failure-isolated sidecar for controller
  return, curriculum, opponent, and causal-evaluation metrics. Keep it in the
  same W&B group as trainer telemetry, but do not make training health depend
  on W&B availability. Store only compact summaries/artifacts there; publish a
  selected adapter separately to Hugging Face and never copy checkpoints to the
  local Mac.

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
