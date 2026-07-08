# Symbolic tool-calling benchmark

This native `verifiers.v1` taskset generates deterministic hidden room graphs for long-horizon,
stateful tool use. Task identity is derived from the normalized generated specification, and the
pure state machine can replay an action sequence without an LLM or runtime.

The environment supports configurable horizon, branching, distractors, recovery depth, observation
verbosity, and imbalance. It exposes `inspect`, `move`, `pickup`, `use`, `query`, and `submit`; only
hidden environment state determines terminal success. A valid solution must acquire and use the key,
activate the switch, query the correct terminal, enter the target room, discover the target system,
and submit the recovered code.

## Artifact pipeline

The benchmark is built as three immutable stages:

1. `tasks/`: deterministic, content-addressed task specifications.
2. `rollouts/`: raw full trajectories grouped by stable `prompt_id`.
3. `compaction/`: a deterministic fixed-token transform over the frozen raw rollouts.
4. `training/`: objective-specific credit records for compacted GRPO, segment-normalized GRPO,
   and compacted PPO.

Each stage contains `config.json`, `schema_version.txt`, `summary.json`, JSONL records, and a
`manifest.json` with the git commit and SHA-256 checksums. Existing output directories are rejected.
The scripted pilot deliberately mixes optimal, exploratory, and broken-code policies inside each
prompt group, creating variable trajectory and segment counts while retaining exact rewards.

Run a small pilot by supplying a JSON config:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync symbolic-benchmark \
  artifacts/pilot-v1 --config environments/symbolic_tool_calling_v1/pilot_config.json
```

Or use the default matrix (20 tasks per horizon/imbalance condition and four rollouts per task):

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync symbolic-benchmark artifacts/default-v1
```

The offline scripted collector uses a deterministic lexical token proxy so pipeline tests do not
require a model tokenizer. Model-collected rollouts should populate the same schema with provider or
tokenizer-derived counts before compaction.

PPO records use terminal-only transition reward and segment-level GAE. Critic values must be causal
state-value predictions supplied by an external critic; the explicit `zero_baseline` is provided for
pipeline validation only. Segment-normalized GRPO assigns each segment weight `1 / num_segments`,
so every rollout retains total policy weight one.

Existing compaction artifacts can be transformed without recollecting rollouts:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync symbolic-training-examples \
  artifacts/run/compaction artifacts/run/training
```

Run all deterministic benchmark checks from the Prime-RL root:

```bash
uv run --no-sync pytest environments/symbolic_tool_calling_v1/tests -q
```
