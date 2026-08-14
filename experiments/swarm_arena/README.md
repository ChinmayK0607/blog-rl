# Swarm Arena: 4v4 coordination at small-model scale

The complete chronological record—including failed SFT runs, regression
diagnosis, infrastructure incidents, rejected credit estimators, immutable
artifacts, and current admission gates—is maintained in `RESEARCH_LOG.md`.

Swarm Arena is a deterministic, discrete network-control game for studying
whether small language-model agents learn useful team coordination. Four BLUE
agents face four fixed-policy RED agents on a partially observed graph. Agents
first broadcast private observations and intentions, then act simultaneously.

The experiment deliberately separates three questions:

1. Can a 4B instruct model obey the strict communication and action protocol?
2. Does generated communication improve team reward over dropped messages?
3. Does LoRA SFT create a reliable warm start without erasing sensitivity to
   other agents' messages?

The simulator does not invoke shells, containers, networks, or external systems.
Every transition and reward is locally deterministic. An exact joint-action
solver supplies oracle regret and filters ambiguous SFT labels.

## Immutable artifacts

- environment: `arena-core-v1`
- prompts: `arena-v2-structured-priority`
- SFT data: `arena-sft-v2`
- SFT SHA-256: `edad09bb301748621a0fab73ebf3de60d60abfd9f56c9afcc6ca02ffe12f3a80`
- frozen evaluation manifest SHA-256:
  `b53bfc523043ec71cc69f851d0819511c5a9f0b4f09520898f30954bbe874b29`

The full SFT JSONL is published to
[`CK0607/swarm-arena-sft-v2`](https://huggingface.co/datasets/CK0607/swarm-arena-sft-v2).
Only its manifest and independent audit are committed here.

## Reproduce the CPU audit

From the Prime-RL repository root:

```bash
uv run --with ./experiments/swarm_arena \
  pytest experiments/swarm_arena/tests -q

uv run --with ./experiments/swarm_arena \
  python -m swarm_ctf_eval.arena_data_audit \
  /path/to/arena_sft_stage1 --require-split-action-coverage

uv run --with ./experiments/swarm_arena \
  python -m swarm_ctf_eval.arena_eval \
  --provider oracle \
  --output-dir experiments/swarm_arena/results/oracle
```

## Experiment sequence

The order is fixed to avoid tuning on the final result:

1. evaluate untouched `Qwen/Qwen3-4B-Instruct-2507` on the frozen 60 cases;
2. run the small overfit config and verify protocol learning;
3. run the full LoRA SFT config;
4. select a checkpoint using validation generation metrics;
5. run the selected checkpoint once on the held-out SFT test split and frozen
   arena cases;
6. report generated, dropped, reference, and shuffled-message conditions.

Prime-RL configs are in `configs/`. See `GPU_HANDOFF.md` for promotion gates and
`ENVIRONMENT_CARD.md` for the exact mechanics and threat model.

## RL-native episode task

The one-turn frozen arena remains a protocol and SFT warm-start gate. The main
RL task is `arena-episode-v2`: 4-8 turns, private observations, grounded and
budgeted communication, opponent-policy switches, and terminal-only team
reward. It deliberately has no oracle message or long-horizon action labels.
See `RL_TASK_V2.md` for the task boundary and required causal controls.

Run its model-free audit and a learned-policy evaluation with:

```bash
uv run --with ./experiments/swarm_arena \
  python -m swarm_ctf_eval.episode_baselines \
  --output-dir experiments/swarm_arena/results/episode_v2/baselines

uv run --with ./experiments/swarm_arena --with peft \
  python experiments/swarm_arena/scripts/run_episode_eval.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter /path/to/lora_adapters \
  --output-dir experiments/swarm_arena/results/episode_v2/adapter
```

The learned-policy evaluator runs generated, dropped, sender-shuffled,
one-turn-delayed, and zero-budget episodes on the same frozen cases. A return
gain without positive paired communication effects is capability learning, not
evidence of swarm cooperation.

## Non-arena regression gate

`REGRESSION_PLAN.md` defines a frozen 256-case overspecialization suite and the
paired base-versus-adapter promotion thresholds. This is run before any RL
checkpoint is promoted and complements pinned IFEval, GSM8K, and ARC-Challenge
runs before a public capability claim.

## Score an SFT checkpoint

Prime-RL writes the unchanged base weights and the learned LoRA adapter
separately. Always pass the adapter explicitly when scoring a checkpoint:

```bash
uv run --with peft python experiments/swarm_arena/scripts/score_sft_split.py \
  --model outputs/swarm_arena/qwen3_4b_stage1/weights/step_310 \
  --adapter outputs/swarm_arena/qwen3_4b_stage1/weights/step_310/lora_adapters \
  --split validation \
  --output-dir experiments/swarm_arena/results/stage1/validation
```

After scoring every stable checkpoint on the validation split, select one with
the frozen behavioral gates and deterministic tie-breakers:

```bash
uv run --with ./experiments/swarm_arena \
  python -m swarm_ctf_eval.checkpoint_selection \
  --results-root experiments/swarm_arena/results/stage1/checkpoints \
  --base-summary experiments/swarm_arena/results/stage1/base_validation_corrected/summary.json \
  --output experiments/swarm_arena/results/stage1/selection.json
```

The selector rejects non-validation summaries and any checkpoint with
unsupported broadcast facts. Only after `selection.json` is written may the
chosen adapter be evaluated on the held-out test split.

The final base-versus-SFT arena claim uses paired case-level statistics. It is
only promoted when the 95% intervals support lower oracle regret, generated
messages beating dropped messages, and generated messages beating shuffled
messages:

```bash
uv run --with ./experiments/swarm_arena \
  python -m swarm_ctf_eval.arena_compare \
  --base-rows experiments/swarm_arena/results/base/arena/rows.jsonl \
  --base-summary experiments/swarm_arena/results/base/arena/summary.json \
  --sft-rows experiments/swarm_arena/results/stage1/selected/arena/rows.jsonl \
  --sft-summary experiments/swarm_arena/results/stage1/selected/arena/summary.json \
  --output experiments/swarm_arena/results/stage1/arena_comparison.json
```
