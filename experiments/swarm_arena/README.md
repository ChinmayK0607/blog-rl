# Swarm Arena: 4v4 coordination at small-model scale

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
