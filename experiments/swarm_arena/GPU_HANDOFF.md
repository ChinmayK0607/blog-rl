# GPU handoff: Stage-1 SFT

## Required machine

- One RTX A6000 (48 GB) or equivalent NVIDIA GPU
- Linux with a working CUDA/PyTorch stack
- At least 80 GB free disk
- Access to the selected base checkpoint

The CPU artifacts are ready. Do not regenerate or edit the frozen evaluation
manifest on the GPU machine.

## Immutable inputs

- Dataset: `data/arena_sft_stage1`
- Dataset version: `arena-sft-v2`
- Dataset content SHA-256:
  `edad09bb301748621a0fab73ebf3de60d60abfd9f56c9afcc6ca02ffe12f3a80`
- Prompt version: `arena-v2-structured-priority`
- Environment version: `arena-core-v1`
- Frozen evaluation manifest SHA-256:
  `b53bfc523043ec71cc69f851d0819511c5a9f0b4f09520898f30954bbe874b29`

Re-run before training:

```bash
uv run --project experiments/swarm_arena --with pytest \
  pytest experiments/swarm_arena/tests -q
uv run --with ./experiments/swarm_arena \
  python -m swarm_ctf_eval.arena_data_audit \
  data/arena_sft_stage1 --require-split-action-coverage
uv run --with ./experiments/swarm_arena \
  python -m swarm_ctf_eval.arena_eval \
  --provider oracle --output-dir results/arena_v2/oracle
```

## Experiment order

1. Serve the untouched base model and run the frozen arena evaluation. Save every
   raw response and the summary. This is the pre-SFT baseline.
2. Run the 256-row overfit/smoke experiment. Stop if strict JSON does not approach
   100%; that indicates a template or loss-mask problem, not insufficient data.
3. Train LoRA on `train.jsonl`; use `validation.jsonl` for checkpoint selection.
4. Select by validation task metrics, not training loss alone.
5. Evaluate the selected checkpoint exactly once on `test.jsonl` and the frozen
   60-case arena manifest.
6. Compare base versus SFT under generated, dropped, reference, and shuffled
   communication. Do not begin MARL until protocol and mechanics gates pass.

## Initial LoRA recipe

- Base: `Qwen/Qwen3-4B-Instruct-2507`
- Training stack: the pinned Prime-RL branch and its default numerical dtypes
- LoRA rank: 32
- LoRA alpha: 64
- LoRA dropout: 0.05
- Targets: attention Q/K/V/O and MLP gate/up/down projections
- Sequence length: 2048
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0.01`
- Scheduler: cosine, 3% warmup
- Effective batch size: 32
- Epochs: start with 2; do not extend automatically
- Gradient clipping: 1.0
- Gradient checkpointing: enabled
- Loss: assistant tokens only; system and user tokens masked
- Save/evaluate frequently enough to obtain at least 8 validation measurements

Raw data are broadcast-heavy (4,608 broadcast versus 900 action examples). The
published training splits sample 60% broadcast, 31% common actions, and 9% rare
`WAIT`/`SCAN`/`TRANSFER` actions. Validation and test remain unweighted.

## SFT promotion gates

The selected checkpoint must satisfy all of the following without relaxing the
strict parser:

- at least 99.5% exact action-schema validity;
- at least 99% exact broadcast-schema validity;
- zero unsupported broadcast facts on validation;
- at least 95% legal actions;
- improvement over the base model in mean oracle regret;
- no material degradation under action-order permutations;
- no improvement claim if generated messages fail to beat dropped messages or
  fail to degrade under shuffled-message intervention.

Failure to meet a gate triggers data/prompt diagnosis. It does not justify opening
the test set repeatedly or weakening the scorer.
