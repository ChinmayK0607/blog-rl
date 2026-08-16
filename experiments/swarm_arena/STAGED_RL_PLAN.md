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
| communication introduction | 40 | 40 / 30 / 30 | 14–16 nodes, 6–8 turns |
| communication heavy | 40 | 30 / 35 / 35 | 16–18 nodes, 8–10 turns |
| adaptive self-play | 20 | 30 / 35 / 35 | 18–20 nodes, 10–12 turns |

Every individual update contains equal numbers of critical and matched-decoy
games. A communication-only update is exactly two critical plus the two matched
decoys, never four unpaired critical games. Across the run this produces 176
ordinary games, 152 critical games, and 152 matched decoys using handoff pair
indices 0–151 from the existing oracle-certified v4 training manifest.

The opponent schedule remains an exact per-update rotation through base, SFT,
historical, and current-policy opponents. This prevents an update from being
identified with one opponent family and keeps the current-policy opponent
adaptive without letting RED share BLUE's private state.

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

## Build the immutable runtime plan

The checked-in curriculum contains no machine paths or opponent checkpoint
locations. On the GPU host, bind it to a verified four-opponent production plan:

```bash
uv run --with ./experiments/swarm_arena \
  python experiments/swarm_arena/scripts/build_staged_rl_plan.py \
  --base-plan /workspace/run-inputs/verified-v4-production-plan.json \
  --curriculum experiments/swarm_arena/data/rl_v4/staged_curriculum_v1.json \
  --handoff-manifest experiments/swarm_arena/data/rl_v4/handoff_train.json \
  --output /workspace/run-inputs/staged-120-production-plan.json
```

The builder reloads the result through the production-plan validator and emits
an audit containing the exact per-stage counts, maximum handoff pair index,
unique ordinary seeds, plan hash, and full schedule hash. The controller rejects
any `--steps` value other than 120 for this plan.

## Evaluation and W&B

Trainer internals are logged to the `swarm-arena-rl` W&B project by the trainer
config. A separate sidecar logs controller-level terminal return, advantage
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
  --project swarm-arena-rl \
  --group qwen3-1.7b-staged-120 \
  --run-name qwen3-1.7b-staged-120-controller \
  --run-id qwen3-1.7b-staged-120-controller-v1 \
  --tag 1.7b --tag causal-communication --tag development \
  --compact-artifact /workspace/run-inputs/staged-120-production-plan.json \
  --compact-artifact experiments/swarm_arena/data/rl_v4/staged_curriculum_v1.json
```

At updates 0, 10, ..., 120, use the 32-game `pulse` development tier with
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

## Storage and publication

The GPU host holds transient caches and checkpoints. W&B receives scalar curves
and compact JSON summaries. Git receives source, plans, hashes, audits, and the
research log. Only explicitly retained adapters are pushed to a public
Hugging Face repository; no checkpoint or raw rollout corpus is copied to the
Mac. Credentials are supplied only through environment variables and never
written into plans, logs, commands committed to Git, or W&B config.
