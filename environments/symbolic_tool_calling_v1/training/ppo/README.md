# Vanilla PPO on the symbolic tool-calling benchmark

Online vanilla PPO (Schulman et al. 2017, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)):
actor-critic with a clipped-surrogate policy loss and GAE advantages from a learned scalar value
head — not DPPO (the default prime-rl trust-region loss) and not a group baseline (GRPO).

## How the pieces map onto prime-rl

| PPO component | Where it runs | Config |
| --- | --- | --- |
| Terminal reward per rollout | Orchestrator: the `ppo` algorithm stamps it on each sample's last action token (`rewards` wire stream) | `orchestrator.algo.type = "ppo"` |
| Critic V(s) | Trainer: zero-initialized scalar head over the policy backbone (custom Qwen3 only), checkpointed, never broadcast to inference | `trainer.model.ppo_value_head = true` |
| GAE(γ, λ) advantages + λ-return value targets | Trainer, from its own value predictions (`compute_token_gae`) | `trainer.loss.gamma`, `trainer.loss.gae_lambda` |
| Clipped surrogate policy loss | Trainer (`ppo_policy_loss_fn`) | `trainer.loss.policy_clip` |
| Clipped value loss | Trainer (inline critic block) | `trainer.loss.value_clip`, `trainer.loss.value_coef` |
| Entropy bonus | Trainer | `trainer.loss.entropy_coef` |

Each timestep is one action token; masked context tokens (prompt, tool responses) are part of the
environment transition between actions. The state value for the action token at position `a` is the
value prediction at `a - 1` (the hidden state that decided it).

Because prime-rl trains every batch exactly once, the trainer's pre-update forward values are the
behavior values of vanilla PPO (so the value clip is a no-op online, kept for the offline compacted
pipeline which ships genuine behavior values). The only deviation from the textbook loop is the
async off-policy lag prime-rl always has (`max_off_policy_steps`): rollouts may come from a policy
up to N steps stale, which the surrogate ratio already corrects.

`group_size` only fans out sampling — PPO credit is per-rollout, so no group barrier or group
baseline exists. Rollouts ship no advantage stream (credit is computed in the trainer), which means
zero-advantage filtering never fires; every rollout trains, including all-zero-reward ones (the
critic still learns from them). Like opd/opsd, the `is_trainable` metric reads 0 for ppo rollouts —
it keys off the shipped advantage stream and is cosmetic here.

## Run

```bash
uv run rl @ environments/symbolic_tool_calling_v1/training/ppo/rl_qwen3_thinking_ppo.toml
```

Same 8×H100 split as the GRPO pilot: 6 data-parallel inference replicas, 2 trainer GPUs.

## What to watch

- `ppo/explained_variance` — should climb from ~0 (zero-init head) toward 1; flat/negative means
  the critic isn't learning and advantages are just `reward - 0`.
- `ppo/value_loss` — should fall after the initial transient.
- `ppo/clip_fraction` — fraction of tokens hitting the surrogate clip; persistently high values
  mean the policy is moving too fast per step (lower the LR or the off-policy lag).
- Reward/val curves vs the GRPO pilot (`rl_qwen3_thinking_6i2t_pilot.toml`) — this is the point of
  the comparison: identical env, sampling, and budget; only the credit assignment differs.

## Knobs

Defaults follow the common LLM-RL PPO setting: `gamma = 1.0` (undiscounted), `gae_lambda = 0.95`,
`policy_clip = 0.2`, `value_coef = 0.5`, `entropy_coef = 0` (the paper's Atari setting is 0.01).
`trainer.loss.normalize_advantages = true` enables per-micro-batch advantage whitening
(openai-baselines style); off by default, matching the paper's objective.
