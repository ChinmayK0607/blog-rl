# Vanilla GRPO on the symbolic tool-calling benchmark

Original GRPO as introduced by DeepSeekMath ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)):
group advantage **whitened by the group std**,

```
A_i = (r_i − mean(r_group)) / std(r_group)
```

versus the repo's default `grpo` type, which is the Dr.GRPO revision (mean baseline only, no std
division — std-normalization upweights near-uniform groups, whose baseline estimate is noisiest).

Scope: the paper's advantage estimator is the only knob this algorithm changes. The clipped /
trust-region policy loss is `trainer.loss` (default DPPO+KL here, same as the GRPO pilot so only
the advantage differs); prime-rl applies no per-sequence `1/|o|` normalization (global token-count
normalization instead); and the paper's KL to a frozen reference policy has no counterpart —
prime-rl hosts no reference model for the rl loss.

Zero-variance groups (all rollouts same reward) carry all-zero advantages and are dropped by the
zero-advantage filter — identical behavior to the mean-only form.

## Run

```bash
uv run rl @ environments/symbolic_tool_calling_v1/training/vanilla_grpo/rl_qwen3_thinking_vanilla_grpo.toml
```

Identical env, sampling, and budget to the GRPO pilot (`rl_qwen3_thinking_6i2t_pilot.toml`) and the
PPO setup (`../ppo/`) — the three-way comparison isolates credit assignment.
