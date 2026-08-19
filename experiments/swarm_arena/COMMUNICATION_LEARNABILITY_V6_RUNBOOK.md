# Compact multi-pair communication learnability run

This is the direct follow-up to the negative pair-7 60-update run. Its narrow
question is whether terminal-return RL can teach two receiver policies to use a
teammate's private fact to switch between two already-legal actions.

## What changed

- Training uses critical pairs 7 and 9, both latent worlds of each pair, in
  every update. The receivers are different policy slots (`blue-1` and
  `blue-0`).
- Matched decoys are excluded from optimizer batches but retained in every
  online intervention evaluation. This removes the easiest generic-capture
  gradient without removing the null control.
- Each world uses eight common-random focused-agent replicas. Only the
  receiver's first ACT span receives its within-world leave-one-out advantage.
- The receiver sees a compact action view: legal actions, inbox, self state,
  local events, unknown neighbors, and known nodes targeted by a legal action.
- The reward remains only replay-verified terminal control delta. There is no
  message, target, capture, or action bonus.

The checked-in CPU audit must pass:

```bash
PYTHONPATH=experiments/swarm_arena uv run \
  experiments/swarm_arena/scripts/audit_communication_learnability_plan.py \
  --curriculum experiments/swarm_arena/data/rl_v4/staged_curriculum_v6_compact_multipair_40.json \
  --handoff-manifest experiments/swarm_arena/data/rl_v4/handoff_train.json
```

It certifies construction, not sampled advantage. The first completed update
should additionally log receiver action diversity and the fraction of non-zero
advantages. This is one cheap diagnostic, not a separate certification stage.

## Paid-run inputs

- Trainer: `configs/rl_v4_1_7b_compact_multipair_40.toml`
- Curriculum: `data/rl_v4/staged_curriculum_v6_compact_multipair_40.json`
- Updates: 40
- Checkpoints and online evaluations: 0, 10, 20, 30, 40
- Online evaluation: 48 matched games per checkpoint across pairs 7 and 9,
  critical and decoy worlds, and normal/dropped/shuffled messages.
- Final selection: the unchanged development suite. Frozen OOD remains unopened
  until a development-selected checkpoint exists.

Build the production plan from the new curriculum and the fresh host runtime
certificate. The plan binds the eight-replica count, compact prompt profile,
and the complete curriculum digest.

Set the normal launch environment from `COMMUNICATION_OVERFIT_RUNBOOK.md`, with
these replacements:

```bash
export SWARM_EXPECTED_UPDATES=40
export SWARM_CHECKPOINT_INTERVAL=10
export SWARM_CURRICULUM_ARTIFACT=$SWARM_REPO_ROOT/experiments/swarm_arena/data/rl_v4/staged_curriculum_v6_compact_multipair_40.json
export SWARM_RUN_ID=rl-v4-compact-multipair40-$(git rev-parse --short HEAD)
export SWARM_WANDB_GROUP=qwen3-1.7b-compact-multipair-40
```

The launcher reads `shared_return_replicas=8`,
`action_prompt_profile=focused_handoff_compact`, and
`online_evaluation_mode=multipair` directly from the bound curriculum. Manual
overrides must match the production plan or preflight fails before optimizer
work.

## Interpretation

A useful result requires all of the following:

1. normal messages improve receiver target choice over dropped and shuffled
   messages in aggregate;
2. the effect appears on both pair 7 and pair 9, rather than only one receiver;
3. critical intervention lift exceeds matched-decoy lift; and
4. development evaluation improves without protocol or ordinary-game
   regression.

Higher return without these intervention effects is another tactical-capability
gain, not communication learning. A positive training-pair result establishes
learnability only; held-out development is required before claiming
generalization.
