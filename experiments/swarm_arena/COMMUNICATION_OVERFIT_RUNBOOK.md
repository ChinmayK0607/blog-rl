# Pair-7 communication overfit: GPU runbook

This run answers one narrow question: can terminal-return RL teach the receiver
in handoff pair 7 to choose between two targets using a teammate's message?
Both latent worlds and matched decoys occur in every update. The online probe is
small; held-out development evaluation happens only for checkpoint selection.

## Before renting the GPU

- Branch: `exp/swarm-arena-4b`
- Trainer: `configs/rl_v4_1_7b_communication_overfit_60.toml`
- Curriculum: `data/rl_v4/staged_curriculum_v5_communication_overfit_60.json`
- Public recovery repo: create or reuse one public HF model repository, for
  example `CK0607/swarm-arena-live-runs`.
- Record the provider's exact termination time as a Unix epoch. Allow at least
  45 minutes between the final planned update and provider termination.

Do not place HF or W&B tokens in Git, shell history, launch logs, or this file.
Authenticate through the standard HF and W&B credential stores.

## One-time host preparation

```bash
git clone --recurse-submodules --branch exp/swarm-arena-4b \
  git@github.com:ChinmayK0607/blog-rl.git /workspace/blog-rl
cd /workspace/blog-rl
git submodule update --init --recursive
uv sync
PYTHONPATH=experiments/swarm_arena uv run pytest -q experiments/swarm_arena/tests
```

The full CPU suite runs once per fresh source commit, not before every restart
of an unchanged commit. The exact-host numerical probe and structured serving
probe remain required once because those properties depend on the GPU runtime.

## Launch contract

Prepare the immutable production plan, runtime certificate, model, initializer,
and three serving endpoints using `STAGED_RL_PLAN.md`. Then export:

```bash
export SWARM_REPO_ROOT=/workspace/blog-rl
export SWARM_RUN_ID=rl-v4-pair7-overfit-60-$(git rev-parse --short HEAD)
export SWARM_RUN_DIR=/workspace/runs/$SWARM_RUN_ID
export SWARM_PRODUCTION_PLAN=/workspace/plans/$SWARM_RUN_ID.json
export SWARM_RUNTIME_CERTIFICATE=/workspace/certificates/$SWARM_RUN_ID.json
export SWARM_MODEL=/workspace/models/qwen3-1.7b
export SWARM_INITIAL_ADAPTER=/workspace/artifacts/warmstart-1.7b-step320
export SWARM_BASE_REVISION=<immutable-base-revision>
export SWARM_INITIAL_POLICY_REVISION=<immutable-sft-revision>
export SWARM_CURRICULUM_ARTIFACT=$SWARM_REPO_ROOT/experiments/swarm_arena/data/rl_v4/staged_curriculum_v5_communication_overfit_60.json
export SWARM_EXPECTED_UPDATES=60
export SWARM_CHECKPOINT_INTERVAL=10
export SWARM_PULSE_MODE=pair7
export SWARM_CONTROLLER_WANDB_MODE=online
export SWARM_LIVE_HF_REPO=CK0607/swarm-arena-live-runs
export SWARM_DEADLINE_EPOCH=<provider-termination-unix-epoch>
export SWARM_FINAL_SYNC_MARGIN=2700
export SWARM_MIRROR_INTERVAL_STEPS=1

bash experiments/swarm_arena/scripts/launch_staged_rl.sh
```

The launcher refuses to start optimizer work unless it can create, anonymously
download, and byte-verify:

`https://huggingface.co/CK0607/swarm-arena-live-runs/blob/main/runs/<run-id>/HEARTBEAT.json`

## What is saved off-node

- Per-update controller and per-checkpoint evaluation metrics stream to W&B.
- Every completed update, compact progress, logs, launch inputs, and completed
  pair-7 evaluation rows/summaries are committed to HF.
- At updates 10, 20, ..., 60, all four complete LoRA adapters are committed to
  `runs/<run-id>/checkpoints/step-<N>/`. Adapter hashes must equal the controller
  barrier record and are verified by anonymous download.
- Full model weights, optimizer `rank_0.pt`, caches, and raw generation traces
  stay on the pod. This prevents Mac storage growth and unnecessary hub usage.

## Minimal monitoring

The run is healthy when:

1. `controller/update` advances in W&B;
2. `eval/train_pair/normal_minus_dropped_return` and receiver target-action rate
   appear at updates 0, 10, ..., 60;
3. `LIVE_MIRROR_STATUS.json` says `healthy`; and
4. after update 10, the public HF repo contains four verified adapter files.

If the trainer fails, preserve the failed run directory and start a clearly
named replacement from the latest publicly verified four-policy checkpoint.
Never resume from a partial update or from an adapter absent from the public
manifest.

## Result interpretation

- Training-pair success: normal messages beat both dropped and sender-shuffled
  messages on critical pair 7, receiver target choice rises, and the same effect
  is materially smaller on the matched decoy.
- Capability-only gain: return rises but intervention differences do not.
- Generalization: a selected checkpoint repeats the effect on the unchanged
  development handoffs. Do this only after the 60-update run or a clearly
  superior earlier checkpoint is selected.
- The frozen OOD suite stays unopened until a checkpoint passes development.

It is safe to decommission only after the latest complete checkpoint and compact
summary are publicly readable and checksum-verified. The mirror's deadline sync
is a safety net, not a substitute for this final check.
