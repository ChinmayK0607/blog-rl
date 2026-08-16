#!/usr/bin/env bash
set -euo pipefail

: "${SWARM_REPO_ROOT:?set SWARM_REPO_ROOT}"
: "${SWARM_RUN_DIR:?set SWARM_RUN_DIR}"
: "${SWARM_PRODUCTION_PLAN:?set SWARM_PRODUCTION_PLAN}"
: "${SWARM_RUNTIME_CERTIFICATE:?set SWARM_RUNTIME_CERTIFICATE}"
: "${SWARM_MODEL:?set SWARM_MODEL}"
: "${SWARM_INITIAL_ADAPTER:?set SWARM_INITIAL_ADAPTER}"
: "${SWARM_BASE_REVISION:?set SWARM_BASE_REVISION}"
: "${SWARM_INITIAL_POLICY_REVISION:?set SWARM_INITIAL_POLICY_REVISION}"
: "${SWARM_RUN_ID:?set SWARM_RUN_ID}"

swarm_uv=${SWARM_UV:-/root/.local/bin/uv}
swarm_inference_config=${SWARM_INFERENCE_CONFIG:-$SWARM_REPO_ROOT/experiments/swarm_arena/configs/inference_1_7b_l40s.toml}
swarm_data_dir=$SWARM_REPO_ROOT/experiments/swarm_arena/data/rl_v4
swarm_eval_root=$SWARM_RUN_DIR/evaluations
swarm_barrier_dir=$SWARM_RUN_DIR/control/checkpoint_barriers
swarm_rescore_dir=$SWARM_RUN_DIR/control/rescore
swarm_session_prefix=${SWARM_SESSION_PREFIX:-swarm-staged}
swarm_source_commit=$(git -C "$SWARM_REPO_ROOT" rev-parse HEAD)

for swarm_port in 8001 8002 8003; do
  curl -fsS "http://127.0.0.1:$swarm_port/health" >/dev/null
done

export PYTHONPATH=$SWARM_REPO_ROOT/experiments/swarm_arena
"$swarm_uv" run python \
  "$SWARM_REPO_ROOT/experiments/swarm_arena/scripts/preflight_staged_rl.py" \
  --repo-root "$SWARM_REPO_ROOT" \
  --source-commit "$swarm_source_commit" \
  --run-dir "$SWARM_RUN_DIR" \
  --inference-config "$swarm_inference_config" \
  --production-plan "$SWARM_PRODUCTION_PLAN" \
  --runtime-certificate "$SWARM_RUNTIME_CERTIFICATE" \
  --data-dir "$swarm_data_dir" \
  --initial-adapter "$SWARM_INITIAL_ADAPTER" \
  --model "$SWARM_MODEL" \
  --base-url http://127.0.0.1:8001 \
  --base-url http://127.0.0.1:8002 \
  --base-url http://127.0.0.1:8003 \
  --expected-updates 120 \
  --checkpoint-interval 10

swarm_plan_sha=$(jq -r .production_plan_sha256 "$SWARM_RUN_DIR/PREFLIGHT.json")
mkdir -p "$SWARM_RUN_DIR/logs" "$swarm_eval_root" "$swarm_rescore_dir"

for swarm_role in trainer rescore pulses wandb controller; do
  if tmux has-session -t "$swarm_session_prefix-$swarm_role" 2>/dev/null; then
    echo "refusing to reuse tmux session $swarm_session_prefix-$swarm_role" >&2
    exit 1
  fi
done

tmux new-session -d -s "$swarm_session_prefix-trainer" \
  "cd $SWARM_REPO_ROOT && export CUDA_VISIBLE_DEVICES=0 && exec $swarm_uv run torchrun --standalone --nproc-per-node=1 .venv/bin/trainer @ $SWARM_RUN_DIR/trainer.toml > $SWARM_RUN_DIR/logs/trainer.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-rescore" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/run_lag_zero_rescore_worker.py --root $swarm_rescore_dir --snapshot-manifest $swarm_rescore_dir/current_snapshots.json --production-plan-sha256 $swarm_plan_sha > $SWARM_RUN_DIR/logs/rescore.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-pulses" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/run_staged_pulses.py --repo-root $SWARM_REPO_ROOT --run-dir $SWARM_RUN_DIR --production-plan $SWARM_PRODUCTION_PLAN --barrier-dir $swarm_barrier_dir --eval-root $swarm_eval_root --data-dir $swarm_data_dir --base-url http://127.0.0.1:8001 --base-url http://127.0.0.1:8002 --base-url http://127.0.0.1:8003 --baseline-revision $SWARM_INITIAL_POLICY_REVISION --expected-updates 120 --interval 10 > $SWARM_RUN_DIR/logs/pulses.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-wandb" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/log_live_rl_wandb.py --progress $SWARM_RUN_DIR/live_rl_progress.json --eval-root $swarm_eval_root --expected-updates 120 --finish-marker $swarm_eval_root/COMPLETE --project swarm-arena-rl --group qwen3-1.7b-staged-120 --run-name $SWARM_RUN_ID-controller --run-id $SWARM_RUN_ID-controller-v1 --tag 1.7b --tag causal-communication --tag development --compact-artifact $SWARM_PRODUCTION_PLAN --compact-artifact $SWARM_REPO_ROOT/experiments/swarm_arena/data/rl_v4/staged_curriculum_v1.json > $SWARM_RUN_DIR/logs/wandb.log 2>&1"

sleep 15
if ! tmux has-session -t "$swarm_session_prefix-trainer" 2>/dev/null; then
  echo "trainer exited during startup" >&2
  exit 1
fi

tmux new-session -d -s "$swarm_session_prefix-controller" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/run_live_rl.py --output-dir $SWARM_RUN_DIR --trainer-config $SWARM_RUN_DIR/trainer.toml --inference-config $swarm_inference_config --data-dir $swarm_data_dir --task-data-version v4 --tokenizer $SWARM_MODEL --initial-adapter $SWARM_INITIAL_ADAPTER --base-url http://127.0.0.1:8001 --base-url http://127.0.0.1:8002 --base-url http://127.0.0.1:8003 --actor vllm --run-id $SWARM_RUN_ID --source-commit $swarm_source_commit --base-revision $SWARM_BASE_REVISION --initial-policy-revision $SWARM_INITIAL_POLICY_REVISION --credit-estimator shared_return --shared-return-replicas 4 --production-plan $SWARM_PRODUCTION_PLAN --async-rescore-dir $swarm_rescore_dir --async-rescore-timeout 600 --steps 120 --groups-per-step 4 --scenario-source curriculum --curriculum-split train --update-timeout 1200 --checkpoint-barrier-dir $swarm_barrier_dir --checkpoint-barrier-interval 10 --checkpoint-barrier-timeout 7200 > $SWARM_RUN_DIR/logs/controller.log 2>&1"

sleep 10
for swarm_role in trainer rescore pulses controller; do
  if ! tmux has-session -t "$swarm_session_prefix-$swarm_role" 2>/dev/null; then
    echo "$swarm_role exited during launch; stopping the new training stack" >&2
    for swarm_cleanup in trainer rescore pulses wandb controller; do
      tmux kill-session -t "$swarm_session_prefix-$swarm_cleanup" 2>/dev/null || true
    done
    exit 1
  fi
done

echo "started $SWARM_RUN_ID at source $swarm_source_commit with plan $swarm_plan_sha"
