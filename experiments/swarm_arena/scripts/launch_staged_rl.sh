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
: "${SWARM_LIVE_HF_REPO:?set SWARM_LIVE_HF_REPO to a public model repo for live recovery artifacts}"
: "${SWARM_DEADLINE_EPOCH:?set SWARM_DEADLINE_EPOCH to the pod termination Unix timestamp}"

swarm_uv=${SWARM_UV:-/root/.local/bin/uv}
swarm_inference_config=${SWARM_INFERENCE_CONFIG:-$SWARM_REPO_ROOT/experiments/swarm_arena/configs/inference_1_7b_l40s.toml}
swarm_data_dir=${SWARM_DATA_DIR:-$SWARM_REPO_ROOT/experiments/swarm_arena/data/rl_v4}
swarm_eval_root=$SWARM_RUN_DIR/evaluations
swarm_barrier_dir=$SWARM_RUN_DIR/control/checkpoint_barriers
swarm_rescore_dir=$SWARM_RUN_DIR/control/rescore
swarm_session_prefix=${SWARM_SESSION_PREFIX:-swarm-staged}
swarm_source_commit=$(git -C "$SWARM_REPO_ROOT" rev-parse HEAD)
swarm_expected_updates=${SWARM_EXPECTED_UPDATES:-$(jq '[.curriculum_stages[].updates] | add' "$SWARM_PRODUCTION_PLAN")}
swarm_checkpoint_interval=${SWARM_CHECKPOINT_INTERVAL:-10}
swarm_credit_assignment=${SWARM_SHARED_RETURN_CREDIT_ASSIGNMENT:-shared_team}
swarm_curriculum_artifact=${SWARM_CURRICULUM_ARTIFACT:-$SWARM_REPO_ROOT/experiments/swarm_arena/data/rl_v4/staged_curriculum_v3_joint_80.json}
swarm_shared_return_replicas=${SWARM_SHARED_RETURN_REPLICAS:-$(jq -r '.runtime.shared_return_replicas // 4' "$swarm_curriculum_artifact")}
swarm_action_prompt_profile=${SWARM_ACTION_PROMPT_PROFILE:-$(jq -r '.runtime.action_prompt_profile // "full"' "$swarm_curriculum_artifact")}
swarm_wandb_group=${SWARM_WANDB_GROUP:-qwen3-1.7b-staged-$swarm_expected_updates}
swarm_wandb_model_tag=${SWARM_WANDB_MODEL_TAG:-1.7b}
swarm_controller_wandb_mode=${SWARM_CONTROLLER_WANDB_MODE:-online}
swarm_pytorch_cuda_alloc_conf=${SWARM_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
swarm_final_sync_margin=${SWARM_FINAL_SYNC_MARGIN:-2700}
swarm_pulse_mode=${SWARM_PULSE_MODE:-$(jq -r '.runtime.online_evaluation_mode // "full"' "$swarm_curriculum_artifact")}
swarm_communication_eval_turns=${SWARM_COMMUNICATION_EVAL_TURNS:-$(jq -r '.runtime.online_eval_remaining_turns // 2' "$swarm_curriculum_artifact")}
swarm_mirror_interval_steps=${SWARM_MIRROR_INTERVAL_STEPS:-1}
swarm_multipair_args=()
while IFS= read -r swarm_pair_index; do
  if [[ -n "$swarm_pair_index" ]]; then
    swarm_multipair_args+=(--multipair-index "$swarm_pair_index")
  fi
done < <(jq -r '.online_eval_pair_indices[]?' "$swarm_curriculum_artifact")

case "$swarm_controller_wandb_mode" in
  online) swarm_wandb_mode_arg=() ;;
  offline) swarm_wandb_mode_arg=(--offline) ;;
  *) echo "SWARM_CONTROLLER_WANDB_MODE must be online or offline" >&2; exit 1 ;;
esac
if ! [[ "$SWARM_DEADLINE_EPOCH" =~ ^[0-9]+$ ]] || (( SWARM_DEADLINE_EPOCH <= $(date +%s) )); then
  echo "SWARM_DEADLINE_EPOCH must be a future Unix timestamp" >&2
  exit 1
fi
if ! [[ "$swarm_mirror_interval_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "SWARM_MIRROR_INTERVAL_STEPS must be a positive integer" >&2
  exit 1
fi

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
  --curriculum-artifact "$swarm_curriculum_artifact" \
  --runtime-certificate "$SWARM_RUNTIME_CERTIFICATE" \
  --data-dir "$swarm_data_dir" \
  --initial-adapter "$SWARM_INITIAL_ADAPTER" \
  --model "$SWARM_MODEL" \
  --base-url http://127.0.0.1:8001 \
  --base-url http://127.0.0.1:8002 \
  --base-url http://127.0.0.1:8003 \
  --expected-updates "$swarm_expected_updates" \
  --checkpoint-interval "$swarm_checkpoint_interval" \
  --shared-return-credit-assignment "$swarm_credit_assignment" \
  --shared-return-replicas "$swarm_shared_return_replicas" \
  --action-prompt-profile "$swarm_action_prompt_profile"

swarm_plan_sha=$(jq -r .production_plan_sha256 "$SWARM_RUN_DIR/PREFLIGHT.json")
mkdir -p "$SWARM_RUN_DIR/logs" "$swarm_eval_root" "$swarm_rescore_dir"

# A run is not safe to leave unattended until the recovery repository is
# anonymously readable.  Verify that before starting any new optimizer work.
"$swarm_uv" run python \
  "$SWARM_REPO_ROOT/experiments/swarm_arena/scripts/run_live_artifact_mirror.py" \
  --repo-id "$SWARM_LIVE_HF_REPO" \
  --run-id "$SWARM_RUN_ID" \
  --run-dir "$SWARM_RUN_DIR" \
  --deadline-epoch "$SWARM_DEADLINE_EPOCH" \
  --final-sync-margin "$swarm_final_sync_margin" \
  --artifact "$SWARM_PRODUCTION_PLAN" \
  --artifact "$SWARM_RUNTIME_CERTIFICATE" \
  --artifact "$swarm_curriculum_artifact" \
  --preflight-only \
  > "$SWARM_RUN_DIR/logs/mirror-preflight.log" 2>&1

for swarm_role in trainer rescore pulses wandb mirror controller; do
  if tmux has-session -t "$swarm_session_prefix-$swarm_role" 2>/dev/null; then
    echo "refusing to reuse tmux session $swarm_session_prefix-$swarm_role" >&2
    exit 1
  fi
done

tmux new-session -d -s "$swarm_session_prefix-trainer" \
  "cd $SWARM_REPO_ROOT && export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=$swarm_pytorch_cuda_alloc_conf && exec $swarm_uv run torchrun --standalone --nproc-per-node=1 .venv/bin/trainer @ $SWARM_RUN_DIR/trainer.toml > $SWARM_RUN_DIR/logs/trainer.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-rescore" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/run_lag_zero_rescore_worker.py --root $swarm_rescore_dir --snapshot-manifest $swarm_rescore_dir/current_snapshots.json --production-plan-sha256 $swarm_plan_sha > $SWARM_RUN_DIR/logs/rescore.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-pulses" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/run_staged_pulses.py --repo-root $SWARM_REPO_ROOT --run-dir $SWARM_RUN_DIR --production-plan $SWARM_PRODUCTION_PLAN --barrier-dir $swarm_barrier_dir --eval-root $swarm_eval_root --data-dir $swarm_data_dir --base-url http://127.0.0.1:8001 --base-url http://127.0.0.1:8002 --base-url http://127.0.0.1:8003 --baseline-revision $SWARM_INITIAL_POLICY_REVISION --expected-updates $swarm_expected_updates --interval $swarm_checkpoint_interval --evaluation-mode $swarm_pulse_mode --communication-remaining-turns $swarm_communication_eval_turns ${swarm_multipair_args[*]} > $SWARM_RUN_DIR/logs/pulses.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-wandb" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/log_live_rl_wandb.py --progress $SWARM_RUN_DIR/live_rl_progress.json --eval-root $swarm_eval_root --expected-updates $swarm_expected_updates --finish-marker $swarm_eval_root/COMPLETE --project swarm-arena-rl --group $swarm_wandb_group --run-name $SWARM_RUN_ID-controller --run-id $SWARM_RUN_ID-controller-v1 --tag $swarm_wandb_model_tag --tag causal-communication --tag development ${swarm_wandb_mode_arg[*]} --compact-artifact $SWARM_PRODUCTION_PLAN --compact-artifact $swarm_curriculum_artifact > $SWARM_RUN_DIR/logs/wandb.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-mirror" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/run_live_artifact_mirror.py --repo-id $SWARM_LIVE_HF_REPO --run-id $SWARM_RUN_ID --run-dir $SWARM_RUN_DIR --deadline-epoch $SWARM_DEADLINE_EPOCH --final-sync-margin $swarm_final_sync_margin --compact-interval-steps $swarm_mirror_interval_steps --artifact $SWARM_PRODUCTION_PLAN --artifact $SWARM_RUNTIME_CERTIFICATE --artifact $swarm_curriculum_artifact > $SWARM_RUN_DIR/logs/mirror.log 2>&1"

sleep 15
if ! tmux has-session -t "$swarm_session_prefix-trainer" 2>/dev/null; then
  echo "trainer exited during startup" >&2
  exit 1
fi

tmux new-session -d -s "$swarm_session_prefix-controller" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec $swarm_uv run python experiments/swarm_arena/scripts/run_live_rl.py --output-dir $SWARM_RUN_DIR --trainer-config $SWARM_RUN_DIR/trainer.toml --inference-config $swarm_inference_config --data-dir $swarm_data_dir --task-data-version v4 --tokenizer $SWARM_MODEL --initial-adapter $SWARM_INITIAL_ADAPTER --base-url http://127.0.0.1:8001 --base-url http://127.0.0.1:8002 --base-url http://127.0.0.1:8003 --actor vllm --run-id $SWARM_RUN_ID --source-commit $swarm_source_commit --base-revision $SWARM_BASE_REVISION --initial-policy-revision $SWARM_INITIAL_POLICY_REVISION --credit-estimator shared_return --shared-return-replicas $swarm_shared_return_replicas --shared-return-credit-assignment $swarm_credit_assignment --shared-return-action-prompt-profile $swarm_action_prompt_profile --production-plan $SWARM_PRODUCTION_PLAN --async-rescore-dir $swarm_rescore_dir --async-rescore-timeout 600 --steps $swarm_expected_updates --groups-per-step 4 --scenario-source curriculum --curriculum-split train --update-timeout 1200 --checkpoint-barrier-dir $swarm_barrier_dir --checkpoint-barrier-interval $swarm_checkpoint_interval --checkpoint-barrier-timeout 7200 > $SWARM_RUN_DIR/logs/controller.log 2>&1"

sleep 10
for swarm_role in trainer rescore pulses wandb mirror controller; do
  if ! tmux has-session -t "$swarm_session_prefix-$swarm_role" 2>/dev/null; then
    echo "$swarm_role exited during launch; stopping the new training stack" >&2
    for swarm_cleanup in trainer rescore pulses wandb mirror controller; do
      tmux kill-session -t "$swarm_session_prefix-$swarm_cleanup" 2>/dev/null || true
    done
    exit 1
  fi
done

echo "started $SWARM_RUN_ID at source $swarm_source_commit with plan $swarm_plan_sha"
