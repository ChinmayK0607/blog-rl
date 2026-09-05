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
: "${SWARM_PUBLIC_BASE_REPO:?set SWARM_PUBLIC_BASE_REPO to the exact prepared public base repository}"
: "${SWARM_PUBLIC_ADAPTER_REPO:?set SWARM_PUBLIC_ADAPTER_REPO to the exact prepared public adapter repository}"
: "${SWARM_RUN_ID:?set SWARM_RUN_ID}"
: "${SWARM_LIVE_HF_REPO:?set SWARM_LIVE_HF_REPO to a public model repo for live recovery artifacts}"
: "${SWARM_DEADLINE_EPOCH:?set SWARM_DEADLINE_EPOCH to the pod termination Unix timestamp}"
: "${SWARM_OPERATIONAL_PROFILE:?set SWARM_OPERATIONAL_PROFILE to configuration-bound measured timing evidence}"

swarm_uv=${SWARM_UV:-/root/.local/bin/uv}
swarm_uv_args=(run --frozen --extra flash-attn)
printf -v swarm_uv_command '%q ' "$swarm_uv" "${swarm_uv_args[@]}"
if [[ -n "${SWARM_UV_CACHE_DIR:-}" ]]; then
  export UV_CACHE_DIR=$SWARM_UV_CACHE_DIR
fi
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
swarm_target_swap_sender_retries=${SWARM_TARGET_SWAP_SENDER_RETRIES:-$(jq -r '.runtime.target_swap_sender_retries // 0' "$swarm_curriculum_artifact")}
swarm_stage_gates=${SWARM_STAGE_GATES:?set SWARM_STAGE_GATES to the frozen stage-gate artifact}
swarm_wandb_group=${SWARM_WANDB_GROUP:-qwen3-1.7b-staged-$swarm_expected_updates}
swarm_wandb_model_tag=${SWARM_WANDB_MODEL_TAG:-1.7b}
swarm_controller_wandb_mode=${SWARM_CONTROLLER_WANDB_MODE:-online}
swarm_pytorch_cuda_alloc_conf=${SWARM_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
swarm_final_sync_margin=${SWARM_FINAL_SYNC_MARGIN:-2700}
swarm_pulse_mode=${SWARM_PULSE_MODE:-$(jq -r '.runtime.online_evaluation_mode // "full"' "$swarm_curriculum_artifact")}
swarm_communication_eval_turns=${SWARM_COMMUNICATION_EVAL_TURNS:-$(jq -r '.runtime.online_eval_remaining_turns // 2' "$swarm_curriculum_artifact")}
swarm_mirror_interval_steps=${SWARM_MIRROR_INTERVAL_STEPS:-1}
swarm_trainer_gpu_ids=${SWARM_TRAINER_GPU_IDS:-0}
swarm_inference_gpu_ids=${SWARM_INFERENCE_GPU_IDS:-1,2,3}
swarm_rollout_ports=${SWARM_ROLLOUT_PORTS:-8001,8002,8003}
IFS=',' read -r -a swarm_trainer_gpus <<< "$swarm_trainer_gpu_ids"
IFS=',' read -r -a swarm_inference_gpus <<< "$swarm_inference_gpu_ids"
IFS=',' read -r -a swarm_ports <<< "$swarm_rollout_ports"
if (( ${#swarm_trainer_gpus[@]} < 1 || ${#swarm_inference_gpus[@]} < 1 )); then
  echo "trainer and inference GPU partitions must both be non-empty" >&2
  exit 1
fi
if (( ${#swarm_inference_gpus[@]} != ${#swarm_ports[@]} )); then
  echo "SWARM_INFERENCE_GPU_IDS and SWARM_ROLLOUT_PORTS must have equal lengths" >&2
  exit 1
fi
swarm_base_url_args=()
swarm_trainer_gpu_args=()
swarm_inference_gpu_args=()
for swarm_gpu in "${swarm_trainer_gpus[@]}"; do
  if ! [[ "$swarm_gpu" =~ ^[0-9]+$ ]]; then
    echo "trainer GPU IDs must be non-negative integers" >&2
    exit 1
  fi
  swarm_trainer_gpu_args+=(--trainer-gpu-id "$swarm_gpu")
done
for swarm_gpu in "${swarm_inference_gpus[@]}"; do
  if ! [[ "$swarm_gpu" =~ ^[0-9]+$ ]]; then
    echo "inference GPU IDs must be non-negative integers" >&2
    exit 1
  fi
  swarm_inference_gpu_args+=(--inference-gpu-id "$swarm_gpu")
done
for swarm_port in "${swarm_ports[@]}"; do
  if ! [[ "$swarm_port" =~ ^[1-9][0-9]{0,4}$ ]] || (( swarm_port > 65535 )); then
    echo "rollout ports must be integers between 1 and 65535" >&2
    exit 1
  fi
  swarm_base_url_args+=(--base-url "http://127.0.0.1:$swarm_port")
done
printf -v swarm_base_url_command '%q ' "${swarm_base_url_args[@]}"
swarm_initial_policy_args=()
swarm_initial_policy_mirror_args=()
if [[ -n "${SWARM_INITIAL_POLICY_ADAPTER_MANIFEST:-}" ]]; then
  if [[ ! -f "$SWARM_INITIAL_POLICY_ADAPTER_MANIFEST" ]]; then
    echo "SWARM_INITIAL_POLICY_ADAPTER_MANIFEST is not a file" >&2
    exit 1
  fi
  swarm_initial_policy_args=(
    --initial-policy-adapter-manifest "$SWARM_INITIAL_POLICY_ADAPTER_MANIFEST"
  )
  swarm_initial_policy_mirror_args=(
    --artifact "$SWARM_INITIAL_POLICY_ADAPTER_MANIFEST"
  )
fi
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

export PYTHONPATH=$SWARM_REPO_ROOT/experiments/swarm_arena
if [[ "$swarm_pulse_mode" != full ]]; then
  echo "staged budget admission currently supports only full 192-row pulses" >&2
  exit 1
fi
"$swarm_uv" "${swarm_uv_args[@]}" python \
  "$SWARM_REPO_ROOT/experiments/swarm_arena/scripts/preflight_staged_budget.py" \
  --profile "$SWARM_OPERATIONAL_PROFILE" \
  --expected-updates "$swarm_expected_updates" --interval "$swarm_checkpoint_interval" \
  --deadline-epoch "$SWARM_DEADLINE_EPOCH" \
  --inference-config "$swarm_inference_config" --trainer-config "$SWARM_RUN_DIR/trainer.toml" \
  --topology "$swarm_trainer_gpu_ids/$swarm_inference_gpu_ids" \
  --gpu-model "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)" \
  --final-sync-seconds "$swarm_final_sync_margin" --output "$SWARM_RUN_DIR/BUDGET_PREFLIGHT.json"
swarm_barrier_timeout=$(jq -r .checkpoint_barrier_timeout_seconds "$SWARM_RUN_DIR/BUDGET_PREFLIGHT.json")
swarm_pulse_wait_timeout=$(jq -r .pulse_wait_timeout_seconds "$SWARM_RUN_DIR/BUDGET_PREFLIGHT.json")

for swarm_port in "${swarm_ports[@]}"; do
  curl -fsS "http://127.0.0.1:$swarm_port/health" >/dev/null
done

export PYTHONPATH=$SWARM_REPO_ROOT/experiments/swarm_arena
"$swarm_uv" "${swarm_uv_args[@]}" python \
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
  "${swarm_initial_policy_args[@]}" \
  --model "$SWARM_MODEL" \
  --expected-public-base-repo "$SWARM_PUBLIC_BASE_REPO" \
  --expected-public-adapter-repo "$SWARM_PUBLIC_ADAPTER_REPO" \
  "${swarm_base_url_args[@]}" \
  "${swarm_trainer_gpu_args[@]}" \
  "${swarm_inference_gpu_args[@]}" \
  --expected-updates "$swarm_expected_updates" \
  --checkpoint-interval "$swarm_checkpoint_interval" \
  --shared-return-credit-assignment "$swarm_credit_assignment" \
  --shared-return-replicas "$swarm_shared_return_replicas" \
  --action-prompt-profile "$swarm_action_prompt_profile"

swarm_plan_sha=$(jq -r .production_plan_sha256 "$SWARM_RUN_DIR/PREFLIGHT.json")
mkdir -p "$SWARM_RUN_DIR/logs" "$swarm_eval_root" "$swarm_rescore_dir"

# A run is not safe to leave unattended until the recovery repository is
# anonymously readable.  Verify that before starting any new optimizer work.
"$swarm_uv" "${swarm_uv_args[@]}" python \
  "$SWARM_REPO_ROOT/experiments/swarm_arena/scripts/run_live_artifact_mirror.py" \
  --repo-id "$SWARM_LIVE_HF_REPO" \
  --run-id "$SWARM_RUN_ID" \
  --run-dir "$SWARM_RUN_DIR" \
  --deadline-epoch "$SWARM_DEADLINE_EPOCH" \
  --final-sync-margin "$swarm_final_sync_margin" \
  --artifact "$SWARM_PRODUCTION_PLAN" \
  --artifact "$SWARM_RUNTIME_CERTIFICATE" \
  --artifact "$swarm_curriculum_artifact" \
  "${swarm_initial_policy_mirror_args[@]}" \
  --preflight-only \
  > "$SWARM_RUN_DIR/logs/mirror-preflight.log" 2>&1

for swarm_role in trainer rescore pulses wandb mirror controller profile; do
  if tmux has-session -t "$swarm_session_prefix-$swarm_role" 2>/dev/null; then
    echo "refusing to reuse tmux session $swarm_session_prefix-$swarm_role" >&2
    exit 1
  fi
done

tmux new-session -d -s "$swarm_session_prefix-trainer" \
  "cd $SWARM_REPO_ROOT && export CUDA_VISIBLE_DEVICES=$swarm_trainer_gpu_ids PYTORCH_CUDA_ALLOC_CONF=$swarm_pytorch_cuda_alloc_conf && exec ${swarm_uv_command}torchrun --standalone --nproc-per-node=${#swarm_trainer_gpus[@]} .venv/bin/trainer @ $SWARM_RUN_DIR/trainer.toml > $SWARM_RUN_DIR/logs/trainer.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-rescore" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec ${swarm_uv_command}python experiments/swarm_arena/scripts/run_lag_zero_rescore_worker.py --root $swarm_rescore_dir --snapshot-manifest $swarm_rescore_dir/current_snapshots.json --production-plan-sha256 $swarm_plan_sha > $SWARM_RUN_DIR/logs/rescore.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-pulses" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec ${swarm_uv_command}python experiments/swarm_arena/scripts/supervise_staged_role.py --run-dir $SWARM_RUN_DIR --role pulses -- experiments/swarm_arena/scripts/run_staged_pulses.py --repo-root $SWARM_REPO_ROOT --run-dir $SWARM_RUN_DIR --production-plan $SWARM_PRODUCTION_PLAN --barrier-dir $swarm_barrier_dir --eval-root $swarm_eval_root --data-dir $swarm_data_dir ${swarm_base_url_command}--expected-updates $swarm_expected_updates --interval $swarm_checkpoint_interval --wait-timeout $swarm_pulse_wait_timeout --evaluation-mode $swarm_pulse_mode --communication-remaining-turns $swarm_communication_eval_turns --stage-gates $swarm_stage_gates ${swarm_multipair_args[*]} > $SWARM_RUN_DIR/logs/pulses.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-wandb" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec ${swarm_uv_command}python experiments/swarm_arena/scripts/log_live_rl_wandb.py --progress $SWARM_RUN_DIR/live_rl_progress.json --eval-root $swarm_eval_root --expected-updates $swarm_expected_updates --finish-marker $swarm_eval_root/COMPLETE --project swarm-arena-rl --group $swarm_wandb_group --run-name $SWARM_RUN_ID-controller --run-id $SWARM_RUN_ID-controller-v1 --tag $swarm_wandb_model_tag --tag causal-communication --tag development ${swarm_wandb_mode_arg[*]} --compact-artifact $SWARM_PRODUCTION_PLAN --compact-artifact $swarm_curriculum_artifact > $SWARM_RUN_DIR/logs/wandb.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-mirror" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec ${swarm_uv_command}python experiments/swarm_arena/scripts/run_live_artifact_mirror.py --repo-id $SWARM_LIVE_HF_REPO --run-id $SWARM_RUN_ID --run-dir $SWARM_RUN_DIR --deadline-epoch $SWARM_DEADLINE_EPOCH --final-sync-margin $swarm_final_sync_margin --compact-interval-steps $swarm_mirror_interval_steps --artifact $SWARM_PRODUCTION_PLAN --artifact $SWARM_RUNTIME_CERTIFICATE --artifact $swarm_curriculum_artifact ${swarm_initial_policy_mirror_args[*]} > $SWARM_RUN_DIR/logs/mirror.log 2>&1"

sleep 15
if ! tmux has-session -t "$swarm_session_prefix-trainer" 2>/dev/null; then
  echo "trainer exited during startup" >&2
  exit 1
fi

tmux new-session -d -s "$swarm_session_prefix-controller" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec ${swarm_uv_command}python experiments/swarm_arena/scripts/supervise_staged_role.py --run-dir $SWARM_RUN_DIR --role controller -- experiments/swarm_arena/scripts/run_live_rl.py --output-dir $SWARM_RUN_DIR --trainer-config $SWARM_RUN_DIR/trainer.toml --inference-config $swarm_inference_config --data-dir $swarm_data_dir --task-data-version v4 --tokenizer $SWARM_MODEL --initial-adapter $SWARM_INITIAL_ADAPTER ${swarm_initial_policy_args[*]} ${swarm_base_url_command}--actor vllm --run-id $SWARM_RUN_ID --source-commit $swarm_source_commit --base-revision $SWARM_BASE_REVISION --initial-policy-revision $SWARM_INITIAL_POLICY_REVISION --credit-estimator shared_return --shared-return-replicas $swarm_shared_return_replicas --shared-return-credit-assignment $swarm_credit_assignment --shared-return-action-prompt-profile $swarm_action_prompt_profile --production-plan $SWARM_PRODUCTION_PLAN --async-rescore-dir $swarm_rescore_dir --async-rescore-timeout 600 --steps $swarm_expected_updates --groups-per-step 4 --scenario-source curriculum --curriculum-split train --target-swap-sender-retries $swarm_target_swap_sender_retries --update-timeout 1200 --checkpoint-barrier-dir $swarm_barrier_dir --checkpoint-barrier-interval $swarm_checkpoint_interval --checkpoint-barrier-timeout $swarm_barrier_timeout > $SWARM_RUN_DIR/logs/controller.log 2>&1"

tmux new-session -d -s "$swarm_session_prefix-profile" \
  "cd $SWARM_REPO_ROOT && export PYTHONPATH=$PYTHONPATH && exec ${swarm_uv_command}python experiments/swarm_arena/scripts/summarize_runtime_profile.py --progress $SWARM_RUN_DIR/live_rl_progress.json --trainer-gpus ${#swarm_trainer_gpus[@]} --inference-gpus ${#swarm_inference_gpus[@]} --minimum-updates 3 --wait-timeout $swarm_pulse_wait_timeout --output $SWARM_RUN_DIR/audit/runtime_profile.json > $SWARM_RUN_DIR/logs/runtime-profile.log 2>&1"

sleep 10
for swarm_role in trainer rescore pulses wandb mirror controller; do
  if ! tmux has-session -t "$swarm_session_prefix-$swarm_role" 2>/dev/null; then
    echo "$swarm_role exited during launch; stopping the new training stack" >&2
    for swarm_cleanup in trainer rescore pulses wandb mirror controller profile; do
      tmux kill-session -t "$swarm_session_prefix-$swarm_cleanup" 2>/dev/null || true
    done
    exit 1
  fi
done

echo "started $SWARM_RUN_ID at source $swarm_source_commit with plan $swarm_plan_sha"
