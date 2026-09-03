#!/usr/bin/env bash
set -euo pipefail

: "${SWARM_REPO_ROOT:?set SWARM_REPO_ROOT}"
: "${SWARM_INFERENCE_CONFIG:?set SWARM_INFERENCE_CONFIG}"
: "${SWARM_RUN_DIR:?set SWARM_RUN_DIR}"

swarm_uv=${SWARM_UV:-/root/.local/bin/uv}
swarm_uv_args=(run --frozen --extra flash-attn)
printf -v swarm_uv_command '%q ' "$swarm_uv" "${swarm_uv_args[@]}"
swarm_session_prefix=${SWARM_SESSION_PREFIX:-swarm-staged}
swarm_trainer_gpu_ids=${SWARM_TRAINER_GPU_IDS:-0}
swarm_inference_gpu_ids=${SWARM_INFERENCE_GPU_IDS:-1,2,3}
swarm_rollout_ports=${SWARM_ROLLOUT_PORTS:-8001,8002,8003}
swarm_rpc_port_base=${SWARM_RPC_PORT_BASE:-14000}
swarm_startup_timeout=${SWARM_INFERENCE_STARTUP_TIMEOUT:-1200}
IFS=',' read -r -a swarm_inference_gpus <<< "$swarm_inference_gpu_ids"
IFS=',' read -r -a swarm_trainer_gpus <<< "$swarm_trainer_gpu_ids"
IFS=',' read -r -a swarm_ports <<< "$swarm_rollout_ports"

if (( ${#swarm_inference_gpus[@]} < 1 )); then
  echo "inference GPU partition cannot be empty" >&2
  exit 1
fi
if (( ${#swarm_inference_gpus[@]} != ${#swarm_ports[@]} )); then
  echo "SWARM_INFERENCE_GPU_IDS and SWARM_ROLLOUT_PORTS must have equal lengths" >&2
  exit 1
fi
if ! [[ "$swarm_rpc_port_base" =~ ^[1-9][0-9]*$ ]]; then
  echo "SWARM_RPC_PORT_BASE must be a positive integer" >&2
  exit 1
fi
if ! [[ "$swarm_startup_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "SWARM_INFERENCE_STARTUP_TIMEOUT must be a positive integer" >&2
  exit 1
fi

swarm_visible_gpus=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
swarm_assigned_gpus=("${swarm_trainer_gpus[@]}" "${swarm_inference_gpus[@]}")
if (( ${#swarm_assigned_gpus[@]} != swarm_visible_gpus )); then
  echo "runtime topology must assign every visible GPU exactly once" >&2
  exit 1
fi
for swarm_left in "${!swarm_assigned_gpus[@]}"; do
  swarm_gpu=${swarm_assigned_gpus[$swarm_left]}
  if ! [[ "$swarm_gpu" =~ ^[0-9]+$ ]] || (( swarm_gpu >= swarm_visible_gpus )); then
    echo "runtime topology contains an invalid GPU index" >&2
    exit 1
  fi
  for ((swarm_right = swarm_left + 1; swarm_right < ${#swarm_assigned_gpus[@]}; swarm_right++)); do
    if [[ "$swarm_gpu" == "${swarm_assigned_gpus[$swarm_right]}" ]]; then
      echo "runtime topology assigns a GPU more than once" >&2
      exit 1
    fi
  done
done
for swarm_left in "${!swarm_ports[@]}"; do
  for ((swarm_right = swarm_left + 1; swarm_right < ${#swarm_ports[@]}; swarm_right++)); do
    if [[ "${swarm_ports[$swarm_left]}" == "${swarm_ports[$swarm_right]}" ]]; then
      echo "rollout ports cannot contain duplicates" >&2
      exit 1
    fi
  done
done

mkdir -p "$SWARM_RUN_DIR/logs" "$SWARM_RUN_DIR/control/inference-cache"
swarm_started_sessions=()
swarm_pool_ready=0
swarm_cleanup_new_pool() {
  for swarm_started_session in "${swarm_started_sessions[@]}"; do
    tmux kill-session -t "$swarm_started_session" 2>/dev/null || true
  done
}
swarm_cleanup_on_exit() {
  swarm_status=$?
  if (( swarm_pool_ready == 0 )); then
    swarm_cleanup_new_pool
  fi
  return "$swarm_status"
}
trap swarm_cleanup_on_exit EXIT
trap 'swarm_cleanup_new_pool; exit 130' INT TERM
for swarm_index in "${!swarm_inference_gpus[@]}"; do
  swarm_gpu=${swarm_inference_gpus[$swarm_index]}
  swarm_port=${swarm_ports[$swarm_index]}
  if ! [[ "$swarm_gpu" =~ ^[0-9]+$ ]]; then
    echo "inference GPU IDs must be non-negative integers" >&2
    exit 1
  fi
  if ! [[ "$swarm_port" =~ ^[1-9][0-9]{0,4}$ ]] || (( swarm_port > 65535 )); then
    echo "rollout ports must be integers between 1 and 65535" >&2
    exit 1
  fi
  swarm_session="$swarm_session_prefix-inference-$swarm_gpu"
  if tmux has-session -t "$swarm_session" 2>/dev/null; then
    echo "refusing to reuse tmux session $swarm_session" >&2
    exit 1
  fi
done

for swarm_index in "${!swarm_inference_gpus[@]}"; do
  swarm_gpu=${swarm_inference_gpus[$swarm_index]}
  swarm_port=${swarm_ports[$swarm_index]}
  swarm_rpc_port=$((swarm_rpc_port_base + swarm_index))
  swarm_session="$swarm_session_prefix-inference-$swarm_gpu"
  swarm_cache="$SWARM_RUN_DIR/control/inference-cache/gpu-$swarm_gpu"
  mkdir -p "$swarm_cache/torch" "$swarm_cache/triton" "$swarm_cache/vllm"
  tmux new-session -d -s "$swarm_session" \
    "cd $SWARM_REPO_ROOT && export CUDA_VISIBLE_DEVICES=$swarm_gpu TORCHINDUCTOR_CACHE_DIR=$swarm_cache/torch TRITON_CACHE_DIR=$swarm_cache/triton VLLM_CACHE_ROOT=$swarm_cache/vllm && exec ${swarm_uv_command}inference @ $SWARM_INFERENCE_CONFIG --server.port $swarm_port --data-parallel-rpc-port $swarm_rpc_port > $SWARM_RUN_DIR/logs/inference-$swarm_gpu.log 2>&1"
  swarm_started_sessions+=("$swarm_session")
done

swarm_deadline=$(( $(date +%s) + swarm_startup_timeout ))
for swarm_port in "${swarm_ports[@]}"; do
  until curl -fsS "http://127.0.0.1:$swarm_port/health" >/dev/null 2>&1; do
    if (( $(date +%s) >= swarm_deadline )); then
      echo "inference pool did not become healthy before timeout" >&2
      exit 1
    fi
    sleep 5
  done
done

swarm_pool_ready=1
trap - INT TERM
echo "started ${#swarm_inference_gpus[@]} isolated inference servers on GPUs $swarm_inference_gpu_ids"
