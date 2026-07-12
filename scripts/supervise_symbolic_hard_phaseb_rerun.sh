#!/usr/bin/env bash
# Phase-B rerun after the disk-full crash. full_grpo already completed (weights
# kept); this reruns the remaining arms and adds the warm-critic PPO arm:
#   1) compacted_grpo   2) compacted_ppo (cold critic)   3) compacted_ppo warm-critic
# All warm-started from cmp-full-grpo/step_90, token_budget=384, on the hard
# curriculum, with eval fixed (tool_call_parser=hermes) and keep_last=2 to bound
# disk. Transient DCP checkpoints + weight broadcasts of each finished arm are
# purged before the next arm starts (the HF `weights` exports are preserved).
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
ART=/home/ubuntu/semi/artifacts
LOG_ROOT=$ART/logs
SUMMARY="$LOG_ROOT/symbolic_hard_phaseb_rerun.status"
ENV_DIR="$ROOT/environments/symbolic_tool_calling_v1"
WARM_INIT="$ART/hardb-ppo-warmcritic/value_head.safetensors"
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-30}
MONITOR_PRUNE_INTERVAL_SECONDS=${MONITOR_PRUNE_INTERVAL_SECONDS:-180}
STALL_TIMEOUT_SECONDS=${STALL_TIMEOUT_SECONDS:-1500}

mkdir -p "$LOG_ROOT"
session_exists() { tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"; }
kill_session_if_exists() { session_exists "$1" && tmux kill-session -t "$1" 2>/dev/null || true; }
timestamp() { date --utc +%Y-%m-%dT%H:%M:%SZ; }
cleanup_gpu_procs() {
  pkill -9 -f "$ROOT/.venv/bin/torchrun" 2>/dev/null || true
  pkill -9 -f "$ROOT/.venv/bin/python3" 2>/dev/null || true
  sleep 6
}
latest_step() { grep -oE 'Step [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1; }
# Drop transient trainer state (DCP checkpoints + fs weight broadcasts); keep HF weights.
clean_transient() {
  local d=$1
  rm -rf "$d/checkpoints" "$d/run_default/broadcasts" 2>/dev/null || true
  echo "$(timestamp) clean_transient $d free=$(df -h / | awk 'NR==2{print $4}')" | tee -a "$SUMMARY"
}

run_one() {
  local session=$1 config=$2 run_root=$3 tmp_root=$4 log_file=$5
  local status_file="$run_root/supervisor_exit_status"
  mkdir -p "$run_root"; rm -f "$status_file"
  echo "$(timestamp) start session=$session config=$config free=$(df -h / | awk 'NR==2{print $4}')" | tee -a "$SUMMARY"
  kill_session_if_exists "${session}_monitor"; rm -rf "$tmp_root"; mkdir -p "$tmp_root"
  tmux new-session -d -s "$session" \
    "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$tmp_root WANDB_MODE=online; uv run --no-sync rl @ $config >$log_file 2>&1; status=\$?; echo \$status >$status_file; rm -rf $tmp_root; exit \$status'"
  tmux new-session -d -s "${session}_monitor" \
    "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=$MONITOR_INTERVAL_SECONDS MONITOR_PRUNE_INTERVAL_SECONDS=$MONITOR_PRUNE_INTERVAL_SECONDS exec scripts/monitor_symbolic_rl_run.sh $session $run_root'"
  local last_step="" last_progress_ts; last_progress_ts=$(date +%s)
  while session_exists "$session"; do
    sleep 30
    local cur_step; cur_step=$(latest_step "$log_file")
    if [[ -n "$cur_step" && "$cur_step" != "$last_step" ]]; then last_step="$cur_step"; last_progress_ts=$(date +%s); fi
    local now; now=$(date +%s)
    if (( now - last_progress_ts > STALL_TIMEOUT_SECONDS )); then
      echo "$(timestamp) stall_detected session=$session last_step=${last_step:-none}" | tee -a "$SUMMARY"
      kill_session_if_exists "$session"; cleanup_gpu_procs; break
    fi
  done
  kill_session_if_exists "${session}_monitor"; cleanup_gpu_procs
  python3 "$ROOT/scripts/prune_symbolic_checkpoints.py" --require-val "$run_root" >>"$log_file" 2>&1 || true
  local status=127; [[ -f "$status_file" ]] && status=$(<"$status_file")
  echo "$(timestamp) exit session=$session status=$status" | tee -a "$SUMMARY"
  return "$status"
}

cd "$ROOT"
export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local
echo "$(timestamp) phaseb_rerun_start free=$(df -h / | awk 'NR==2{print $4}')" | tee -a "$SUMMARY"

run_one hardb-compacted-grpo-v1 \
  "$ENV_DIR/rl_qwen3_instruct_hardb_compacted_grpo.toml" \
  "$ART/hardb-compacted-grpo-v1" /tmp/semi-rl-hardb-compacted-grpo \
  "$LOG_ROOT/hardb-compacted-grpo-v1.log"
s_cmp=$?
clean_transient "$ART/hardb-compacted-grpo-v1"

run_one hardb-compacted-ppo-v1 \
  "$ENV_DIR/rl_qwen3_instruct_hardb_compacted_ppo.toml" \
  "$ART/hardb-compacted-ppo-v1" /tmp/semi-rl-hardb-compacted-ppo \
  "$LOG_ROOT/hardb-compacted-ppo-v1.log"
s_ppo=$?
clean_transient "$ART/hardb-compacted-ppo-v1"

# Ensure the pretrained value head exists (regenerate if missing).
if [[ ! -f "$WARM_INIT" ]]; then
  echo "$(timestamp) value head missing -> pretraining" | tee -a "$SUMMARY"
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/pretrain_ppo_value_head.py \
    --model "$ART/cmp-full-grpo-v1/weights/step_90" \
    --rollouts "$ART/hardb-full-grpo-v1/run_default/rollouts" \
    --steps 1-25 --token-budget 384 --max-rollouts 400 --max-seq-len 16384 \
    --out "$WARM_INIT" >"$LOG_ROOT/hardb-warmcritic-pretrain.log" 2>&1
  cleanup_gpu_procs
fi

run_one hardb-compacted-ppo-warmcritic-v1 \
  "$ENV_DIR/rl_qwen3_instruct_hardb_compacted_ppo_warmcritic.toml" \
  "$ART/hardb-compacted-ppo-warmcritic-v1" /tmp/semi-rl-hardb-warmcritic \
  "$LOG_ROOT/hardb-compacted-ppo-warmcritic-v1.log"
s_warm=$?
clean_transient "$ART/hardb-compacted-ppo-warmcritic-v1"

echo "$(timestamp) phaseb_rerun_done compacted=$s_cmp ppo=$s_ppo warmcritic=$s_warm free=$(df -h / | awk 'NR==2{print $4}')" | tee -a "$SUMMARY"
[[ "$s_cmp" == 0 && "$s_ppo" == 0 && "$s_warm" == 0 ]] && exit 0
exit 1
