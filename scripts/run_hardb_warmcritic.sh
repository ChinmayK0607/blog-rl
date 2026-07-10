#!/usr/bin/env bash
# Phase-B follow-up: PPO with a *pretrained* (warm-started) value head, to test
# whether the critic cold-start is what hobbles compacted PPO on hard tasks.
# Waits for the running 3-arm Phase-B queue to finish, pretrains the value head
# offline on arm-1's early rollouts (policy ~= the shared full-GRPO warm start),
# then trains compacted PPO warm-critic with the same watchdog/monitor/prune.
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
ART=/home/ubuntu/semi/artifacts
LOG_ROOT=$ART/logs
SUMMARY="$LOG_ROOT/hardb_warmcritic.status"
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

cd "$ROOT"
export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local

echo "$(timestamp) warmcritic_wait_for_queue" | tee -a "$SUMMARY"
# Wait until the 3-arm Phase-B queue is fully done (supervisor + arm sessions gone).
while session_exists semi_hardb_supervisor || session_exists hardb-full-grpo-v1 \
   || session_exists hardb-compacted-grpo-v1 || session_exists hardb-compacted-ppo-v1; do
  sleep 60
done
cleanup_gpu_procs
echo "$(timestamp) queue_done -> pretrain value head" | tee -a "$SUMMARY"

# ---- 1. Offline value-head pretraining (single GPU, no-grad forward + ridge) ----
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/pretrain_ppo_value_head.py \
  --model "$ART/cmp-full-grpo-v1/weights/step_90" \
  --rollouts "$ART/hardb-full-grpo-v1/run_default/rollouts" \
  --steps 1-25 --token-budget 384 --max-rollouts 400 --max-seq-len 16384 \
  --out "$WARM_INIT" >"$LOG_ROOT/hardb-warmcritic-pretrain.log" 2>&1
if [[ ! -f "$WARM_INIT" ]]; then
  echo "$(timestamp) PRETRAIN_FAILED (see hardb-warmcritic-pretrain.log)" | tee -a "$SUMMARY"
  exit 1
fi
echo "$(timestamp) pretrain_done -> launch warm-critic PPO" | tee -a "$SUMMARY"
cleanup_gpu_procs

# ---- 2. Warm-critic PPO training arm ----
session=hardb-compacted-ppo-warmcritic-v1
config="$ENV_DIR/rl_qwen3_instruct_hardb_compacted_ppo_warmcritic.toml"
run_root="$ART/hardb-compacted-ppo-warmcritic-v1"
tmp_root=/tmp/semi-rl-hardb-warmcritic
log_file="$LOG_ROOT/hardb-compacted-ppo-warmcritic-v1.log"
status_file="$run_root/supervisor_exit_status"
mkdir -p "$run_root"; rm -f "$status_file"
kill_session_if_exists "${session}_monitor"; rm -rf "$tmp_root"; mkdir -p "$tmp_root"

echo "$(timestamp) start session=$session config=$config" | tee -a "$SUMMARY"
tmux new-session -d -s "$session" \
  "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$tmp_root WANDB_MODE=online; uv run --no-sync rl @ $config >$log_file 2>&1; status=\$?; echo \$status >$status_file; rm -rf $tmp_root; exit \$status'"
tmux new-session -d -s "${session}_monitor" \
  "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=$MONITOR_INTERVAL_SECONDS MONITOR_PRUNE_INTERVAL_SECONDS=$MONITOR_PRUNE_INTERVAL_SECONDS exec scripts/monitor_symbolic_rl_run.sh $session $run_root'"

last_step="" last_progress_ts=$(date +%s)
while session_exists "$session"; do
  sleep 30
  cur_step=$(latest_step "$log_file")
  if [[ -n "$cur_step" && "$cur_step" != "$last_step" ]]; then last_step="$cur_step"; last_progress_ts=$(date +%s); fi
  now=$(date +%s)
  if (( now - last_progress_ts > STALL_TIMEOUT_SECONDS )); then
    echo "$(timestamp) stall_detected session=$session last_step=${last_step:-none}" | tee -a "$SUMMARY"
    kill_session_if_exists "$session"; cleanup_gpu_procs; break
  fi
done
kill_session_if_exists "${session}_monitor"; cleanup_gpu_procs
python3 "$ROOT/scripts/prune_symbolic_checkpoints.py" --require-val "$run_root" >>"$log_file" 2>&1 || true
status=127; [[ -f "$status_file" ]] && status=$(<"$status_file")
echo "$(timestamp) warmcritic_done status=$status" | tee -a "$SUMMARY"
exit "$status"
