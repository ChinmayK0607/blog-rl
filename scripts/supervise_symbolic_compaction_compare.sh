#!/usr/bin/env bash
# GRPO-vs-PPO-under-compaction comparison queue.
# Runs the 4 regimes back-to-back (one 8-GPU job at a time) on the curriculum
# dataset with token_budget=384. Single bounded attempt each: advance the queue
# and preserve evidence rather than crash-fix-crash.
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
SUMMARY="$LOG_ROOT/symbolic_compaction_compare.status"
MAX_ATTEMPTS=${SUPERVISOR_MAX_ATTEMPTS:-1}
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-30}
MONITOR_PRUNE_INTERVAL_SECONDS=${MONITOR_PRUNE_INTERVAL_SECONDS:-180}
# If no new training "Step N" appears within this many seconds, treat the run as
# wedged (orchestrator rollout-loop stall / dead inference) and kill+advance.
STALL_TIMEOUT_SECONDS=${STALL_TIMEOUT_SECONDS:-1500}
ENV_DIR="$ROOT/environments/symbolic_tool_calling_v1"

mkdir -p "$LOG_ROOT"

session_exists() { tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"; }
kill_session_if_exists() { session_exists "$1" && tmux kill-session -t "$1" 2>/dev/null || true; }
timestamp() { date --utc +%Y-%m-%dT%H:%M:%SZ; }

# Kill only THIS project's training/inference procs; never touch other checkouts
# (e.g. /home/ubuntu/blogger/... or /home/ubuntu/ivllm/...).
cleanup_gpu_procs() {
  pkill -9 -f "$ROOT/.venv/bin/torchrun" 2>/dev/null || true
  pkill -9 -f "$ROOT/.venv/bin/python3" 2>/dev/null || true
  sleep 6
}

# Highest "Step N" seen in a run log (rollout progress heartbeat).
latest_step() {
  grep -oE 'Step [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1
}

run_one() {
  local session=$1 config=$2 run_root=$3 tmp_root=$4 log_file=$5
  local status_file="$run_root/supervisor_exit_status"
  local attempts=0 status=1
  mkdir -p "$run_root"
  rm -f "$status_file"
  while (( attempts < MAX_ATTEMPTS )); do
    attempts=$((attempts + 1))
    echo "$(timestamp) start session=$session attempt=$attempts config=$config" | tee -a "$SUMMARY"
    kill_session_if_exists "${session}_monitor"
    rm -rf "$tmp_root"; mkdir -p "$tmp_root"
    tmux new-session -d -s "$session" \
      "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$tmp_root WANDB_MODE=online; uv run --no-sync rl @ $config >$log_file 2>&1; status=\$?; echo \$status >$status_file; rm -rf $tmp_root; exit \$status'"
    tmux new-session -d -s "${session}_monitor" \
      "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=$MONITOR_INTERVAL_SECONDS MONITOR_PRUNE_INTERVAL_SECONDS=$MONITOR_PRUNE_INTERVAL_SECONDS exec scripts/monitor_symbolic_rl_run.sh $session $run_root'"
    # Wait for the run, but bail out if training progress stalls (wedged
    # orchestrator keeps the tmux session alive forever otherwise).
    local last_step="" last_progress_ts=$(date +%s)
    while session_exists "$session"; do
      sleep 30
      local cur_step; cur_step=$(latest_step "$log_file")
      if [[ -n "$cur_step" && "$cur_step" != "$last_step" ]]; then
        last_step="$cur_step"; last_progress_ts=$(date +%s)
      fi
      local now; now=$(date +%s)
      if (( now - last_progress_ts > STALL_TIMEOUT_SECONDS )); then
        echo "$(timestamp) stall_detected session=$session last_step=${last_step:-none} idle_s=$((now - last_progress_ts))" | tee -a "$SUMMARY"
        kill_session_if_exists "$session"
        cleanup_gpu_procs
        break
      fi
    done
    kill_session_if_exists "${session}_monitor"
    cleanup_gpu_procs
    python3 "$ROOT/scripts/prune_symbolic_checkpoints.py" --require-val "$run_root" >>"$log_file" 2>&1 || true
    if [[ -f "$status_file" ]]; then status=$(<"$status_file"); else status=127; fi
    echo "$(timestamp) exit session=$session attempt=$attempts status=$status" | tee -a "$SUMMARY"
    [[ "$status" == 0 ]] && return 0
  done
  echo "$(timestamp) giving_up session=$session attempts=$attempts last_status=$status" | tee -a "$SUMMARY"
  return "$status"
}

cd "$ROOT"
echo "$(timestamp) compare_supervisor_start max_attempts=$MAX_ATTEMPTS" | tee -a "$SUMMARY"

# Run 1 (full GRPO baseline) already completed cleanly on 2026-07-09
# (converged val 0.971 @ step 90). Skip to avoid re-burning GPU hours.
echo "$(timestamp) skip session=cmp-full-grpo-v1 reason=already_converged" | tee -a "$SUMMARY"
s_full=0

# Runs 2 & 3 already completed cleanly on 2026-07-10 (compacted GRPO peak val
# 0.96 to step 116; segment-normalized GRPO peak val 0.99 to step ~140). Skip.
echo "$(timestamp) skip session=cmp-compacted-grpo-v1 reason=already_converged" | tee -a "$SUMMARY"
s_cmp=0
echo "$(timestamp) skip session=cmp-segnorm-grpo-v1 reason=already_converged" | tee -a "$SUMMARY"
s_seg=0

run_one cmp-compacted-ppo-v1 \
  "$ENV_DIR/rl_qwen3_instruct_cmp_compacted_ppo.toml" \
  /home/ubuntu/semi/artifacts/cmp-compacted-ppo-v1 \
  /tmp/semi-rl-cmp-compacted-ppo \
  "$LOG_ROOT/cmp-compacted-ppo-v1.log"
s_ppo=$?

echo "$(timestamp) compare_supervisor_done full=$s_full compacted=$s_cmp segnorm=$s_seg ppo=$s_ppo" | tee -a "$SUMMARY"
[[ "$s_full" == 0 && "$s_cmp" == 0 && "$s_seg" == 0 && "$s_ppo" == 0 ]] && exit 0
exit 1
