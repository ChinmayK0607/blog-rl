#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
SUMMARY="$LOG_ROOT/symbolic_rl_supervisor.status"
MAX_ATTEMPTS=${SUPERVISOR_MAX_ATTEMPTS:-1}
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-30}
MONITOR_PRUNE_INTERVAL_SECONDS=${MONITOR_PRUNE_INTERVAL_SECONDS:-180}

mkdir -p "$LOG_ROOT"

session_exists() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"
}

kill_session_if_exists() {
  local session=$1
  if session_exists "$session"; then
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
}

timestamp() {
  date --utc +%Y-%m-%dT%H:%M:%SZ
}

run_one() {
  local session=$1
  local config=$2
  local run_root=$3
  local tmp_root=$4
  local log_file=$5
  local status_file="$run_root/supervisor_exit_status"
  local attempts=0
  local status=1

  mkdir -p "$run_root" "$tmp_root"
  rm -f "$status_file"

  while (( attempts < MAX_ATTEMPTS )); do
    attempts=$((attempts + 1))
    echo "$(timestamp) start session=$session attempt=$attempts config=$config" | tee -a "$SUMMARY"

    kill_session_if_exists "${session}_monitor"
    rm -rf "$tmp_root"
    mkdir -p "$tmp_root"

    tmux new-session -d -s "$session" \
      "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$tmp_root WANDB_MODE=online; uv run --no-sync rl @ $config >$log_file 2>&1; status=\$?; echo \$status >$status_file; rm -rf $tmp_root; exit \$status'"

    tmux new-session -d -s "${session}_monitor" \
      "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=$MONITOR_INTERVAL_SECONDS MONITOR_PRUNE_INTERVAL_SECONDS=$MONITOR_PRUNE_INTERVAL_SECONDS exec scripts/monitor_symbolic_rl_run.sh $session $run_root'"

    while session_exists "$session"; do
      sleep 30
    done

    python3 "$ROOT/scripts/prune_symbolic_checkpoints.py" --require-val "$run_root" >>"$log_file" 2>&1 || true

    if [[ -f "$status_file" ]]; then
      status=$(<"$status_file")
    else
      status=127
    fi

    echo "$(timestamp) exit session=$session attempt=$attempts status=$status" | tee -a "$SUMMARY"
    if [[ "$status" == 0 ]]; then
      return 0
    fi
  done

  echo "$(timestamp) giving_up session=$session attempts=$attempts last_status=$status" | tee -a "$SUMMARY"
  return "$status"
}

cd "$ROOT"
echo "$(timestamp) supervisor_start max_attempts=$MAX_ATTEMPTS" | tee -a "$SUMMARY"

COMPACT_ROOT=/home/ubuntu/semi/artifacts/rl-qwen3-instruct-compacted-grpo-pilot-v1
if [[ "${SKIP_COMPACTED_GRPO:-0}" == 1 ]]; then
  echo "$(timestamp) skip session=rl-qwen3-instruct-compacted-grpo-long-v1 reason=explicit_skip" | tee -a "$SUMMARY"
  compact_status=0
elif [[ ! -f "$COMPACT_ROOT/checkpoints/step_50/trainer/.metadata" && ! -f "$COMPACT_ROOT/checkpoints/step_60/trainer/.metadata" ]]; then
  echo "$(timestamp) skip session=rl-qwen3-instruct-compacted-grpo-long-v1 reason=no_stable_resume_checkpoint" | tee -a "$SUMMARY"
  compact_status=0
else
  run_one \
    rl-qwen3-instruct-compacted-grpo-long-v1 \
    environments/symbolic_tool_calling_v1/rl_qwen3_instruct_compacted_grpo_long.toml \
    "$COMPACT_ROOT" \
    /tmp/semi-rl-compact-long-v1 \
    /home/ubuntu/semi/artifacts/logs/rl-qwen3-instruct-compacted-grpo-long-v1.log
  compact_status=$?
fi

run_one \
  rl-qwen3-instruct-segment-normalized-grpo-long-v1 \
  environments/symbolic_tool_calling_v1/rl_qwen3_instruct_segment_normalized_grpo_long.toml \
  /home/ubuntu/semi/artifacts/rl-qwen3-instruct-segment-normalized-grpo-pilot-v1 \
  /tmp/semi-rl-seg-norm-long-v1 \
  /home/ubuntu/semi/artifacts/logs/rl-qwen3-instruct-segment-normalized-grpo-long-v1.log
segment_status=$?

run_one \
  rl-qwen3-instruct-compacted-ppo-pilot-v1 \
  environments/symbolic_tool_calling_v1/rl_qwen3_instruct_compacted_ppo_pilot.toml \
  /home/ubuntu/semi/artifacts/rl-qwen3-instruct-compacted-ppo-pilot-v1 \
  /tmp/semi-rl-ppo-pilot-v1 \
  /home/ubuntu/semi/artifacts/logs/rl-qwen3-instruct-compacted-ppo-pilot-v1.log
ppo_status=$?

echo "$(timestamp) supervisor_done compact=$compact_status segment=$segment_status ppo=$ppo_status" | tee -a "$SUMMARY"
if [[ "$compact_status" == 0 && "$segment_status" == 0 && "$ppo_status" == 0 ]]; then
  exit 0
fi
exit 1
