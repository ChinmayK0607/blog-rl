#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
CURRENT_SEGMENT_SESSION=rl-qwen3-instruct-segment-normalized-grpo-pilot-v1

session_exists() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"
}

wait_for_session_exit() {
  local session=$1
  while session_exists "$session"; do
    sleep 60
  done
}

run_one() {
  local session=$1
  local config=$2
  local run_root=$3
  local tmp_root=$4
  local log_file=$5
  local status_file="$run_root/long_queue_exit_status"

  rm -rf "$tmp_root"
  mkdir -p "$tmp_root" "$LOG_ROOT" "$run_root"
  rm -f "$status_file"

  tmux new-session -d -s "$session" \
    "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$tmp_root WANDB_MODE=online; uv run --no-sync rl @ $config >$log_file 2>&1; status=\$?; echo \$status >$status_file; rm -rf $tmp_root; exit \$status'"
  tmux new-session -d -s "${session}_monitor" \
    "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=30 exec scripts/monitor_symbolic_rl_run.sh $session $run_root'"

  while session_exists "$session"; do
    sleep 60
  done
  python3 scripts/prune_symbolic_checkpoints.py --require-val "$run_root" >>"$log_file" 2>&1 || true
  if [[ ! -f "$status_file" ]] || [[ "$(<"$status_file")" != 0 ]]; then
    echo "long run failed: $session" >&2
    return 1
  fi
}

cd "$ROOT"
wait_for_session_exit "$CURRENT_SEGMENT_SESSION"

run_one \
  rl-qwen3-instruct-compacted-grpo-long-v1 \
  environments/symbolic_tool_calling_v1/rl_qwen3_instruct_compacted_grpo_long.toml \
  /home/ubuntu/semi/artifacts/rl-qwen3-instruct-compacted-grpo-pilot-v1 \
  /tmp/semi-rl-compact-long-v1 \
  /home/ubuntu/semi/artifacts/logs/rl-qwen3-instruct-compacted-grpo-long-v1.log

run_one \
  rl-qwen3-instruct-segment-normalized-grpo-long-v1 \
  environments/symbolic_tool_calling_v1/rl_qwen3_instruct_segment_normalized_grpo_long.toml \
  /home/ubuntu/semi/artifacts/rl-qwen3-instruct-segment-normalized-grpo-pilot-v1 \
  /tmp/semi-rl-seg-norm-long-v1 \
  /home/ubuntu/semi/artifacts/logs/rl-qwen3-instruct-segment-normalized-grpo-long-v1.log
