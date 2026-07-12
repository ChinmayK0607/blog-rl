#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
CONTROL_SESSION=semi_rl_full_grpo_long_v1

session_exists() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"
}

while session_exists "$CONTROL_SESSION"; do
  sleep 60
done

run_one() {
  local name=$1
  local config=$2
  local run_root=/home/ubuntu/semi/artifacts/$name
  # vLLM/ZMQ creates ipc:// sockets under TMPDIR with a UUID suffix.  Long
  # artifact-style names can exceed sockaddr_un.sun_path (107 bytes), so keep
  # runtime tmp roots deliberately short.
  local tmp_root=/tmp/semi-rl-${name##rl-qwen3-instruct-}
  rm -rf "$tmp_root"
  mkdir -p "$tmp_root" "$LOG_ROOT"
  local status_file="$run_root/queue_exit_status"
  rm -f "$status_file"
  tmux new-session -d -s "$name" \
    "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$tmp_root WANDB_MODE=online; uv run --no-sync rl @ $config >$LOG_ROOT/$name.log 2>&1; status=\$?; mkdir -p $run_root; python3 scripts/prune_symbolic_checkpoints.py --require-val $run_root >>$LOG_ROOT/$name.log 2>&1 || true; echo \$status >$status_file; exit \$status'"
  tmux new-session -d -s "${name}_monitor" \
    "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=30 exec scripts/monitor_symbolic_rl_run.sh $name $run_root'"
  while session_exists "$name"; do
    sleep 60
  done
  if [[ ! -f "$status_file" ]] || [[ "$(<"$status_file")" != 0 ]]; then
    echo "run failed: $name" >&2
    return 1
  fi
  rm -rf "$tmp_root"
}

cd "$ROOT"
run_one \
  rl-qwen3-instruct-compacted-grpo-pilot-v1 \
  environments/symbolic_tool_calling_v1/rl_qwen3_instruct_compacted_grpo_pilot.toml
run_one \
  rl-qwen3-instruct-segment-normalized-grpo-pilot-v1 \
  environments/symbolic_tool_calling_v1/rl_qwen3_instruct_segment_normalized_grpo_pilot.toml
