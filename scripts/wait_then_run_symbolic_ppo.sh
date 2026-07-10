#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
WAIT_SESSION=${WAIT_SESSION:-rl-qwen3-instruct-segment-normalized-grpo-long-v1}
RUN_NAME=${RUN_NAME:-rl-qwen3-instruct-compacted-ppo-pilot-v1}
RUN_ROOT=/home/ubuntu/semi/artifacts/$RUN_NAME
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
TMP_ROOT=/tmp/semi-rl-ppo-pilot-v1
CONFIG=environments/symbolic_tool_calling_v1/rl_qwen3_instruct_compacted_ppo_pilot.toml

session_exists() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"
}

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

while session_exists "$WAIT_SESSION"; do
  sleep 60
done

if session_exists "$RUN_NAME"; then
  exit 0
fi

rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
rm -f "$RUN_ROOT/supervisor_exit_status"

tmux new-session -d -s "$RUN_NAME" \
  "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$TMP_ROOT WANDB_MODE=online; uv run --no-sync rl @ $CONFIG >$LOG_ROOT/$RUN_NAME.log 2>&1; status=\$?; echo \$status >$RUN_ROOT/supervisor_exit_status; rm -rf $TMP_ROOT; exit \$status'"
tmux new-session -d -s "${RUN_NAME}_monitor" \
  "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=30 MONITOR_PRUNE_INTERVAL_SECONDS=180 exec scripts/monitor_symbolic_rl_run.sh $RUN_NAME $RUN_ROOT'"
