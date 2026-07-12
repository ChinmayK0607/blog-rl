#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
RUN_NAME=rl-qwen3-instruct-compacted-ppo-pilot-v1
RUN_ROOT=/home/ubuntu/semi/artifacts/$RUN_NAME
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
TMP_ROOT=/tmp/semi-rl-ppo-pilot-v1
CONFIG=environments/symbolic_tool_calling_v1/rl_qwen3_instruct_compacted_ppo_pilot.toml
WAIT_SESSIONS=(
  semi_rl_compaction_long_queue
  rl-qwen3-instruct-compacted-grpo-long-v1
  rl-qwen3-instruct-segment-normalized-grpo-long-v1
)

session_exists() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"
}

cd "$ROOT"
mkdir -p "$LOG_ROOT" "$RUN_ROOT"

for session in "${WAIT_SESSIONS[@]}"; do
  while session_exists "$session"; do
    sleep 60
  done
done

rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
rm -f "$RUN_ROOT/queue_exit_status"

tmux new-session -d -s "$RUN_NAME" \
  "bash -lc 'cd $ROOT && export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local TMPDIR=$TMP_ROOT WANDB_MODE=online; uv run --no-sync rl @ $CONFIG >$LOG_ROOT/$RUN_NAME.log 2>&1; status=\$?; python3 scripts/prune_symbolic_checkpoints.py --require-val $RUN_ROOT >>$LOG_ROOT/$RUN_NAME.log 2>&1 || true; echo \$status >$RUN_ROOT/queue_exit_status; rm -rf $TMP_ROOT; exit \$status'"
tmux new-session -d -s "${RUN_NAME}_monitor" \
  "bash -lc 'cd $ROOT && MONITOR_INTERVAL_SECONDS=30 MONITOR_PRUNE_INTERVAL_SECONDS=180 exec scripts/monitor_symbolic_rl_run.sh $RUN_NAME $RUN_ROOT'"
