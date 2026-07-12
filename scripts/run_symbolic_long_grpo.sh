#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
RUN_NAME=rl-qwen3-instruct-full-grpo-long-v1
RUN_ROOT=/home/ubuntu/semi/artifacts/$RUN_NAME
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
TMP_ROOT=/home/ubuntu/semi/tmp/$RUN_NAME
CONFIG=environments/symbolic_tool_calling_v1/rl_qwen3_instruct_curated_6i2t_long_grpo.toml

mkdir -p "$LOG_ROOT"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
cd "$ROOT"

export UV_PROJECT_ENVIRONMENT=.venv
export EMPTY=local
export TMPDIR="$TMP_ROOT"
export WANDB_MODE=online

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

set +e
uv run --no-sync rl @ "$CONFIG" >"$LOG_ROOT/$RUN_NAME.log" 2>&1
status=$?
set -e
python3 scripts/prune_symbolic_checkpoints.py --require-val "$RUN_ROOT" >>"$LOG_ROOT/$RUN_NAME.log" 2>&1 || true
exit "$status"
