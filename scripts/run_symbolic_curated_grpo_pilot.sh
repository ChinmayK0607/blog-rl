#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
RUN_ROOT=/home/ubuntu/semi/artifacts/rl-qwen3-instruct-curated-6i2t-pilot-v3
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
TMP_ROOT=/home/ubuntu/semi/tmp/rl-qwen3-instruct-curated-6i2t-pilot-v3
mkdir -p "$LOG_ROOT"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
cd "$ROOT"

export UV_PROJECT_ENVIRONMENT=.venv
export EMPTY=local
export TMPDIR="$TMP_ROOT"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

set +e
uv run --no-sync rl @ environments/symbolic_tool_calling_v1/rl_qwen3_instruct_curated_6i2t_pilot.toml \
  >"$LOG_ROOT/rl-qwen3-instruct-curated-6i2t-pilot-v3.log" 2>&1
status=$?
set -e
python3 scripts/prune_symbolic_checkpoints.py --require-val "$RUN_ROOT" >>"$LOG_ROOT/rl-qwen3-instruct-curated-6i2t-pilot-v3.log" 2>&1 || true
exit "$status"
