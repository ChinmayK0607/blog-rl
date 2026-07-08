#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
RUN_ROOT=/home/ubuntu/semi/artifacts/symbolic-scaled-v1
LOG_ROOT=/home/ubuntu/semi/artifacts/logs
mkdir -p "$LOG_ROOT"
cd "$ROOT"

export UV_PROJECT_ENVIRONMENT=.venv

uv run --no-sync symbolic-benchmark \
  "$RUN_ROOT" \
  --config environments/symbolic_tool_calling_v1/scaled_pipeline_config.json \
  --repo "$ROOT" \
  >"$LOG_ROOT/symbolic-scaled-v1.log" 2>&1
