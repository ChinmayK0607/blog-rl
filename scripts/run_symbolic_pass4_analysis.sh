#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
RUN_ROOT=/home/ubuntu/semi/artifacts/qwen3-4b-instruct-long-high-pass4-v1

"$ROOT/scripts/run_symbolic_model_benchmark.sh" \
  Qwen/Qwen3-4B-Instruct-2507 0,1,2,3,4,5,6,7 8030 2 4 29750 \
  "$ROOT/environments/symbolic_tool_calling_v1/eval_qwen3_instruct_long_high_pass4.toml" \
  "$RUN_ROOT"

cd "$ROOT"
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync symbolic-pass-at-k \
  "$RUN_ROOT/eval/results.jsonl" "$RUN_ROOT/pass_at_4" --expected-k 4 \
  >"$RUN_ROOT/pass_at_4.console.log" 2>&1
