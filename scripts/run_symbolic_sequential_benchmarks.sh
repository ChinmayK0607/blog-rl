#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
RUNNER="$ROOT/scripts/run_symbolic_model_benchmark.sh"
GPUS=0,1,2,3,4,5,6,7

"$RUNNER" \
  Qwen/Qwen3-4B-Instruct-2507 "$GPUS" 8020 2 4 29550 \
  "$ROOT/environments/symbolic_tool_calling_v1/eval_qwen3_instruct_high_budget.toml" \
  /home/ubuntu/semi/artifacts/qwen3-4b-instruct-high-budget-12x8-v4

"$RUNNER" \
  Qwen/Qwen3-4B-Thinking-2507 "$GPUS" 8021 2 4 29650 \
  "$ROOT/environments/symbolic_tool_calling_v1/eval_qwen3_thinking_high_budget.toml" \
  /home/ubuntu/semi/artifacts/qwen3-4b-thinking-high-budget-12x8-v4
