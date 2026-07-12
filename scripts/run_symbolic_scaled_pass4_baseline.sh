#!/usr/bin/env bash
# Frozen-policy pass@4 baseline + offline segment/CV analysis for the scaled
# datasets. Launch this ONLY when GPUs are free (no other 8-GPU RL job active).
# Runs both candidate difficulty bands (solvable-long and hard mixed) back to
# back on a single inference server, then computes compaction stats.
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
BENCH="$ROOT/scripts/run_symbolic_model_benchmark.sh"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
GPUS="0,1,2,3,4,5,6,7"
PORT=8043
TP=2
DP=4
DP_RPC_PORT=30070

run_cell() {
  local name=$1 config=$2
  local run_root=/home/ubuntu/semi/artifacts/qwen3-4b-instruct-${name}-pass4-v1
  echo "=== pass@4 baseline: ${name} ==="
  "$BENCH" "$MODEL" "$GPUS" "$PORT" "$TP" "$DP" "$DP_RPC_PORT" "$config" "$run_root"
  cd "$ROOT"
  UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync symbolic-pass-at-k \
    "$run_root/eval/results.jsonl" "$run_root/pass_at_4" --expected-k 4 \
    >"$run_root/pass_at_4.console.log" 2>&1 || true
  UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python "$ROOT/scripts/analyze_symbolic_compaction.py" \
    "$run_root" --token-budget 2048 --tokenizer "$MODEL" \
    --output "$run_root/compaction_analysis/summary_budget2048.json"
}

run_cell "scaled-solvable" \
  "$ROOT/environments/symbolic_tool_calling_v1/eval_qwen3_instruct_scaled_solvable_pass4.toml"
run_cell "scaled-mixed" \
  "$ROOT/environments/symbolic_tool_calling_v1/eval_qwen3_instruct_scaled_mixed_pass4.toml"

echo "=== done: inspect pass_at_4/summary.json and compaction_analysis/summary_budget2048.json in each run root ==="
