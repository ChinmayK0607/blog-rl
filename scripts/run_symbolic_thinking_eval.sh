#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/semi/prime-rl
RUN_ROOT=/home/ubuntu/semi/artifacts/qwen3-4b-thinking-short-12x8-v1
PORT=8011
mkdir -p "$RUN_ROOT"
cd "$ROOT"

export UV_PROJECT_ENVIRONMENT=.venv
export EMPTY=local
export CUDA_VISIBLE_DEVICES=0

cleanup() {
  if [[ -n "${INFERENCE_PID:-}" ]] && kill -0 "$INFERENCE_PID" 2>/dev/null; then
    kill -INT "$INFERENCE_PID" 2>/dev/null || true
    wait "$INFERENCE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

uv run --no-sync inference \
  --model.name Qwen/Qwen3-4B-Thinking-2507 \
  --model.max-model-len 32768 \
  --model.tool-call-parser hermes \
  --gpu-memory-utilization 0.85 \
  --server.port "$PORT" \
  --output-dir "$RUN_ROOT/inference" \
  >"$RUN_ROOT/inference.log" 2>&1 &
INFERENCE_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
    echo "inference exited before becoming healthy" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

uv run --no-sync eval \
  @ environments/symbolic_tool_calling_v1/eval_qwen3_thinking_short.toml \
  --output-dir "$RUN_ROOT/eval" \
  >"$RUN_ROOT/eval.console.log" 2>&1
