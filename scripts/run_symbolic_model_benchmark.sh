#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 MODEL GPUS PORT TP DP DP_RPC_PORT CONFIG RUN_ROOT" >&2
  exit 2
fi

MODEL=$1
GPUS=$2
PORT=$3
TP=$4
DP=$5
DP_RPC_PORT=$6
CONFIG=$7
RUN_ROOT=$8
ROOT=/home/ubuntu/semi/prime-rl
mkdir -p "$RUN_ROOT"
cd "$ROOT"

export UV_PROJECT_ENVIRONMENT=.venv
export EMPTY=local
export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_DP_MASTER_PORT="$DP_RPC_PORT"

cleanup() {
  if [[ -n "${INFERENCE_PID:-}" ]] && kill -0 "$INFERENCE_PID" 2>/dev/null; then
    kill -INT "$INFERENCE_PID" 2>/dev/null || true
    wait "$INFERENCE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

uv run --no-sync inference \
  --model.name "$MODEL" \
  --model.max-model-len 32768 \
  --model.tool-call-parser hermes \
  --parallel.tp "$TP" \
  --parallel.dp "$DP" \
  --vllm-extra "{\"data_parallel_rpc_port\":$DP_RPC_PORT}" \
  --gpu-memory-utilization 0.85 \
  --server.port "$PORT" \
  --output-dir "$RUN_ROOT/inference" \
  >"$RUN_ROOT/inference.log" 2>&1 &
INFERENCE_PID=$!

for _ in $(seq 1 240); do
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

uv run --no-sync eval @ "$CONFIG" --output-dir "$RUN_ROOT/eval" >"$RUN_ROOT/eval.console.log" 2>&1
