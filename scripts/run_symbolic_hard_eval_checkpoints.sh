#!/usr/bin/env bash
# Phase-A hard-task pass@4 eval of all four trained regime checkpoints on a
# depth-graded hard ladder (medium -> long -> xlong). Probes whether PPO's
# per-segment critic generalizes to deeper/sparser tasks better than the GRPO
# variants. Serves each checkpoint once, runs all bands, computes pass@4.
#
# Run ONLY when the 8 GPUs are free (no RL job active).
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
ART=/home/ubuntu/semi/artifacts
ENV_DIR="$ROOT/environments/symbolic_tool_calling_v1"
OUT_ROOT="$ART/hard-eval-phaseA-v1"
GPUS="0,1,2,3,4,5,6,7"
PORT=8055
TP=2
DP=4
DP_RPC_PORT=30075
BANDS=(medium long xlong)

# name:weights_path (val-representative exported HF checkpoint per arm)
CKPTS=(
  "full_grpo:$ART/cmp-full-grpo-v1/weights/step_90"
  "compacted_grpo:$ART/cmp-compacted-grpo-v1/weights/step_90"
  "segnorm_grpo:$ART/cmp-segnorm-grpo-v1/weights/step_60"
  "compacted_ppo:$ART/cmp-compacted-ppo-v1/weights/step_90"
)

export UV_PROJECT_ENVIRONMENT=.venv
export EMPTY=local
export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_DP_MASTER_PORT="$DP_RPC_PORT"
cd "$ROOT"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.log"
echo "$(date --utc +%FT%TZ) hard_eval_start" | tee -a "$SUMMARY"

INFERENCE_PID=""
kill_infer() {
  if [[ -n "$INFERENCE_PID" ]] && kill -0 "$INFERENCE_PID" 2>/dev/null; then
    kill -INT "$INFERENCE_PID" 2>/dev/null || true
    wait "$INFERENCE_PID" 2>/dev/null || true
  fi
  INFERENCE_PID=""
}
trap kill_infer EXIT INT TERM

for entry in "${CKPTS[@]}"; do
  name="${entry%%:*}"; model="${entry#*:}"
  echo "$(date --utc +%FT%TZ) serve name=$name model=$model" | tee -a "$SUMMARY"
  ckpt_root="$OUT_ROOT/$name"
  mkdir -p "$ckpt_root"

  uv run --no-sync inference \
    --model.name "$model" \
    --model.max-model-len 32768 \
    --model.tool-call-parser hermes \
    --parallel.tp "$TP" --parallel.dp "$DP" \
    --vllm-extra "{\"data_parallel_rpc_port\":$DP_RPC_PORT}" \
    --gpu-memory-utilization 0.85 \
    --server.port "$PORT" \
    --output-dir "$ckpt_root/inference" \
    >"$ckpt_root/inference.log" 2>&1 &
  INFERENCE_PID=$!

  healthy=0
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then healthy=1; break; fi
    if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then break; fi
    sleep 2
  done
  if [[ "$healthy" != 1 ]]; then
    echo "$(date --utc +%FT%TZ) UNHEALTHY name=$name (see $ckpt_root/inference.log)" | tee -a "$SUMMARY"
    kill_infer; continue
  fi

  for band in "${BANDS[@]}"; do
    cfg_tmpl="$ENV_DIR/eval_graded_hard_${band}_pass4.toml"
    cfg="$ckpt_root/eval_${band}.toml"
    sed "s#__MODEL__#${model}#g" "$cfg_tmpl" > "$cfg"
    run_root="$ckpt_root/$band"
    echo "$(date --utc +%FT%TZ) eval name=$name band=$band" | tee -a "$SUMMARY"
    uv run --no-sync eval @ "$cfg" --output-dir "$run_root/eval" \
      >"$run_root.eval.console.log" 2>&1
    uv run --no-sync symbolic-pass-at-k \
      "$run_root/eval/results.jsonl" "$run_root/pass_at_4" --expected-k 4 \
      >"$run_root.passk.console.log" 2>&1 || true
    # one-line result
    if [[ -f "$run_root/pass_at_4/summary.json" ]]; then
      python3 -c "import json;d=json.load(open('$run_root/pass_at_4/summary.json'));print('  -> groups=%s mean_pass_rate=%.3f mixed_frac=%.3f buckets=%s'%(d.get('num_groups'),d.get('mean_pass_rate',0),d.get('mixed_fraction',0),d.get('bucket_counts')))" | tee -a "$SUMMARY"
    fi
  done
  kill_infer
  sleep 5
done

echo "$(date --utc +%FT%TZ) hard_eval_done" | tee -a "$SUMMARY"
