#!/usr/bin/env bash
# Unified pass@4 of all four Phase-B arms' final (step_70) checkpoints on the SAME
# held-out hard val set (symbolic-hard-curriculum-v1/tasks_val.jsonl), identical
# harness + hermes tool parser. Fair cross-arm comparison incl. full_grpo (whose
# in-run val was disabled by the tool-parser bug).
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
ART=/home/ubuntu/semi/artifacts
ENV_DIR="$ROOT/environments/symbolic_tool_calling_v1"
OUT_ROOT="$ART/hardb-unified-passk-v1"
PORT=8055; TP=2; DP=4; DP_RPC_PORT=30075
CFG_TMPL="$ENV_DIR/eval_hardb_val_pass4.toml"

ARMS=(
  "full_grpo:$ART/hardb-full-grpo-v1/weights/step_70"
  "compacted_grpo:$ART/hardb-compacted-grpo-v1/weights/step_70"
  "compacted_ppo_cold:$ART/hardb-compacted-ppo-v1/weights/step_70"
  "compacted_ppo_warm:$ART/hardb-compacted-ppo-warmcritic-v1/weights/step_70"
)

export UV_PROJECT_ENVIRONMENT=.venv EMPTY=local CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_DP_MASTER_PORT="$DP_RPC_PORT"
cd "$ROOT"; mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.log"
echo "$(date --utc +%FT%TZ) unified_passk_start" | tee -a "$SUMMARY"

INFERENCE_PID=""
kill_infer(){ [[ -n "$INFERENCE_PID" ]] && kill -0 "$INFERENCE_PID" 2>/dev/null && { kill -INT "$INFERENCE_PID" 2>/dev/null||true; wait "$INFERENCE_PID" 2>/dev/null||true; }; INFERENCE_PID=""; }
trap kill_infer EXIT INT TERM

for entry in "${ARMS[@]}"; do
  name="${entry%%:*}"; model="${entry#*:}"
  root="$OUT_ROOT/$name"; mkdir -p "$root"
  if [[ ! -f "$model/model.safetensors" ]]; then
    echo "$(date --utc +%FT%TZ) SKIP $name (no checkpoint at $model)" | tee -a "$SUMMARY"; continue
  fi
  echo "$(date --utc +%FT%TZ) serve $name" | tee -a "$SUMMARY"
  uv run --no-sync inference \
    --model.name "$model" --model.max-model-len 32768 --model.tool-call-parser hermes \
    --parallel.tp "$TP" --parallel.dp "$DP" \
    --vllm-extra "{\"data_parallel_rpc_port\":$DP_RPC_PORT}" \
    --gpu-memory-utilization 0.85 --server.port "$PORT" \
    --output-dir "$root/inference" >"$root/inference.log" 2>&1 &
  INFERENCE_PID=$!
  healthy=0
  for _ in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
    kill -0 "$INFERENCE_PID" 2>/dev/null || break
    sleep 2
  done
  if [[ "$healthy" != 1 ]]; then echo "$(date --utc +%FT%TZ) UNHEALTHY $name" | tee -a "$SUMMARY"; kill_infer; continue; fi

  cfg="$root/eval.toml"; sed "s#__MODEL__#${model}#g" "$CFG_TMPL" > "$cfg"
  uv run --no-sync eval @ "$cfg" --output-dir "$root/eval" >"$root/eval.console.log" 2>&1
  rm -rf "$root/pass_at_4"
  uv run --no-sync symbolic-pass-at-k "$root/eval/results.jsonl" "$root/pass_at_4" --expected-k 4 \
    >"$root/passk.console.log" 2>&1 || true
  if [[ -f "$root/pass_at_4/summary.json" ]]; then
    python3 -c "import json;d=json.load(open('$root/pass_at_4/summary.json'));print('  -> $name: groups=%s mean_pass_rate=%.3f mixed_frac=%.3f buckets=%s'%(d.get('num_groups'),d.get('mean_pass_rate',0),d.get('mixed_fraction',0),d.get('bucket_counts')))" | tee -a "$SUMMARY"
  fi
  kill_infer; sleep 5
done
echo "$(date --utc +%FT%TZ) unified_passk_done" | tee -a "$SUMMARY"
