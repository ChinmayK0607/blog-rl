#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 MODEL WEIGHTS_ROOT DATASET RESULTS_ROOT STEP [STEP ...]" >&2
  exit 2
fi

model=$1
weights_root=$2
dataset=$3
results_root=$4
shift 4
uv_bin=${SWARM_UV_BIN:-uv}
project_root=${SWARM_PROJECT_ROOT:-/root/blog-rl}
arena_root="$project_root/experiments/swarm_arena"

for step in "$@"; do
  adapter="$weights_root/step_$step/lora_adapters"
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    echo "missing adapter checkpoint: $adapter" >&2
    exit 1
  fi
  PYTHONPATH="$arena_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$uv_bin" run --project "$project_root" --no-sync python \
    "$arena_root/scripts/score_warmstart_v3.py" \
    --model "$model" --adapter "$adapter" --dataset "$dataset" --batch-size 16 \
    --output-dir "$results_root/validation/step_$step"
  PYTHONPATH="$arena_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$uv_bin" run --project "$project_root" --no-sync python \
    "$arena_root/scripts/score_regressions.py" \
    --model "$model" --adapter "$adapter" --suite v1 --batch-size 16 \
    --output-dir "$results_root/regression_v1/step_$step"
  PYTHONPATH="$arena_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$uv_bin" run --project "$project_root" --no-sync python \
    "$arena_root/scripts/score_regressions.py" \
    --model "$model" --adapter "$adapter" --suite v2 --batch-size 16 \
    --output-dir "$results_root/regression_v2/step_$step"
done

PYTHONPATH="$arena_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$uv_bin" run --project "$project_root" --no-sync python \
  -m swarm_ctf_eval.warm_start_selection_v3 \
  --validation-root "$results_root/validation" \
  --regression-v1-root "$results_root/regression_v1" \
  --regression-v2-root "$results_root/regression_v2" \
  --base-v1-rows "$results_root/regression_v1/base/rows.jsonl" \
  --base-v2-rows "$results_root/regression_v2/base/rows.jsonl" \
  --output "$results_root/selection.json"
