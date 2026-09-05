#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 MODEL CHECKPOINT_ROOT OUTPUT_ROOT STEP [STEP ...]" >&2
  exit 2
fi

swarm_model=$1
swarm_checkpoint_root=$2
swarm_output_root=$3
shift 3

swarm_repo_root=$(git rev-parse --show-toplevel)
swarm_uv_bin=${SWARM_UV_BIN:-uv}
swarm_arena_root="${swarm_repo_root}/experiments/swarm_arena"
swarm_eval_runtime=${SWARM_EVAL_RUNTIME:-}
if [[ -z "${swarm_eval_runtime}" ]]; then
  echo "SWARM_EVAL_RUNTIME must name an isolated directory containing peft==0.19.1 and accelerate==1.13.0" >&2
  exit 2
fi
if [[ ${swarm_checkpoint_root} != /* ]]; then
  swarm_checkpoint_root="${swarm_repo_root}/${swarm_checkpoint_root}"
fi
if [[ ${swarm_output_root} != /* ]]; then
  swarm_output_root="${swarm_repo_root}/${swarm_output_root}"
fi
for swarm_step in "$@"; do
  swarm_adapter="${swarm_checkpoint_root}/step_${swarm_step}/lora_adapters"
  swarm_output="${swarm_output_root}/step_${swarm_step}"
  if [[ ! -f "${swarm_adapter}/adapter_config.json" ]]; then
    echo "missing adapter: ${swarm_adapter}" >&2
    exit 1
  fi
  mkdir -p "${swarm_output}"
  PYTHONPATH="${swarm_eval_runtime}:${swarm_arena_root}${PYTHONPATH:+:$PYTHONPATH}" \
    "${swarm_uv_bin}" run --project "${swarm_repo_root}" --no-sync python \
    "${swarm_arena_root}/scripts/score_regressions.py" \
    --model "${swarm_model}" \
    --adapter "${swarm_adapter}" \
    --batch-size 16 \
    --output-dir "${swarm_output}"
done
