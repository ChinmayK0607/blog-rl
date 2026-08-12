#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 BASE_URL BASE_MODEL BASE_ARTIFACT ADAPTER_URL ADAPTER_MODEL ADAPTER_ARTIFACT RESULTS_ROOT" >&2
  exit 2
fi

base_url=$1
base_model=$2
base_artifact=$3
adapter_url=$4
adapter_model=$5
adapter_artifact=$6
results_root=$7
uv_bin=${SWARM_UV_BIN:-uv}
project_root=${SWARM_PROJECT_ROOT:-/root/blog-rl}
arena_root="$project_root/experiments/swarm_arena"
export PYTHONPATH="$arena_root${PYTHONPATH:+:$PYTHONPATH}"

# Four base self-play games: absolute base capability without a scripted opponent.
"$uv_bin" run --project "$project_root" --no-sync python -m swarm_ctf_eval.crossplay_eval \
  --blue-base-url "$base_url" --blue-model "$base_model" --blue-artifact-id "$base_artifact" \
  --red-base-url "$base_url" --red-model "$base_model" --red-artifact-id "$base_artifact" \
  --split development --cases 4 --conditions generated:generated --resume \
  --output-dir "$results_root/selfplay_base"

# Four adapter self-play games: protocol warm-start capability and stability.
"$uv_bin" run --project "$project_root" --no-sync python -m swarm_ctf_eval.crossplay_eval \
  --blue-base-url "$adapter_url" --blue-model "$adapter_model" --blue-artifact-id "$adapter_artifact" \
  --red-base-url "$adapter_url" --red-model "$adapter_model" --red-artifact-id "$adapter_artifact" \
  --split development --cases 4 --conditions generated:generated --resume \
  --output-dir "$results_root/selfplay_adapter"

# Sixteen paired cross-policy games: four seeds x two message conditions x two
# side assignments. The adapter is the focal policy, so --swap-sides moves its
# dropped-message condition with it rather than changing which team is ablated.
"$uv_bin" run --project "$project_root" --no-sync python -m swarm_ctf_eval.crossplay_eval \
  --blue-base-url "$adapter_url" --blue-model "$adapter_model" --blue-artifact-id "$adapter_artifact" \
  --red-base-url "$base_url" --red-model "$base_model" --red-artifact-id "$base_artifact" \
  --split development --cases 4 \
  --conditions generated:generated,dropped:generated --swap-sides --resume \
  --output-dir "$results_root/pair_adapter_vs_base"

