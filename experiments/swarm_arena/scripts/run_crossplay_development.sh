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

run_pair() {
  local output_dir=$1
  local conditions=$2
  local blue_history=$3
  local red_history=$4
  "$uv_bin" run --project "$project_root" --no-sync python -m swarm_ctf_eval.crossplay_eval \
    --blue-base-url "$adapter_url" --blue-model "$adapter_model" --blue-artifact-id "$adapter_artifact" \
    --red-base-url "$base_url" --red-model "$base_model" --red-artifact-id "$base_artifact" \
    --split development --cases 8 --conditions "$conditions" --swap-sides --resume \
    --blue-history-window "$blue_history" --red-history-window "$red_history" \
    --output-dir "$output_dir"
}

run_pair "$results_root/pair_h3_h3" \
  "generated:generated,dropped:generated,sender_shuffled:generated,delayed:generated" 3 3
run_pair "$results_root/pair_focal_h0_opponent_h3" "generated:generated" 0 3
run_pair "$results_root/pair_focal_h3_opponent_h0" "generated:generated" 3 0

"$uv_bin" run --project "$project_root" --no-sync python -m swarm_ctf_eval.crossplay_eval \
  --blue-base-url "$base_url" --blue-model "$base_model" --blue-artifact-id "$base_artifact" \
  --red-base-url "$base_url" --red-model "$base_model" --red-artifact-id "$base_artifact" \
  --split development --cases 4 --conditions generated:generated --resume \
  --output-dir "$results_root/selfplay_base"

"$uv_bin" run --project "$project_root" --no-sync python -m swarm_ctf_eval.crossplay_eval \
  --blue-base-url "$adapter_url" --blue-model "$adapter_model" --blue-artifact-id "$adapter_artifact" \
  --red-base-url "$adapter_url" --red-model "$adapter_model" --red-artifact-id "$adapter_artifact" \
  --split development --cases 4 --conditions generated:generated --resume \
  --output-dir "$results_root/selfplay_adapter"

"$uv_bin" run --project "$project_root" --no-sync python -m swarm_ctf_eval.crossplay_compare \
  --normal-dir "$results_root/pair_h3_h3" \
  --focal-no-history-dir "$results_root/pair_focal_h0_opponent_h3" \
  --opponent-no-history-dir "$results_root/pair_focal_h3_opponent_h0" \
  --focal-policy "$adapter_model" \
  --output "$results_root/history_effects.json"
