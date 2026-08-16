#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/blog-rl-run
eval_repo=/workspace/blog-rl-eval-6d6fe88e
run_a=/workspace/runs/rl-v4-ablate-lr1e5-mix211-12e0c461
run_b=/workspace/runs/rl-v4-ablate-lr1e5-mix233-12e0c461
plan_b=/workspace/ablation-configs/plan_commheavy_233.json
plan_b_sha=614f71605d6e063cb6a26a2c10ce0a7fe46a22f95cb9139c51db72bc7603d840
out=/workspace/runs/rl-v4-ablation-pulse-offsets-6d6fe88e
uv=/root/.local/bin/uv
export PYTHONPATH="$repo/experiments/swarm_arena"
mkdir -p "$out/logs" "$out/configs"
exec >>"$out/logs/followup.log" 2>&1

progress_count() {
  local run=$1
  if test -s "$run/live_rl_progress.json"; then
    jq length "$run/live_rl_progress.json"
  else
    echo 0
  fi
}

wait_for_updates() {
  local run=$1 expected=$2 controller=$3
  while true; do
    local count
    count=$(progress_count "$run")
    echo "$(date -u +%FT%TZ) $controller updates=$count/$expected"
    if test "$count" -ge "$expected"; then
      return 0
    fi
    if ! tmux has-session -t "$controller" 2>/dev/null; then
      echo "controller $controller exited before $expected updates"
      return 1
    fi
    sleep 30
  done
}

if test "${EVAL_ONLY:-0}" != 1; then
  wait_for_updates "$run_a" 12 ablate-controller
  tmux kill-session -t ablate-trainer 2>/dev/null || true
  tmux kill-session -t ablate-rescore 2>/dev/null || true
  tmux kill-session -t ablate-health 2>/dev/null || true
  sleep 15

  mkdir -p "$run_b/logs"
  if tmux has-session -t ablate-b-controller 2>/dev/null; then
    echo "$(date -u +%FT%TZ) Variant B already running; resuming watcher"
  else
    tmux new-session -d -s ablate-b-trainer \
      "cd $repo && export CUDA_VISIBLE_DEVICES=0 && exec $uv run torchrun --standalone --nproc-per-node=1 .venv/bin/trainer @ $run_b/trainer.toml > $run_b/logs/trainer.log 2>&1"
    tmux new-session -d -s ablate-b-rescore \
      "cd $repo && export PYTHONPATH=$repo/experiments/swarm_arena && exec $uv run python experiments/swarm_arena/scripts/run_lag_zero_rescore_worker.py --root $run_b/control/rescore --snapshot-manifest $run_b/control/rescore/current_snapshots.json --production-plan-sha256 $plan_b_sha > $run_b/logs/rescore.log 2>&1"
    sleep 15
    tmux new-session -d -s ablate-b-controller \
      "cd $repo && export PYTHONPATH=$repo/experiments/swarm_arena && exec $uv run python experiments/swarm_arena/scripts/run_live_rl.py --output-dir $run_b --trainer-config $run_b/trainer.toml --inference-config experiments/swarm_arena/configs/inference_1_7b_l40s.toml --data-dir experiments/swarm_arena/data/rl_v4 --task-data-version v4 --tokenizer /workspace/models/qwen3-1.7b-70d244c --initial-adapter /workspace/artifacts/warmstart-1.7b-step320 --base-url http://127.0.0.1:8001 --base-url http://127.0.0.1:8002 --base-url http://127.0.0.1:8003 --actor vllm --run-id rl-v4-ablate-lr1e5-mix233-12e0c461 --source-commit 12e0c461a28c3d0311d0353ab1ed45bcffb0b569 --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --initial-policy-revision 534522a8f3ff3489b1dd8318dc8e533e51264cde --credit-estimator shared_return --shared-return-replicas 4 --production-plan $plan_b --async-rescore-dir $run_b/control/rescore --async-rescore-timeout 600 --steps 8 --groups-per-step 8 --scenario-source curriculum --curriculum-split train --update-timeout 1200 > $run_b/logs/controller.log 2>&1"
  fi

  wait_for_updates "$run_b" 8 ablate-b-controller
  tmux kill-session -t ablate-b-trainer 2>/dev/null || true
  tmux kill-session -t ablate-b-rescore 2>/dev/null || true
fi

revision_for_run() {
  local run=$1 step=$2
  jq -r --argjson target "$step" \
    '.[] | select(.step == ($target - 1)) | .policy_revision' \
    "$run/live_rl_progress.json"
}

load_candidate() {
  local run=$1 step=$2 prefix=$3 port=$4
  for name in blue-0 blue-1 blue-2 blue-3 current-template; do
    curl -fsS -X POST "http://127.0.0.1:$port/v1/unload_lora_adapter" \
      -H 'Content-Type: application/json' -d "{\"lora_name\":\"$name\"}" >/dev/null 2>&1 || true
  done
  for role in 0 1 2 3; do
    local name="$prefix-blue-$role"
    local adapter_path="$run/run_blue_$role/broadcasts/step_$step"
    if test -d "$run/exports/step_$step/blue-$role"; then
      adapter_path="$run/exports/step_$step/blue-$role"
    fi
    curl -fsS -X POST "http://127.0.0.1:$port/v1/unload_lora_adapter" \
      -H 'Content-Type: application/json' -d "{\"lora_name\":\"$name\"}" >/dev/null 2>&1 || true
    curl -fsS -X POST "http://127.0.0.1:$port/v1/load_lora_adapter" \
      -H 'Content-Type: application/json' \
      -d "{\"lora_name\":\"$name\",\"lora_path\":\"$adapter_path\"}" >/dev/null
  done
}

write_eval_config() {
  local path=$1 prefix=$2 port=$3 revision=$4
  jq -n \
    --arg base_url "http://127.0.0.1:$port/v1" \
    --arg prefix "$prefix" \
    --arg revision "$revision" \
    '{
      base_url: $base_url,
      candidate: {revision: $revision, models: [0,1,2,3] | map($prefix + "-blue-" + tostring)},
      baseline: {revision: "2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b", models: ["sft-opponent","sft-opponent","sft-opponent","sft-opponent"]},
      opponents: [
        {id:"base", revision:"70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", models:["/workspace/models/qwen3-1.7b-70d244c","/workspace/models/qwen3-1.7b-70d244c","/workspace/models/qwen3-1.7b-70d244c","/workspace/models/qwen3-1.7b-70d244c"]},
        {id:"sft", revision:"2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b", models:["sft-opponent","sft-opponent","sft-opponent","sft-opponent"]},
        {id:"historical_league", revision:"1004e012cd96a6377006c334d997825e3ebb25828b482a4644b7149a823d873a", models:["historical-opponent","historical-opponent","historical-opponent","historical-opponent"]}
      ]
    }' > "$path"
}

run_pulse() {
  local label=$1 run=$2 step=$3 prefix=$4 port=$5 offset=$6
  local revision
  revision=$(revision_for_run "$run" "$step")
  load_candidate "$run" "$step" "$prefix" "$port"
  write_eval_config "$out/configs/$label.json" "$prefix" "$port" "$revision"
  cd "$repo"
  PYTHONPATH="$eval_repo/experiments/swarm_arena" "$uv" run python \
    "$eval_repo/experiments/swarm_arena/scripts/run_final_eval_development.py" \
    --config "$out/configs/$label.json" \
    --data-dir "$eval_repo/experiments/swarm_arena/data/rl_v3" \
    --output-dir "$out/$label" \
    --ordinary-cases 1 --ordinary-offset "$offset" \
    --curriculum-pairs 1 --curriculum-offset "$offset" \
    >"$out/logs/$label.log" 2>&1
}

run_pulse variant-a-step12 "$run_a" 12 ablate-a12 8001 6 & pulse_a12=$!
run_pulse variant-b-step4 "$run_b" 4 ablate-b4 8002 6 & pulse_b4=$!
run_pulse variant-a-step8 "$run_a" 8 ablate-a8 8003 6 & pulse_a8=$!
wait "$pulse_a12"
wait "$pulse_b4"
wait "$pulse_a8"

jq -n \
  --slurpfile a12 "$out/variant-a-step12/summary.json" \
  --slurpfile b4 "$out/variant-b-step4/summary.json" \
  --slurpfile a8 "$out/variant-a-step8/summary.json" \
  '{variant_a_step12:$a12[0], variant_b_step4:$b4[0], variant_a_step8:$a8[0]}' > "$out/comparison.json"
echo "$(date -u +%FT%TZ) ablation pair and pulses complete"
