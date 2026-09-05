#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/blog-rl-run
eval_repo=/workspace/blog-rl-eval-6d6fe88e
uv=/root/.local/bin/uv
run=/workspace/runs/rl-v4-ablate-lr1e5-mix211-12e0c461
out=/workspace/runs/rl-v4-ablation-holdout-a8-shards-retry2-6d6fe88e
mkdir -p "$out/configs" "$out/logs"
exec >>"$out/logs/launcher.log" 2>&1

revision=$(jq -r '.[] | select(.step == 7) | .policy_revision' "$run/live_rl_progress.json")

load_candidate() {
  local port=$1
  for name in blue-0 blue-1 blue-2 blue-3 current-template ablate-a12-blue-0 ablate-a12-blue-1 ablate-a12-blue-2 ablate-a12-blue-3 ablate-b4-blue-0 ablate-b4-blue-1 ablate-b4-blue-2 ablate-b4-blue-3; do
    curl -fsS -X POST "http://127.0.0.1:$port/v1/unload_lora_adapter" \
      -H 'Content-Type: application/json' -d "{\"lora_name\":\"$name\"}" >/dev/null 2>&1 || true
  done
  for role in 0 1 2 3; do
    local name="ablate-a8-blue-$role"
    curl -fsS -X POST "http://127.0.0.1:$port/v1/unload_lora_adapter" \
      -H 'Content-Type: application/json' -d "{\"lora_name\":\"$name\"}" >/dev/null 2>&1 || true
    curl -fsS -X POST "http://127.0.0.1:$port/v1/load_lora_adapter" \
      -H 'Content-Type: application/json' \
      -d "{\"lora_name\":\"$name\",\"lora_path\":\"$run/exports/step_8/blue-$role\"}" >/dev/null
  done
}

write_config() {
  local port=$1 path=$2
  jq -n --arg base_url "http://127.0.0.1:$port/v1" --arg revision "$revision" \
    '{
      base_url:$base_url,
      candidate:{revision:$revision,models:["ablate-a8-blue-0","ablate-a8-blue-1","ablate-a8-blue-2","ablate-a8-blue-3"]},
      baseline:{revision:"2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b",models:["sft-opponent","sft-opponent","sft-opponent","sft-opponent"]},
      opponents:[
        {id:"base",revision:"70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",models:["/workspace/models/qwen3-1.7b-70d244c","/workspace/models/qwen3-1.7b-70d244c","/workspace/models/qwen3-1.7b-70d244c","/workspace/models/qwen3-1.7b-70d244c"]},
        {id:"sft",revision:"2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b",models:["sft-opponent","sft-opponent","sft-opponent","sft-opponent"]},
        {id:"historical_league",revision:"1004e012cd96a6377006c334d997825e3ebb25828b482a4644b7149a823d873a",models:["historical-opponent","historical-opponent","historical-opponent","historical-opponent"]}
      ]
    }' > "$path"
}

run_shard() {
  local port=$1
  local offset=$2
  local label="offset-$offset"
  cd "$repo"
  PYTHONPATH="$eval_repo/experiments/swarm_arena" "$uv" run python \
    "$eval_repo/experiments/swarm_arena/scripts/run_final_eval_development.py" \
    --config "$out/configs/port-$port.json" \
    --data-dir "$eval_repo/experiments/swarm_arena/data/rl_v3" \
    --output-dir "$out/$label" \
    --ordinary-cases 1 --ordinary-offset "$offset" \
    --curriculum-pairs 1 --curriculum-offset "$offset" \
    >"$out/logs/$label.log" 2>&1
}

for port in 8001 8002 8003; do
  load_candidate "$port"
  write_config "$port" "$out/configs/port-$port.json"
done

run_shard 8001 9 & shard_9=$!
run_shard 8002 10 & shard_10=$!
run_shard 8003 11 & shard_11=$!
wait "$shard_9"
wait "$shard_10"
wait "$shard_11"

mkdir -p "$out/combined"
for offset in 9 10 11; do
  sed '/^$/d' "$out/offset-$offset/rows.jsonl"
done > "$out/combined/rows.jsonl"

cd "$repo"
PYTHONPATH="$eval_repo/experiments/swarm_arena/scripts:$eval_repo/experiments/swarm_arena" "$uv" run python -c '
import json
from pathlib import Path
from run_final_eval_development import _summary
root = Path("/workspace/runs/rl-v4-ablation-holdout-a8-shards-retry2-6d6fe88e")
rows = [json.loads(line) for line in (root / "combined/rows.jsonl").read_text().splitlines() if line]
summary = _summary(rows)
(root / "combined/summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
' > "$out/logs/combined.log" 2>&1
echo "$(date -u +%FT%TZ) holdout complete"
