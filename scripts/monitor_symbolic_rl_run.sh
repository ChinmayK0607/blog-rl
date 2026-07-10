#!/usr/bin/env bash
set -uo pipefail

SESSION=${1:-semi_rl_curated_grpo_pilot_v2}
RUN_ROOT=${2:-/home/ubuntu/semi/artifacts/rl-qwen3-instruct-curated-6i2t-pilot-v2}
INTERVAL=${MONITOR_INTERVAL_SECONDS:-30}
PRUNE_INTERVAL=${MONITOR_PRUNE_INTERVAL_SECONDS:-300}
MONITOR_DIR="$RUN_ROOT/monitor"
EVENTS="$MONITOR_DIR/events.log"
FINAL="$MONITOR_DIR/final_snapshot.log"
PRUNE_LOG="$MONITOR_DIR/prune.log"
LAST_PRUNE=0
mkdir -p "$MONITOR_DIR"

session_exists() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"
}

snapshot() {
  local destination=$1
  local timestamp
  local health_code
  local models_code
  timestamp=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
  health_code=$(curl --max-time 3 --silent --output /dev/null --write-out '%{http_code}' http://127.0.0.1:8040/health 2>/dev/null || true)
  models_code=$(curl --max-time 3 --silent --output /dev/null --write-out '%{http_code}' http://127.0.0.1:8040/v1/models 2>/dev/null || true)
  {
    echo "timestamp=$timestamp"
    if session_exists "$SESSION"; then
      echo "session=running"
    else
      echo "session=exited"
    fi
    echo "inference_health_http=${health_code:-000}"
    echo "inference_models_http=${models_code:-000}"
    echo "component_processes:"
    pgrep -af 'prime_rl|/bin/trainer|/bin/orchestrator|/bin/inference' 2>/dev/null || echo none
    echo "gpu_state:index,memory_used_mib,utilization_percent"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo unavailable
    echo "checkpoints:"
    find "$RUN_ROOT/checkpoints" "$RUN_ROOT/weights" "$RUN_ROOT/run_default/checkpoints" "$RUN_ROOT/run_default/broadcasts" \
      -maxdepth 2 -type d -name 'step_*' -printf '%p\n' 2>/dev/null | sort || true
    echo "best_manifest:"
    if [[ -f "$RUN_ROOT/best_checkpoints.json" ]]; then
      python3 - <<PY 2>/dev/null || true
import json
from pathlib import Path
p = Path("$RUN_ROOT") / "best_checkpoints.json"
d = json.loads(p.read_text())
print({"selected_steps": d.get("selected_steps"), "policy": d.get("selection_policy")})
PY
    else
      echo missing
    fi
    echo "disk:"
    df -h "$RUN_ROOT" /tmp 2>/dev/null || true
    echo "log_sizes:"
    find "$RUN_ROOT/logs" -maxdepth 2 -type f -printf '%P %s\n' 2>/dev/null | sort || true
    echo "---"
  } >>"$destination"
}

maybe_prune() {
  local now
  now=$(date +%s)
  if (( now - LAST_PRUNE < PRUNE_INTERVAL )); then
    return 0
  fi
  LAST_PRUNE=$now
  {
    echo "timestamp=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
    python3 scripts/prune_symbolic_checkpoints.py --require-val "$RUN_ROOT"
    df -h "$RUN_ROOT" /tmp
    echo "---"
  } >>"$PRUNE_LOG" 2>&1 || {
    {
      echo "timestamp=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
      echo "prune_skipped_or_failed"
      echo "---"
    } >>"$PRUNE_LOG"
  }
}

snapshot "$EVENTS"
while session_exists "$SESSION"; do
  sleep "$INTERVAL"
  maybe_prune
  snapshot "$EVENTS"
done

maybe_prune
snapshot "$FINAL"
for component in trainer orchestrator inference; do
  log="$RUN_ROOT/logs/$component.log"
  {
    echo "===== tail:$component ====="
    tail -n 200 "$log" 2>/dev/null || echo missing
  } >>"$FINAL"
done
