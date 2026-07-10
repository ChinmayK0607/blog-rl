#!/usr/bin/env bash
# Auto-finalizer: once the Phase-B rerun queue ends, keep only lean HF weights.
# Drops any residual DCP checkpoints + weight broadcasts and collapses each arm's
# HF export to a single vLLM-loadable model.safetensors (value_head stripped for
# PPO). Runs independently of the live supervisor so it never perturbs it.
set -uo pipefail

ROOT=/home/ubuntu/semi/prime-rl
ART=/home/ubuntu/semi/artifacts
SUMMARY="$ART/logs/finalize_phaseb.status"
export UV_PROJECT_ENVIRONMENT=.venv
timestamp() { date --utc +%Y-%m-%dT%H:%M:%SZ; }
session_exists() { tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"; }

cd "$ROOT"
echo "$(timestamp) finalize_wait_for_queue" | tee -a "$SUMMARY"
while session_exists semi_hardb_rerun; do sleep 60; done
sleep 20  # let final checkpoint export flush

ARMS=(hardb-full-grpo-v1 hardb-compacted-grpo-v1 hardb-compacted-ppo-v1 hardb-compacted-ppo-warmcritic-v1)
for arm in "${ARMS[@]}"; do
  d="$ART/$arm"
  [[ -d "$d" ]] || { echo "$(timestamp) skip $arm (absent)" | tee -a "$SUMMARY"; continue; }
  rm -rf "$d/checkpoints" "$d/run_default/broadcasts" 2>/dev/null || true
  if [[ -d "$d/weights" ]]; then
    uv run --no-sync python scripts/finalize_hf_checkpoint.py --run-root "$d" >>"$SUMMARY" 2>&1 || true
  fi
  echo "$(timestamp) finalized $arm free=$(df -h / | awk 'NR==2{print $4}')" | tee -a "$SUMMARY"
done
echo "$(timestamp) finalize_done free=$(df -h / | awk 'NR==2{print $4}')" | tee -a "$SUMMARY"
