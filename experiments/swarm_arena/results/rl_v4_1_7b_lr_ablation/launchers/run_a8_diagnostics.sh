#!/usr/bin/env bash
set -euo pipefail

train_repo=/workspace/blog-rl-run
eval_repo=/workspace/blog-rl-eval-6d6fe88e
candidate=/workspace/runs/rl-v4-ablate-lr1e5-mix211-12e0c461/exports/step_8
holdout=/workspace/runs/rl-v4-ablation-holdout-a8-shards-retry2-6d6fe88e
out=/workspace/runs/rl-v4-ablation-a8-diagnostics-6d6fe88e
python=/root/.cache/uv/environments-v2/prime-rl-cp3.12.3-6d0588b29bf55ef8/bin/python

mkdir -p "$out/logs" "$out/regression/comparisons" "$out/eval"
exec > >(tee -a "$out/logs/launcher.log") 2>&1

echo "$(date -u +%FT%TZ) starting A8 policy-KL and regression diagnostics"
cd "$eval_repo"
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$eval_repo/experiments/swarm_arena" "$python" \
  "$eval_repo/experiments/swarm_arena/scripts/measure_constrained_policy_kl.py" \
  --model /workspace/models/qwen3-1.7b-70d244c \
  --baseline-adapter /workspace/artifacts/warmstart-1.7b-step320 \
  --candidate-adapter "blue-0=$candidate/blue-0" \
  --candidate-adapter "blue-1=$candidate/blue-1" \
  --candidate-adapter "blue-2=$candidate/blue-2" \
  --candidate-adapter "blue-3=$candidate/blue-3" \
  --probe "$eval_repo/experiments/swarm_arena/results/pre_rl_1_7b/parity_eager_313c1aa7/parity_probe.json" \
  --output "$out/policy_kl.json" >"$out/logs/policy-kl.log" 2>&1

for role in 0 1 2 3; do
  for suite in v1 v2; do
    target="$out/regression/blue-$role-$suite"
    env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$eval_repo/experiments/swarm_arena" "$python" \
      "$eval_repo/experiments/swarm_arena/scripts/score_regressions.py" \
      --model /workspace/models/qwen3-1.7b-70d244c \
      --adapter "$candidate/blue-$role" \
      --device cuda --batch-size 16 --suite "$suite" --output-dir "$target" \
      >"$out/logs/reg-blue-$role-$suite.log" 2>&1
    protocol=paired-swarm-regression-v1
    if [[ "$suite" == v2 ]]; then protocol=paired-swarm-regression-v2; fi
    env PYTHONPATH="$eval_repo/experiments/swarm_arena" "$python" \
      -m swarm_ctf_eval.regression_compare \
      --base-rows "/workspace/runs/rl-v4-eval-1379f9c2/regression/sft-$suite/rows.jsonl" \
      --adapter-rows "$target/rows.jsonl" \
      --comparison-protocol "$protocol" \
      --output "$out/regression/comparisons/blue-$role-$suite.json" >/dev/null
  done
done

"$python" - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
comparisons = {}
for role in range(4):
    for suite in ("v1", "v2"):
        key = f"blue-{role}-{suite}"
        comparisons[key] = json.loads((out / "regression" / "comparisons" / f"{key}.json").read_text())
report = {
    "comparisons": comparisons,
    "passed": all(row["gates"]["passed"] for row in comparisons.values()),
    "baseline": "pinned SFT step320",
    "selected_step": 8,
}
(out / "regression_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"regression_passed": report["passed"]}))
PY

cp "$holdout/combined/rows.jsonl" "$out/eval/rows.jsonl"
: > "$out/eval/raw.jsonl"
for shard in offset-9 offset-10 offset-11; do
  cat "$holdout/$shard/raw.jsonl" >> "$out/eval/raw.jsonl"
done

env PYTHONPATH="$eval_repo/experiments/swarm_arena" "$python" \
  "$eval_repo/experiments/swarm_arena/scripts/audit_final_eval_collapse.py" \
  --eval-dir "$out/eval" --policy-kl "$out/policy_kl.json" \
  --policy-alias ablate-a8-blue-0=blue-0 \
  --policy-alias ablate-a8-blue-1=blue-1 \
  --policy-alias ablate-a8-blue-2=blue-2 \
  --policy-alias ablate-a8-blue-3=blue-3 \
  --output "$out/collapse_audit.json" >"$out/logs/collapse.log" 2>&1

sha256sum "$out/policy_kl.json" "$out/regression_summary.json" "$out/collapse_audit.json" > "$out/SHA256SUMS"
echo "$(date -u +%FT%TZ) A8 diagnostics complete"
