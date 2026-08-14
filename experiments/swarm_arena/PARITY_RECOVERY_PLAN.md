# Swarm Arena parity recovery

Status: CPU implementation complete; GPU execution pending.

## Why RL remains blocked

The authoritative broad probe proved exact serving/client constraint-mask
agreement but rejected the current vLLM/Prime pair on maximum probability error
and maximum mismatch-KL. Replacing behavior log-probabilities with trainer
values would hide the off-policy difference and is forbidden.

Prime's custom Qwen3 training implementation does not currently support a
KV-cache generation path. A Prime-native rollout actor would therefore require
new cached inference engineering; repeatedly running its full training forward
for each generated token is not an acceptable fallback.

## Predeclared four-GPU diagnosis

Start with the already-published 64-sample broad probe that rejected the
baseline. It is deterministic, covers the exact known outliers, and avoids
paying for four GPUs while only one runs vLLM. Run the checked-in matrix with
one variant per GPU:

1. custom implementation + FlashAttention 2 (baseline);
2. Hugging Face implementation + FlashAttention 2;
3. Hugging Face implementation + SDPA;
4. custom implementation + eager attention.

All variants preserve the same model, LoRA initialization, loss, optimizer,
precision defaults and frozen parity thresholds. Selection is the first variant
in this declared order that passes both numerical parity and four-policy
isolation. The matrix writes the exact commands, source/probe/config hashes,
per-variant logs and machine-readable reports. It exits nonzero if none pass.
The matrix launcher requires the expected probe SHA-256 and rejects a mismatch
before creating an output directory. For the published broad probe, the raw
digest is
`fe0ae52d78c3e85607bd1c74a265a7f7721df917fedb407c1d67d75b28d3162d`.

After decompressing the public probe to a temporary remote path, the launch
shape is:

```bash
uv run python experiments/swarm_arena/scripts/certify_prime_parity_matrix.py \
  --matrix experiments/swarm_arena/configs/parity_matrix_1_7b.json \
  --model /workspace/models/qwen3-1.7b-70d244c \
  --adapter /workspace/artifacts/warmstart-1.7b-step320 \
  --adapter-sha256 2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b \
  --probe /workspace/runs/parity-broad4-probe.json \
  --probe-sha256 fe0ae52d78c3e85607bd1c74a265a7f7721df917fedb407c1d67d75b28d3162d \
  --output-dir /workspace/runs/parity-recovery-matrix \
  --source-commit "$(git rev-parse HEAD)"
```

The matrix is a development diagnostic, not RL admission. If a variant passes,
start serving and generate fresh schema-v2 four-group evidence with full
constrained-distribution telemetry and that trainer config's hash bound into the
run lock. Stop serving and recertify the selected variant. Only that fresh bound
certificate can authorize a bounded optimizer pilot.

If no variant passes, stop the instance. The next task is a cached Prime-native
policy actor or another inference backend that is numerically equivalent to the
trainer; do not tune thresholds, discard outlier samples, or relabel behavior
log-probabilities.

## Cost-control sequence

1. Provision 4x L40S only when an operator can inspect the first 10–15 minutes.
2. Decompress the published failed broad probe and run all four trainer variants
   concurrently. Do not start vLLM, OOD evaluation or RL.
3. If none pass, save compact reports/logs, stop all processes and terminate.
4. If one passes, run exactly one fresh schema-v2 rollout and bound
   recertificate; abort on any mask, transport, replay, hash or admission
   failure.
5. Start RL only after that pass and configure provider termination for the
   later bounded pilot.

The first GPU session is an observed systems diagnosis, not an overnight sweep.
