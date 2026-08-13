# Swarm Arena 1.7B pre-RL certification

Status: **passed** on one NVIDIA RTX 6000 Ada 48 GB.

## Immutable inputs

- Base: pinned Qwen3-1.7B revision `70d244cc`
- Warm-start adapter: `CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible`
- Adapter SHA-256: `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Code under test: commit `578a98a7`

## What was certified

- The frozen RL-v3 manifests pass their global hash, seed-disjointness,
  role-balance and counterfactual-isolation audit: 240 train, 48 development
  and 72 frozen-OOD pairs.
- vLLM structured rollouts and the actual Prime FSDP trainer agree within the
  checked-in constrained-distribution envelope across 32 action/broadcast
  prompts, 1,331 completion tokens and 149 branching tokens.
- Four LoRA policy slots initialize from the same pinned adapter digest while
  retaining disjoint optimizer parameter sets.
- A real optimizer step routed to `run_blue_0` changes only `run_blue_0`.
- The signed supervisor/router path fails closed on replay, reward, private
  context, run-lock, constraint, policy-routing, token-ownership and parity
  mismatches.
- Focused CPU suite: 56 tests passed. The complete GPU LM-head regression file
  passed, including the three targeted FP32-head forward/backward checks.

## Interpretation

This certifies correctness of the pre-RL data, safety/admission path,
constrained loss, pinned warm start, four-policy isolation and numerical
serving/training compatibility. It does not certify asynchronous throughput;
that requires the planned multi-GPU topology. Re-run parity after any change to
the model, adapter, tokenizer, constraints, precision path, vLLM or FSDP stack.

The detailed machine-readable gates are in `parity_certificate.json` and
`global_audit.json`.
