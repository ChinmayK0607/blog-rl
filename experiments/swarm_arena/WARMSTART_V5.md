# Canonical protocol warm start

`arena-warmstart-v5` is the current supervised checkpoint recipe for entering
Swarm Arena RL. It is deliberately a narrow interface warm start: it teaches a
single agent to emit one grounded broadcast JSON object and one legal action ID
without claiming to teach cooperation.

## Why v5 exists

The replay-protected v4 run preserved both frozen regression suites, but did
not clear the fixed protocol gate. Inspection showed that its broadcast labels
had an avoidable identifiability defect: an unobserved seed-index rule selected
between empty, fact-only, fact-plus-intent, and full messages. The same
prompt-level task therefore had conflicting targets.

v5 removes that label noise. Every broadcast target is now the canonical local
report: up to the environment fact limit, the legal local intent, and any
prompt-identifiable resource request. The target depends only on the agent's
private observation and displayed legal actions. Communication sparsity and
multi-agent usefulness remain terminal-reward RL objectives rather than hidden
SFT labels.

Both deployed protocol prompts now explicitly require exactly one JSON object
and tell the model to stop, while the parser remains strict about trailing
content.

## Data and training

- 640 balanced canonical arena rows, 640 exact instruction-preservation rows,
  and 1,280 deterministic base-behavior replay rows;
- 192 held-out rows: 96 arena and 96 preservation;
- rank-8 LoRA on Q/V attention projections only (2,949,120 parameters);
- 128 SFT steps, `5e-5` peak learning rate, and checkpoints every 32 steps.

Build and audit from the repository root:

```bash
uv run --project /root/blog-rl --no-sync python -m scripts.build_warmstart_v5 \
  --replay-rows experiments/swarm_arena/data/warmstart_v3_work/base_replay_concise_system_b128.jsonl \
  --output-dir experiments/swarm_arena/data/warmstart_v5

uv run --project /root/blog-rl --no-sync python -m scripts.audit_warmstart_v3 \
  --dataset experiments/swarm_arena/data/warmstart_v5 \
  --output experiments/swarm_arena/data/warmstart_v5/audit.json

uv run --project /root/blog-rl --no-sync sft @ experiments/swarm_arena/configs/sft_warmstart_v5.toml
```

## Promotion rule

Promotion requires all of the following on frozen evidence:

1. 100% schema-valid warm-start validation responses;
2. at least 99% grounded broadcast responses;
3. at least 99% legal action responses;
4. both paired general-behavior regression suites pass their fixed overall,
   per-category, and arena-leakage gates;
5. a frozen one-step arena bootstrap has no catastrophic change.

Only a promoted adapter may be published or used to initialize RL.
