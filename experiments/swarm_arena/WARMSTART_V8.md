# Sequential protocol warm start v8

This corrective run was specified after the six-match 1.7B engineering pilot
failed and before any result-producing development tournament began. The pilot
is diagnostic and remains preserved; its thresholds are not retroactively used
to claim a gameplay result.

## Confirmed v7 failure

- The base model selected legal actions but wrapped otherwise structured
  broadcasts in Markdown, giving 0/112 raw broadcast protocol validity.
- The selected v7 adapter kept action validity at 100% but produced only 42/112
  protocol-valid broadcasts and 27/112 accepted non-empty broadcasts.
- Adapter errors included unsupported facts, illegal intents, fact-budget
  violations, and depleted-message-budget violations.
- Prompt rendering was byte-identical between SFT and serving. The failure was
  therefore not an HTTP, adapter-load, renderer, or chat-template mismatch.
- V7 arena rows covered only BLUE agents at turn zero with no inbox, private
  history, or depleted episodic message budget. Its selector required grounded
  fallback output but omitted raw broadcast legality, allowing a false positive.

## V8 data contract

V8 replaces the turn-zero arena rows while retaining the replay and general
instruction preservation rows from v7.

- 40 training episodes and eight validation episodes;
- both BLUE and RED agents;
- graph sizes 12/13 and horizons 4/6;
- every broadcast and action phase through termination;
- exact inboxes, locally visible events, and up to three prior private turns;
- live remaining message budget, including depleted-budget states;
- deterministic messages composed only of prompt-visible facts and legal
  intents, clipped to the live message budget;
- explicit empty targets when no useful component can legally be sent;
- independent prompt-visible local action targets, not solver-optimal joint
  trajectories.

This remains a protocol warm start. It does not supervise long-horizon joint
allocation, opponent modelling, or a communication-dependent swarm policy.

## Frozen training recipe

- base: `Qwen/Qwen3-1.7B`;
- rank-16 LoRA, alpha 32, dropout 0.05;
- Q/K/V/O attention projections only;
- 320 steps, effective batch 16, sequence length 2048;
- learning rate 4e-5 with 16 warm-up steps and cosine decay;
- checkpoints every 64 steps;
- assistant-token loss only with thinking disabled.

## Promotion gates

A checkpoint is eligible for the next engineering screen only if it satisfies
all of the following without parser repair or post-hoc threshold changes:

- 100% aggregate JSON schema validity on sequential validation;
- at least 99% raw legal broadcasts, including grounding, intent, fact-count,
  and remaining-budget checks;
- at least 99% grounded broadcasts;
- at least 99% legal actions;
- both frozen paired non-arena regression v1 and v2 gates pass.

The checkpoint with the highest phase-balanced exactness wins; ties prefer the
earlier checkpoint. A selected checkpoint must then pass a fresh six-match
engineering screen before the 104-game development tournament:

- at least 99% raw broadcast protocol validity;
- at least 99% grounded accepted broadcasts;
- at least 95% action protocol validity;
- non-empty accepted messages in at least 10% of broadcast opportunities;
- aggregate measured completion throughput at least 150 tokens/s after warm-up;
- no HTTP, NaN, OOM, or server errors.

The serving screen uses compiled/CUDA-graph inference and an 8192-token context
limit. Any context overflow is an engineering failure, not permission to truncate
private history silently.
