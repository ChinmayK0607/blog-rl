# Swarm Arena RL v4: eight-update development run

This is a compact record of the first complete four-policy RL v4 run. Four
distinct Qwen3-1.7B LoRA policies were optimized from the same SFT warm start.
Every BLUE role received its own optimizer and token spans; the only reward was
the shared, verified terminal team return. There was no message, silence,
capture, action, or judge bonus.

## Mechanical result

- Eight updates completed from the pinned SFT warm start: 32 signed, replayed,
  current-policy-rescored groups and 128 game replicas.
- All four policies produced distinct, stable PEFT adapters.
- Every locked aggregate actor/trainer parity bound passed at every update.
- The final collapse audit found no action, speech, repeated-target, KL,
  opponent-specialization, or return-without-message collapse.
- The full Linux Swarm Arena suite passed: 102 tests in 40.00 seconds.
- Both 128-case non-arena regression suites passed for all policies with zero
  leakage. Candidate-to-SFT reference-state KL mean was 0.000410 and p99 was
  0.01261.

## Development evaluation

Each opponent monitor contains 96 games. Confidence intervals use four paired
independent units per cell, so these results guide the next run but do not
support a final claim.

| Opponent | hard: RL - SFT | legacy: RL - SFT | critical: normal - dropped | matched decoy: normal - dropped |
|---|---:|---:|---:|---:|
| Base | +0.0306 | -0.0351 | +0.0846, CI [+0.0397, +0.1295] | +0.0583, CI [+0.0094, +0.1072] |
| SFT | +0.0193 | +0.0187 | +0.0315, CI [-0.0063, +0.0711] | -0.0052, CI [-0.0800, +0.0564] |
| Historical RL | -0.0280 | +0.0307 | +0.0904, CI [+0.0219, +0.1677] | +0.0393, CI [-0.0170, +0.0955] |

Action validity, broadcast protocol validity, and broadcast grounding were
1.0 throughout. Message removal hurt performance in the critical cases against
all three opponents, which is useful evidence that the environment and training
route can expose communication sensitivity. However, message removal also hurt
the matched decoy cases against the base opponent. The effect is therefore not
cleanly information-specific, and this checkpoint does not establish learned
swarm cooperation.

## Verdict

The multi-agent RL system is mechanically working and stable. Capability change
after eight updates is small and mixed across opponent families. The next run
should increase causal information-handoff density while retaining matched
decoys, use a longer locked schedule, and select one checkpoint on a larger
development/selection tier before opening the frozen final evaluation.

Public, checksum-verified LoRA bundle:
<https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-step8-development>
(revision `6a660a3fabfebd3270753155a131d2148d463b82`).

Selection and frozen final evaluation were never opened.
