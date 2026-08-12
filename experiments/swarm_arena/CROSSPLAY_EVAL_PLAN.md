# Swarm Arena cross-play evaluation plan

This protocol is frozen before inspecting any learned-policy cross-play result.
It tests whether SFT warm starts can play complete games, coordinate through
private communication, and adapt in context to model-controlled opponents. It
does not treat relative wins between two weak policies as absolute competence.

## Policies

The initial pool contains every available base/warm-start pair:

1. `Qwen/Qwen3-1.7B`;
2. the validation-selected 1.7B rank-8 LoRA trained on warm-start v7;
3. `Qwen/Qwen3-4B-Instruct-2507`;
4. `CK0607/Qwen3-4B-Swarm-Arena-SFT-v2`;
5. the archived 4B warm-start v7 adapter.

A trained policy that fails protocol or non-arena regression gates remains in
diagnostic cross-play but is labeled ineligible as an RL initialization.

## Information boundary and timing

- All eight agents are model-controlled.
- Every agent has an independent private context and receives no model identity,
  global state, opponent messages, or opponent actions outside local events.
- Each turn has one eight-request broadcast barrier, private within-team message
  delivery, one eight-request action barrier, and one simultaneous transition.
- A bounded three-turn private history contains only that agent's accepted
  broadcasts, teammate inbox, selected actions, and locally visible events.
- Generation is greedy with thinking disabled. Prompts are flushed to raw rows.
- Every run manifest requires an immutable Hugging Face revision or adapter
  SHA-256 for each served policy; mutable model names alone are insufficient.

## Engineering screen

Before the result-producing sweep, run two complete development matches for each
model and stop to diagnose any of:

- action protocol validity below 95%;
- raw broadcast protocol validity below 99%;
- grounded broadcast validity below 99%;
- accepted non-empty messages below 10% of broadcast opportunities;
- NaN/OOM/server errors;
- aggregate inference below 150 generated tokens/s after warm-up;
- GPU utilization persistently below 60% while requests are pending.

Fix only implementation faults. Do not alter frozen seeds, parsers, rewards, or
promotion thresholds in response to model behavior.

## Development tournament

Use eight development seeds with sizes 12/13 and horizons 4/6.
For a single base/adapter pair this produces 104 complete games: 64 paired
communication-intervention games, 32 paired asymmetric-history games, and eight
self-play stability games.

- Run every unordered cross-model pairing on the same seeds with sides swapped.
- Run four same-model self-play seeds per policy as a stability control.
- For every trained-versus-base pair, repeat with the focal trained team's
  messages generated, dropped, sender-shuffled, and delayed while the opponent
  remains generated.
- Repeat the trained-versus-base normal matchup with asymmetric history
  controls: focal history three versus zero while opponent history stays three,
  then opponent history three versus zero while focal history stays three. This
  separately estimates how the focal model adapts to its adversary and how much
  the adversary's adaptation affects the focal model.

Development results select at most two trained policies for the sealed sweep.
Selection order is: protocol eligibility, no material regression, side-swapped
return against its base, generated-minus-dropped return, generated-minus-shuffled
return, then lower duplicate-target rate. No threshold is relaxed post hoc.

## Sealed OOD sweep

`FROZEN_CROSSPLAY_CASES` contains 24 untouched seeds, graph sizes 14/16, and
horizons 6/8. The split cannot be truncated from the CLI.

For each selected policy:

- play its base and the other selected policy, with both side assignments;
- run generated/generated, dropped/generated, sender-shuffled/generated, and
  delayed/generated conditions on identical seeds;
- retain a small deterministic-policy/oracle anchor solely for absolute task
  calibration, never as the main adversary result.

## Endpoints and claims

Primary endpoint: paired, side-averaged terminal return.

Mechanism endpoints:

- generated-minus-dropped return;
- generated-minus-sender-shuffled return;
- generated-minus-delayed return;
- focal-history-three-minus-focal-history-zero return with opponent fixed;
- opponent-history-three-minus-opponent-history-zero return with focal fixed;
- shared fact updates, duplicate-target turns, communication spend, and protocol
  failure rates.

Report paired seed-level nonparametric bootstrap 95% intervals (20,000
deterministic resamples), paired two-sided sign-randomization tests, and raw
seed-level results. Randomization is exact through 16 seeds and uses 100,000
fixed-seed trials above that. A policy may be called better at gameplay when its
paired side-averaged interval excludes zero. It may
be called communication-dependent only when generated messages beat dropped and
shuffled messages. History sensitivity is evidence of in-context adaptation,
not online learning and not weight adaptation.

## Artifact and storage policy

- Append and flush each complete match; resume by seed and condition.
- Commit compact manifests, summaries, analysis, and representative trajectories.
- Upload selected adapters and full raw trajectory archives to the Hugging Face
  Hub with checksums.
- Never copy model weights or caches to the Mac.
- Delete remote caches and unselected checkpoints only after verifying durable
  copies; never delete the only copy.
