# V12: counterfactual-robust communication continuation

V12 is a 160-update continuation from the four distinct public V11 update-180
policies. It is not a retroactive promotion of V11: V11 failed its predeclared
development selector. The warm start is used because V11 nevertheless retained
valid play and learned a strong receiver/message relationship, while its two
observed deficiencies are directly targeted here.

## What V11 taught us

V11's best development checkpoint had positive semantic return and
critical-minus-decoy specificity, but ordinary legacy return was `-0.0093884`
relative to SFT. Rollout inspection also showed overly literal receiver behavior:
the receiver often followed a swapped teammate target even when its own private
observation already identified the correct world. This is not protocol collapse;
it is a conflict-resolution and retention problem.

## The change

The reward remains verified terminal team return. There is no communication,
truthfulness, target-action, capture, or validity bonus.

- Critical handoffs keep the existing receiver-only target-swap contrast. The
  factual receiver ACT branch is trained on the absolute paired
  `factual_return - swapped_return`.
- Matched decoys use a new challenge contrast. The receiver can already infer
  the world privately, so the misleading-message receiver ACT branch is trained
  on the absolute paired `swapped_return - factual_return`. Blind obedience can therefore
  receive negative policy-gradient credit on the exact action tokens that caused
  it, without exposing the oracle target as a supervised label.
- V12 does not center these paired differences across the four replicas. If all
  four receivers make the same good or bad choice, all four retain the same
  positive or negative verified terminal-return signal instead of collapsing
  to zero. Legacy runs retain their original replica-mean centering semantics.
- Ordinary games keep leave-one-out terminal-return training and occupy 260 of
  640 groups. The first 40 updates use two ordinary groups per update to repair
  V11's small legacy regression before longer-horizon consolidation.

The schedule contains 260 ordinary, 220 critical, and 160 challenge groups.
Critical receivers are exactly balanced at 55 groups per policy; challenge
receivers are exactly balanced at 40 per policy. Every challenge case is a
matched critical topology/world case, legal action sets are identical across
the intervention, and all four opponent snapshots rotate once per update.

## Warm start and policy isolation

All four policies remain separate. `prepare_v12_distinct_warmstart.py` reads the
anonymous public V11 step-180 manifest, hashes the four local adapter files,
writes a trainer config with one adapter path/hash per `run_blue_*`, and emits a
matching controller manifest. Trainer and controller both fail closed if any
mapping differs. No policy merging or post-hoc averaging is permitted.

## Development and frozen evaluation

V12 uses a fresh 36-bundle/36-ordinary development split. Its final suite is the
still-unopened V11 frozen suite copied byte-for-byte. Candidate checkpoints are
20, 40, 80, 120, and 160. The earliest candidate is selected only when:

1. critical normal-minus-target-swapped mean return is positive;
2. critical-minus-decoy semantic specificity mean is positive;
3. legacy ordinary clustered 95% lower bound is at least `-0.02`; and
4. hard ordinary clustered 95% lower bound is at least `-0.02`.

The `-0.02` non-inferiority margin is fixed before V12 data are observed. It is
not applied retroactively to V11. Frozen evaluation runs once, only after formal
selection, and still requires positive clustered message-use, RL-specific, and
specificity intervals plus protocol/KL/collapse integrity.

## Fail-fast GPU policy

The update-20 and update-40 pulses compare against the exact u180 initializer.
If neither ordinary retention nor decoy robustness improves and critical
semantic return is non-positive at both checkpoints, stop rather than pay for
the remaining 120 updates. A positive pulse is only a continuation signal; it is
not a held-out result.

## CPU verification and remaining GPU-only gate

The deterministic builder, task binding, production plan, curriculum audit,
selector, warm-start manifest generator, syntax checks, and lightweight tests
are completed locally. The exact counterfactual audit certifies private-world
structure, legal-action invariance, terminal-reward opportunity, split
disjointness, and role/world balance. A first-update Linux smoke must still show
nonzero challenge advantages for every policy slot, exact trainer/serving parity,
and successful distinct-adapter loading before the long run is admitted.
