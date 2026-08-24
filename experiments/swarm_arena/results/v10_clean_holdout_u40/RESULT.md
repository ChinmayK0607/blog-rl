# V10 clean held-out update-40 result

The once-selected v10 update-40 checkpoint completed the 4,260-game clean
held-out remainder. The result is promising evidence that its actions depend on
teammate message content, but it does not pass the stricter claim that RL
created a new, communication-specific capability beyond the SFT initializer.

## Headline result

- Candidate critical normal minus receiver-only target-swapped terminal return:
  `+0.0320811`, 95% CI `[+0.0157342, +0.0506818]`, 22 independent bundles.
- Receiver target-action rate: `0.7803` normally versus `0.2121` after the
  receiver-only target swap; paired gap `+0.5682`, 95% CI
  `[+0.3939, +0.7348]`.
- The semantic return effect was positive against base, SFT, and the frozen
  historical-league label.
- Candidate sender target-fact rate and intervention eligibility were both
  `0.8826`.

Two confirmatory requirements narrowly failed:

- RL-minus-SFT semantic sensitivity was `+0.0194063`, 95% CI
  `[-0.0003998, +0.0386262]`.
- Critical-minus-matched-decoy semantic specificity was `+0.0131639`, 95% CI
  `[-0.0022569, +0.0290604]`.

Delayed and sender-shuffled messages caused significant degradation. The
normal-minus-dropped and normal-minus-zero-budget intervals crossed zero.
Ordinary and overall RL-minus-SFT gameplay intervals also crossed zero. All
defined action, broadcast, and grounding rates were `1.0`.

## Post-evaluation diagnostics

The public candidate adapters had incorrect metadata: every
`adapter_config.json` declared rank 16 / alpha 32, while the immutable tensor
files and frozen trainer config independently established rank 32 / alpha 64.
The KL audit used metadata-corrected views that symlink the original unchanged
weight files. `ADAPTER_METADATA_REPAIR.json` records all original/repaired
config and weight hashes.

The frozen 32-decision constrained probe produced 1,339 token rows, including
154 branching tokens. Candidate-to-SFT KL was:

- overall mean `0.0013960`, p99 `0.0027070`;
- blue-0 mean/p99 `0.0000167 / 0.0001377`;
- blue-1 mean/p99 `0.0025050 / 0.1540455`;
- blue-2 mean/p99 `0.0026870 / 0.0027501`;
- blue-3 mean/p99 `0.0011854 / 0.0356621`.

All policies pass the frozen mean `0.08` and p99 `0.30` limits. One isolated
blue-2 token has KL `1.5688`; it does not change the preregistered diagnostic
gate, but the maximum is retained rather than hidden.

The complete 4,260-trajectory collapse audit passed every flag:

- speaking rates: `0.2947` to `0.3132`;
- action concentration: `0.2549` to `0.2856`;
- message-target concentration: `0.0394` to `0.0484`;
- no always/never-speaking, repeated-target, action, excessive-KL,
  one-opponent, or return-without-message-gain flag;
- zero orphan raw records.

## Artifact hashes

- held-out summary: `13c7f787e3930d0893f52b1f26290ccd63ca49b00b722aedbd7f0ecd7c39aee2`
- KL probe: `db525f1201775febc4fb8933758c16e4821fd7e1d2e9a524db0a650cc7142edb`
- KL report: `09673097fcef7b8e30106f58aafcef77020bdab51bf1ab7bbb4e2c15bd3008ae`
- collapse audit: `54f72bd078f2371e0be52f4f57aa151b7876a86fc5b03cc4d397c7af67025be6`
- metadata-repair manifest: `91243fadb80cb7441e23a9353610f34a0f8a2aeb5b687676e35aa49c0847b3b4`

Verdict: **promising causal message-use signal; strict confirmatory RL-specific
communication claim rejected narrowly; protocol and collapse diagnostics pass**.
