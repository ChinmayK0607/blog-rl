# Compiled-serving parity diagnostic

This is a **failed diagnostic**, not an RL admission certificate.

- Source commit: `fb272aed`
- Probe: 16 first-turn broadcast samples, 1,002 completion tokens, 857
  branching tokens, derived from the authoritative v4 shared-return smoke.
- Result: six aggregate/tail/KL gates passed, but maximum probability error
  was `0.140238` against the frozen `0.10` limit.
- Four-policy isolation passed: optimizer parameter sets were disjoint and the
  disposable test step changed only `run_blue_0`.
- No training checkpoint was produced or promoted.

`report.json` SHA-256:
`0dfe60139c207b376300f3a581cd6c1a7a31be98a455ae404866e99a9ace11fe`.

