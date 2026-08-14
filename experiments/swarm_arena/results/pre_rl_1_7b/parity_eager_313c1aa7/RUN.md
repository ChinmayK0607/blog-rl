# Eager-serving single-group diagnostic

This is a **passing narrow diagnostic**, not the final RL admission
certificate. It established that eager serving can meet the frozen envelope on
one approved group and motivated the broader mask-audited run.

- Source commit: `313c1aa7`
- Probe: 16 samples, 1,004 completion tokens, 860 branching tokens.
- Result: all frozen parity components passed. Mean absolute log-probability
  error was `0.002077`, maximum probability error `0.078062`, and maximum
  mismatch-KL `0.018078`.
- Four-policy isolation passed and only `run_blue_0` changed in the disposable
  isolation step.
- This evidence predates per-token verification of the server's actual finite
  constrained-token set and covers only one group; it therefore cannot admit
  RL by itself.

Artifact SHA-256 values are the output of `shasum -a 256`:

- `parity_report.json`: `6b86efd079b78af4242f756930e051df8983729e0f3b6b744e5ec9120a9defec`
- `parity_probe.json`: `54fd04d40f83adbe438a9f0c214fcd377f10c6e24088831aea3bfe4b4f4da3ae`
- `shared_return_evidence.jsonl`: `ae96d090e3b29399ce3300b3becde4cb0097284deaa977c258c5da68c1c5e0b5`
- `admission.jsonl`: `1691d2544fc9d230f4afbbcd06c83aeba2b0b0c9febab3e240ecdadeadd3b229`
- `live_rl_diagnostic.json`: `3a302274b389c8724fc488b7dc3312cc3fe31cebc16a246b728c8b2d4eadeee4`

