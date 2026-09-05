# Broad-run transport failure

This directory preserves the failed attempt that motivated transport
hardening. It is not parity or RL-admission evidence.

- Source commit: `ae857d9d`
- Two of four groups completed their replay and signed admission.
- A subsequent request raised `httpx.ReadError`; the vLLM server remained
  healthy and its completed requests were HTTP 200.
- No trainer or optimizer ran.
- Fix: disable local HTTP keep-alive and permit at most three identical,
  seed-bound retries only for network/protocol exceptions. Every retry count is
  committed in the decision evidence. The fresh authoritative run then
  completed 256/256 decisions on attempt one.

Artifact SHA-256 values:

- `run.log`: `e87b2f0d9686b26f3549aa10b2ab18e02af44049759de3922af3ee8ea178bbc9`
- `admission.jsonl`: `c698b7a65e57bf41622e146eac0008ec87cd353a310923754906edd3cb858dd4`
- `shared_return_evidence.jsonl.gz`:
  `71ae7167730af16d4d5a92aaab96fc6e666eabeef80f7cab86dc34abf659c31b`

The supervisor signing key is intentionally excluded.

