# Focused atomic RL run through update 60

This compact record preserves the prospectively capped endpoint of the
four-policy Qwen3-1.7B Swarm Arena run. The immutable plan declared 80 updates;
the user requested a stop at update 60 before the update-50 or update-60
results were observed. The controller stopped at the content-bound update-60
barrier and the complete frozen update-60 pulse then finished.

## Verdict

**Exploratory capability signal; not admitted for communication.** Overall
gameplay improved by `+0.029311` from the exact same-weight update-zero anchor
to update 60. The largest gain was on the legacy tier (`+0.081167`). Hard-tier
performance was slightly worse (`-0.009878`), while critical-handoff capability
improved (`+0.016645`).

The causal communication result went in the wrong direction. RL-specific
message lift changed by `-0.063705`, and critical-minus-decoy specificity by
`-0.038171`, from update 0 to update 60. The communication gate failed at every
pulse. Return improvement therefore means task-capability learning, not learned
swarm communication.

Update 50 had the best observed overall development mean (`+0.025233` versus
SFT) and is retained as the retrospectively selected capability checkpoint.
Its 95% interval still crossed zero and it is explicitly not communication-
admitted. Update 60 is retained separately as the transparent capped endpoint.

## Public artifacts

- [Selected update-50 adapters](https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-focused-step50-development), revision `049e95062903501a8a50efac09d1b2caab393364`
- [Capped update-60 adapters](https://huggingface.co/CK0607/Qwen3-1.7B-Swarm-Arena-RL-v4-focused-step60-truncated-development), revision `a64eb9278f88cd1a31528be1460e22a7fd311945`
- [Trainer metrics](https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/220pn93o)
- [Controller and evaluation metrics](https://wandb.ai/ChinmayK0604/swarm-arena-rl/runs/rl-v4-focused80-atomic-6c5eea73-l40-20260817-controller-v1)

Both Hugging Face bundles are public and were downloaded anonymously after
upload; all four adapter checksums matched. Large checkpoints and raw rollouts
were not copied to the developer Mac.

See `summary.json` for the machine-readable curve, immutable pins, hashes, and
claim boundary.
