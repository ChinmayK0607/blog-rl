# Receiver terminal-proximal curriculum screen

Training-split diagnostic only. No model weights changed, and no development,
selection, or frozen OOD case was opened.

The pinned Qwen3-1.7B SFT initializer played both teams in 12 pass@k-selected
handoff worlds and their exact matched decoys. Each generated/dropped condition
used four common-seed repetitions. Both screens contain 192 complete 4v4 games
with 100% action, broadcast, and grounding validity.

| Remaining turns | Wall time | Critical return-contrast cells | Receiver-action contrast | Nonzero-return cells | Terminal critical-minus-decoy specificity | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 202.44 s | 50.0% | 75.0% | 50.0% | +0.00259 `[0.00000, +0.00617]` | reject: reward too sparse |
| 2 | 501.33 s | 83.3% | 83.3% | 91.7% | +0.01213 `[-0.00410, +0.02962]` | adopt for receiver curriculum |

The two-turn specificity change versus the same original-horizon repetitions is
`+0.03411`, clustered bootstrap interval `[-0.01696, +0.07902]`. This is an
unresolved positive diagnostic, not evidence that RL learned communication.
The useful finding is that two remaining turns preserve stochastic terminal
reward and receiver-action variation, whereas one remaining turn removes too
much learning signal.

Selected curriculum:

- updates 0--19: two remaining handoff turns;
- updates 20--39: original per-scenario horizon;
- receiver `ACT` focus only;
- 40 ordinary, 60 critical, and 60 exactly matched-decoy groups;
- unchanged verified terminal control-delta reward and no shaping.

Compact artifact SHA-256 values:

- one-turn manifest: `10400bf12c91d9560f3d93bd398e110095d1c077ea1db1ca212effe19c4e512e`
- one-turn rows: `29e12ecbb98ad9845cc07d11e9b791ffa22024b202625dc473a40cf7ecee49b3`
- one-turn summary: `0f3355ce4e881e967355f442da76bbc216f70ab366e1bb9bc7af7d6c39f158ce`
- two-turn manifest: `a9021c5097c7eff5804e6978cfe3617bd2427e63e863b84d68b2dd29509bdd8d`
- two-turn rows: `7be6dc43b04ab33a691cde6db96445be8d4717d1aa8a1eccd0559a2f694f777f`
- two-turn summary: `d21b14c020801fbf879e95178be8389f317dce28387ae6e5ab2e18cb91fb4cfb`

Rollout source commit: `13526728`. Final analyzer/curriculum commit is recorded
by the containing Git revision.
