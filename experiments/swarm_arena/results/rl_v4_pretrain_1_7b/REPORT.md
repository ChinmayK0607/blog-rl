# RL v4 pre-training baseline and reward-density diagnostic

## Verdict

The 1.7B SFT warm start is suitable for a shared-terminal-reward RL pilot. It
has perfect structured-protocol and broadcast-grounding rates on this
development run, materially stronger gameplay than the base model, and
nonzero leave-one-out advantages in 95.8% of 96 stage-1 rollout replicas.
No additive reward shaping is justified by these results.

This is not an RL improvement result. No optimizer update was executed, and the
SFT checkpoint does not yet pass a causal communication claim.

## Immutable inputs

- Base: `Qwen/Qwen3-1.7B` revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- SFT: `CK0607/Qwen3-1.7B-Swarm-Arena-SFT-v2-step320-noneligible`
  revision `534522a8f3ff3489b1dd8318dc8e533e51264cde`
- Adapter file SHA-256:
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`
- Evaluation source: `3fdf72b1`; corrected live-v4 rollout source:
  `7af9d102`; content-addressed summarizer source: `473c19df`
- Hardware: one NVIDIA RTX A6000 48 GB, BF16 vLLM 0.22 serving with dynamic
  rank-16 LoRA adapters

## 96-game online development baseline

The progress runner's fixed names are historical: `candidate_rl` denotes the
SFT adapter in this pre-training run, while `sft_init` denotes the untouched
base model. The immutable revisions in `manifest.json` are authoritative.

- action protocol: `1.0`
- broadcast protocol: `1.0`
- grounded broadcasts: `1.0`
- SFT minus base, legacy ordinary return: `+0.28685`, 95% interval
  `[+0.11915, +0.46530]`
- SFT minus base, hard ordinary return: `+0.20314`, 95% interval
  `[+0.15727, +0.27206]`
- critical normal minus dropped: `+0.05976`, 95% interval
  `[-0.00226, +0.10797]`
- matched-decoy normal minus dropped: `+0.03884`, 95% interval
  `[-0.03343, +0.11110]`

Interpretation: the warm start can play, but the four-bundle online tier does
not establish causal communication. That leaves measurable headroom for RL.

## Stage-1 shared-return diagnostic

The diagnostic used the exact 50/25/25 curriculum mixture: 12 ordinary groups,
six critical handoffs, and six matched decoys. Each group contained four
independent complete-game replicas. The environment replayed every transition,
verified terminal team return, and signed four isolated policy routes. Only
first-turn BLUE broadcast spans were eligible for this initial curriculum
stage. No batch entered an optimizer.

- groups / complete games: `24 / 96`
- overall mean return: `+0.02611`
- return range: `[-0.30, +0.62069]`
- overall mean absolute advantage: `0.10428`
- nonzero-advantage rate: `0.95833`
- groups with within-group return variance: `0.95833`
- ordinary mean absolute advantage: `0.14889`
- critical mean absolute advantage: `0.05979`
- decoy mean absolute advantage: `0.05954`
- critical minus matched-decoy mean return: `-0.02124`

The final line is expected to be unimpressive before RL and is not itself a
communication estimator. The useful result here is that a pure terminal reward
already supplies abundant within-state learning signal without introducing a
hackable speaking, capture, or message bonus.

## Failures preserved

Two initial v4 rollout attempts failed before any model decision or admitted
sample. The controller first used the legacy v3 scenario reconstructor, then
assumed a v3 single-state object instead of selecting one of v4's two latent
worlds. Commits `c1c2fdb2` and `7af9d102` corrected those assumptions. Latent
worlds are now selected deterministically and balanced by pair index; critical
and matched-decoy groups share the same selected world.

The final Linux suite passed `90` tests with two third-party SWIG deprecation
warnings. The next experiment is a multi-GPU asynchronous optimizer run using
the same terminal reward and stage-1 mixture, followed by the 96-game online
monitor. The 1,296-game selection and 3,168-game frozen suites remain closed.

The public raw-evidence release attachment has SHA-256
`162eae39d36cc7906c2d865ed924d8eaa140237b8d10910c50f966ac695f8fd0`.
It contains the compressed raw evaluation, signed admission/evidence JSONL, and
runtime logs; supervisor signing keys are explicitly excluded.
The recomputable compact reward-density summary has SHA-256
`8618fb4bebf9e165ba40f997efd405f9f296a8f9ac18900db91cd1bea62d4a27`.

Public raw evidence:
`https://github.com/ChinmayK0607/blog-rl/releases/tag/swarm-arena-v4-pretrain-2026-08-15`
