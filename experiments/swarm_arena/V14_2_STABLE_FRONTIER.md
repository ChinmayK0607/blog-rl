# V14.2: Stable-Frontier Pilot and Pilot-Bound Curriculum

Status: CPU design, implementation, tests, and credential-free freeze complete.
No GPU is rented, no optimizer update is authorized, and no development or
frozen evaluation data has been opened.

## What V14.1 actually taught us

V14.1 stopped because neither blue-0/current-policy case produced return or
action variation. That did not establish a current-policy capacity limit. The
partial screen contains a stronger control: the exact blue-0/current scenario
`ordinary-screen-blue-0-15` had been variable in V14, then produced four
identical actions and flat returns in V14.1.

Across the seven previously observed V14 frontier cases that V14.1 completed,
only four remained frontier on the new four-replica block. Retention was
therefore `4/7 = 0.5714`. Base and SFT cases for blue-0/1 remained variable;
historical cases for both policies and the blue-0 current case flattened.
This is direct evidence that one pass@4 block is a noisy case classifier.

The shared-return critic is not the demonstrated bottleneck. Whenever focused
actions varied, terminal returns varied and the existing leave-one-out baseline
produced both positive and negative advantages. The failed groups had identical
actions and identical returns. V14.2 therefore changes scenario selection, not
the reward, terminal objective, or credit estimator.

## Disjoint zero-update pilot

The next paid action is one bounded 24-case / 96-game training-only pilot:

- four blocking non-current candidates per policy;
- at least one candidate from each of base, SFT, and historical opponents;
- two current-policy probes per policy, both created by transferring an
  observed frontier scenario across opponent family;
- four replicas per case;
- no retry-until-variable behavior and no GPU seed search;
- pilot trajectories are discarded and can never enter an optimizer batch.

The current-policy probes answer whether same-frontier competition exposes a
usable action boundary. They are diagnostic rather than a hard admission cell.
A flat current probe is retained as an anchor/probe, but it cannot veto real
signed signal against three independent opponent families.

The blocking gate is evaluated per policy over its four non-current candidates:

- protocol admission exactly `1.0`;
- at least two variable groups;
- variable groups from at least two opponent families;
- nonzero-advantage rate at least `0.25`;
- mean focused-action diversity at least `1.25`;
- at least four positive and four negative advantages.

This gate is still fail-closed. It removes the statistically brittle demand
that every policy/opponent cell pass one of only two four-sample cases; it does
not lower the aggregate signed-signal, diversity, or protocol requirements.

## Why selection does not bias the policy gradient

The pilot deliberately changes the training distribution by selecting scenario
identities near the current policy frontier. That is the curriculum. It does
not condition an optimizer update on favorable trajectories:

1. the pilot is completed once under its immutable manifest;
2. the complete result selects and classifies scenario identities;
3. every pilot trajectory is discarded;
4. training uses new sampling namespaces and fresh rollouts;
5. the existing on-policy shared-return estimator is applied only to those
   fresh training groups.

The gradient is therefore for the explicitly selected training distribution,
not a gradient computed from cherry-picked pilot samples.

## Fail-closed pilot-to-training transition

The committed assessor is the only gate implementation. If it fails, optimizer
update 1 remains prohibited and the exact pod is synced and removed. If it
passes, the committed finalizer:

- binds every pilot result back to its exact pool case;
- classifies variable cases as frontier, positive-flat cases as mastered, and
  nonpositive-flat cases as stalled;
- creates a 24-case ordinary pool covering all 16 policy/opponent cells;
- retains current-policy probes as nonblocking anchors;
- embeds the exact pilot-assessment hash and pool into a new curriculum;
- authorizes at most the existing 40-update V14 stage plan.

The production plan must then be rebuilt from that pilot-bound curriculum.
Update 1 is forbidden if any binding, body hash, public mirror, fresh sampling
namespace, or runtime certificate is missing.

## What remains unchanged

- four distinct V13 update-80 initializers;
- rank-32 Qwen3-4B LoRA and optimizer settings;
- critical, decoy, and ordinary rewards;
- leave-one-out ordinary shared-return credit;
- the four ten-update stage shapes and multi-turn receiver offsets;
- development gates at updates 10/20/30/40;
- frozen evaluation and final selection rules;
- `$15` maximum spend and nine-hour TTL;
- immediate exact-pod teardown on rejection or verified completion.

## CPU artifacts

- `results/rl_v14_1_zero_update_rejection/PARTIAL_SCREEN_ASSESSMENT.json`;
- `data/rl_v14_2/diagnosis.json`;
- `data/rl_v14_2/pilot_screen_manifest.json`;
- `data/rl_v14_2/curriculum.json` (pre-pilot template);
- `data/rl_v14_2/audit.json`;
- `data/rl_v14_2/cpu_bundle.json`;
- `scripts/assess_v14_2_stable_frontier_screen.py`;
- `scripts/finalize_v14_2_stable_frontier.py`.

The post-pilot `ordinary_case_pool.json`, `curriculum.json`, and
`finalization_audit.json` do not exist yet by design. They can be produced only
from a complete, passed, hash-valid pilot assessment.

## Compute decision

Do not rent yet. The next source commit should include this complete CPU bundle,
then be published and anonymously verified. Credentials, watcher, evidence
mirror, immediate teardown command, and concurrent setup plan must be verified
before allocation. Once those checks exist, the only pre-training GPU science
work is the 96-game pilot—not another broad screen or exploratory seed search.
