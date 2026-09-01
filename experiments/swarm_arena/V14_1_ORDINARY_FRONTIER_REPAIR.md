# V14.1: Ordinary Frontier Repair

Status: CPU repair implemented and audited. No GPU is rented and no optimizer
update has run. The next action is one bounded 128-game, zero-update screen.

## What failed in V14

V14 reached a valid runtime certificate and then correctly stopped before update
1. The complete 64-case / 256-game ordinary screen had protocol admission `1.0`,
but blue-1 and blue-2 each produced variable returns in only `3/16` cases
(`0.1875`) versus the frozen `0.25` minimum. Neither had a variable case against
the current-policy opponent. Blue-1 also had mean focused-action diversity
`1.1875` versus the required `1.25`.

This is not evidence that V13 lost its communication mechanism, because the
screen measured ordinary terminal-return variation before training. It is
evidence that the old fixed ordinary set was locally saturated for two policy
slots and could not reliably produce policy-gradient comparisons.

## Repair boundary

The repair uses only the completed V14 training-only screen. It does not inspect
development or frozen cases, weaken a gate, search GPU seeds, change a reward,
or run an optimizer. It creates a 128-case immutable pool:

- 19 observed frontier cases;
- 27 observed mastered anchors;
- 18 observed stalled anchors;
- 32 cross-opponent transfers of observed frontier cases;
- 32 deterministic unseen neighbours of observed frontier cases.

The last two classes are marked unseen, not successful. They remain frontier
exploration only until a complete logical group is observed.

Every case is bound to one exact policy slot and opponent family. Adaptive
selection may change the seed/size/horizon case inside that cell, but it cannot
change focused-policy balance, opponent rotation, the stage group mix, reward,
counterfactual, or evaluation data. Each next 10-update stage uses only the
immediately preceding complete training stage to classify cases, then selects
80% frontier, 10% mastered, and 10% stalled anchors. Resume must reproduce the
same atomic selection artifact or fail closed.

## Bounded screen before update 1

The new screen contains 32 cases / 128 games: eight cases per policy and exactly
two per policy/opponent-family cell. Its thresholds are byte-for-byte the V14
thresholds:

- protocol admission `1.0`;
- per-policy variable-group and nonzero-advantage rates at least `0.25`;
- per-policy mean focused-action diversity at least `1.25`;
- at least four positive and four negative advantages per policy;
- at least one variable group for every policy/opponent-family cell.

For blue-1 and blue-2 against the current opponent, the cases are one
cross-opponent transfer and one deterministic unseen neighbour. No failed V14
current-opponent case is silently relabelled as frontier.

If this screen fails, update 1 remains prohibited and the exact pod should be
synced and removed immediately. The failure should trigger a critic/capacity or
scenario-generator decision on CPU, not another seed search. If it passes, the
existing maximum-40-update V14 stage plan may run with unchanged gates at
updates 10, 20, 30, and 40.

## Compute discipline

Before another rental:

1. commit and publish the exact source and verify it anonymously;
2. verify that HF write credentials are actually available locally;
3. freeze the launch bundle, screen manifest, source SHA, and runtime config;
4. arm the watcher and immediate exact-pod teardown path before GPU setup.

On the pod, downloads and environment preparation should be concurrent where
safe. The runtime certificate and 128-game screen are the only work allowed
before update 1. A failed screen or stage gate triggers evidence sync and
immediate teardown; verified completion does the same after checkpoint, mirror,
and W&B finalization.

## Frozen CPU identities

- failed V14 assessment SHA-256:
  `5e72fd994091608ef99518b646f8ddc6df5b44607e032c9068e845b519906884`;
- ordinary case-pool body SHA-256:
  `521fb25bdd04d45524124bb8bea3f1aec1958b141ee01730f1ecfc6e51103d28`;
- repair-screen body SHA-256:
  `6ce147cf145918ab4f32d56512b500f63a34bc3bd1981585afee985033c0df7b`;
- repaired-curriculum body SHA-256:
  `eb40420decf44d8b58b8e8a7a33333a710ef5c445a249bebe1a5fcb93578d5e1`;
- repair-audit body SHA-256:
  `4711dba42b4c2c038d04b583fb324f5dcc5360c0f8215da23c4e6bd425bc1a2d`;
- credential-free CPU-bundle body SHA-256:
  `ce84f961050b470407ddc3c18f41223dd55c81c37b81563a82922d2d2e12bae4`.

The CPU dry-run built a 40-update, 160-group plan with 50 ordinary, 70
critical, and 40 decoy groups. The pool is embedded in the production-plan
identity; a one-byte pool mutation fails loading before rollout.
