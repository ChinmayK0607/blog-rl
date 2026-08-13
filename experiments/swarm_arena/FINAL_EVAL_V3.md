# Swarm Arena final evaluation v3

The final evaluation separates capability, communication, semantics,
specialization, and opponent robustness. No single aggregate score is used.
`final_eval_runner.py` runs four independently assigned models per team through
the existing concurrent cross-play engine and emits the strict row schema
consumed by `final_eval_v3.py`.

## Frozen suites

1. The unchanged 72-case episode-v2 OOD suite measures long-horizon gameplay on
   graph sizes 14/16 and horizons 6/8.
2. The added certified OOD suite contains 72 critical examples and their 72
   matched decoys. Its seeds and manifests are disjoint from training and
   development.
3. The frozen non-arena regression suite measures retained general capability.

## Opponents and side control

Every learned swarm is evaluated against at least three immutable
model-controlled opponent swarms: base, SFT, and historical-league snapshot.
Every case is played with sides swapped. Opponents never use a deterministic
policy in the headline result. For certified critical/decoy cases, the side
swap exchanges team and agent labels in the frozen state so the focal swarm—not
its opponent—retains the same private-information problem on both sides.

## Interventions

On identical cases and sampling keys, run normal, dropped, sender-shuffled,
one-turn-delayed, and zero-budget communication. Also run:

- all non-identity permutations of the four learned adapters;
- role-label permutations while preserving the physical state;
- action-option permutations;
- an action-only training control;
- the untouched SFT initialization.

The action-only control is trained on the same maps and terminal objective with
communication disabled. It separates improvements due to action learning from
improvements that require a learned communication channel.

## Predeclared endpoints

Primary capability endpoint: paired, side-averaged terminal return of RL minus
the SFT initialization on ordinary OOD games.

Communication endpoints: paired normal-minus-dropped,
normal-minus-sender-shuffled, normal-minus-delayed, and
normal-minus-zero-budget return on certified critical cases.

Mechanism endpoints: critical capture rate, matched-decoy message rate,
duplicate-target turns, message-budget exhaustion, action entropy, adapter
shuffle effect, role permutation effect, and per-opponent return.

Use seed-level paired bootstrap 95% intervals with 20,000 deterministic
resamples. Side swaps, opponents, intervention rows, and adapter permutations
are averaged inside each seed. Keep each seed—not each agent, side, opponent,
or intervention row—as the independent statistical unit.

## Claim gates

- Higher ordinary return with no communication effects is capability learning.
- Communication is supported only if all four intervention intervals exclude
  zero in the positive direction and the effect persists across unseen
  opponents and side swaps. The matched-decoy normal-minus-dropped interval
  must include zero, guarding against a general penalty caused by deleting any
  message.
- Stable specialization is supported only if adapter shuffling harms return,
  role-label permutation alone does not explain the effect, and communication
  interventions remain causal.
- Role-label and legal-action-option robustness use a predeclared equivalence
  margin: the entire paired 95% interval for identity-minus-permuted return must
  lie inside `[-0.02, 0.02]`.
- A broad emergent-swarm claim is out of scope even if these gates pass; the
  supported claim is learned decentralized coordination inside Swarm Arena.

Protocol validity produced by structured decoding is reported as an
infrastructure invariant, never as learned competence.

Checkpoint selection uses the development split. The frozen final matrix is
run once per selected checkpoint; repeatedly selecting against it invalidates
its confirmatory status. The committed design records 4,320 headline games
(864 ordinary candidate/SFT, 432 action-only control, 2,160 critical
interventions, and 864 matched decoys), plus a fixed 12-case mechanism subset.
