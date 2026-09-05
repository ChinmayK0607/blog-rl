# Common-randomness paired counterfactual audit (24 cases)

This training-free audit repeats 12 certified communication-critical scenarios
and their 12 matched decoys with one shared sampling namespace per pair. Within
each scenario, the actual SFT team and all four one-agent base-model
replacement branches also share the same keys. Every result therefore uses
common random numbers at both counterfactual levels.

## Immutable identity

- source: `bbb8fcc92fa07cdee6145fe3d0c568a49c17e52a`;
- base: `Qwen/Qwen3-1.7B` at
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
- SFT adapter SHA-256:
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`;
- constraint: `arena-structured-protocol-v2-xgrammar-choice-mask`;
- live diagnostic SHA-256:
  `4908e085b564e510dcb188de4afdd2078353b8cb3e887b679e7df11ab8650678`;
- hash-chained admission audit SHA-256:
  `56d987be69613652e5ab7f8a34dea23d59dc8e34330947eb53b0c469ce6c3609`.

All 24 scenarios passed exact manifest reconstruction, dynamic xgrammar masks,
independent replay, policy/token ownership, revision, and run-lock validation.
No optimizer ran.

## Result: reject training admission

Shared namespaces were verified for all 12 pairs. Nevertheless:

- critical nonzero-credit cases: 9/12; decoys: 8/12;
- critical mean absolute agent credit: 0.05350; decoys: 0.04975;
- intended sender paired-difference nonzero rate: 3/12;
- intended receiver paired-difference nonzero rate: 4/12;
- off-role paired-difference nonzero rate: 12/12;
- paired agent-credit signs: 15 positive, 7 negative, 26 zero.

The broad downstream effects of changing one agent from SFT to the base model
dominate the private-information intervention. Common randomness removes one
source of variance but does not make whole-policy replacement a sufficiently
local credit estimator. Training on these values would assign many agents
communication credit for generic behavioral differences.

## Consequence

The current environment, serving constraints, replay supervisor, and terminal
reward are mechanically RL-capable. The current whole-policy replacement
credit estimator is **not admitted for RL**. Do not route these approvals to an
optimizer and do not reinterpret the paired differences as communication
rewards.

The next iteration is CPU-side: define a more local causal intervention that
holds each agent's action policy fixed while changing only the broadcast or
delivery edge being attributed, then certify that matched decoys produce zero
credit and that off-role attribution stays below a fixed predeclared bound.
This preserves terminal-only verifiability without adding a communication
bonus or model-graded reward.
