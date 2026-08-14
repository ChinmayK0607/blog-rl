# Base-counterfactual rollout audit (24 cases)

This is a training-free pre-RL diagnostic over 12 certified
communication-critical scenarios and their 12 matched decoys. Four independent
BLUE policies used the same Qwen3-1.7B SFT adapter; RED used that frozen SFT
adapter; one BLUE agent at a time was replaced by the pinned no-adapter base
model. The only return is normalized terminal team-control delta.

## Immutable identity

- source: `ff5772127514edbaefdac99a842a2a09ff0e554c`;
- base: `Qwen/Qwen3-1.7B` at
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
- SFT adapter SHA-256:
  `2dc1694c35a414cef254273f6daf3a4ea1e611856c9d0c3d815eec60428f949b`;
- constraint: `arena-structured-protocol-v2-xgrammar-choice-mask`;
- live diagnostic SHA-256:
  `e526c0aeab16860509729677f496ccfe814ab7ed914a3be7c59402ea79176d28`;
- hash-chained admission audit SHA-256:
  `49f8b6e8d1971350f722e1fbac2ea9391df63922db7eda22169f6b5922128cb4`.

Every branch passed independent replay, policy/token ownership, dynamic
xgrammar constraint, revision, and run-lock checks. No optimizer ran.

## Result

Critical cases had nonzero replacement credit in 7/12 cases versus 5/12
decoys. Mean absolute agent credit was 0.05045 versus 0.02841. Intended senders
were nonzero in 5/12 critical cases and intended receivers in 3/12. However,
6/12 critical cases also credited at least one off-role agent; 3/12 decoys had
off-role credit. Critical mean return was +0.02769 and decoy mean return was
-0.04444.

This establishes that the base comparator can produce optimizer signal, but it
does **not** certify communication-specific credit. The false/off-role rate is
too high for training admission.

## Known design limitation

Critical and decoy rollouts in this audit used different sampling namespaces.
Their per-case replacement branches used common random numbers internally, but
the matched critical/decoy pair did not. Therefore the paired differences in
`summary.json` mix the information intervention with sampling noise and must
not be used as training advantages.

The next admission test must give each matched critical/decoy pair one shared
sampling namespace and compute a signed, terminal-only difference-in-
differences credit:

`(critical actual - critical replacement_i) -
 (decoy actual - decoy replacement_i)`.

This cancels generic base-versus-SFT action effects while retaining the
agent-specific marginal value caused by the private-information intervention.
No communication bonus, heuristic action reward, or model-graded signal is
introduced.
