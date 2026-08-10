# SFT warm-start data plan

## Objective

Teach a 1–3B instruct model the simulator mechanics, observation discipline,
strict semantic broadcast protocol, and legal-action interface. Do not teach the
final swarm policy with SFT; multi-agent RL must retain meaningful coordination
behavior to discover.

All examples use the same versioned prompts as evaluation. Targets contain only
strict JSON. Solver traces and rationales remain metadata and are never placed in
the assistant target.

## Example types

1. **Broadcast**
   - Input: objective, agent identity, private observation, legal actions.
   - Target: up to three exact timestamped facts, an optional legal intent, an
     optional one-resource request, or a fully empty broadcast.
   - Teach silence explicitly so agents do not flood the channel.

2. **Local action mechanics**
   - Input: private observation, empty inbox, legal actions.
   - Target: one legal action ID.
   - Cover every mechanic, resource boundary, stale-event rule, and action-order
     permutation.

3. **Message-conditioned grounding**
   - Input: private observation plus factual teammate messages and declared
     intents.
   - Target: the best legal action after incorporating a clear factual update or
     one explicit teammate intent.
   - Use only simple two-way disambiguations here. Reserve multi-agent allocation,
     negotiation, and difficult collision avoidance for RL so SFT does not bake in
     the result we want to study.

4. **Robustness**
   - Input: missing, stale, irrelevant, or explicitly untrusted messages.
   - Target: the action supported by fresh trusted evidence.
   - Do not train on arbitrary malicious text; perturb only simulator fields.

## Generation pipeline

1. Sample a small graph, node states, objectives, resources, event freshness, and
   four agents' overlapping but incomplete observations.
2. Enumerate each agent's legal actions from simulator rules.
3. Use an exact joint-action solver to find every maximum-reward assignment.
4. Keep an action row only when the agent has one acceptable action across the
   complete optimal set, that action agrees with a prompt-visible local policy,
   and no teammate declares the same intent. Reject everything else.
5. Derive broadcasts only from the sender's observation. Validate every fact and
   intent through the same strict parser used at evaluation time.
6. Replay the state with generated, dropped, oracle, stale, and shuffled inboxes;
   solve again rather than reusing a label that may no longer be correct.
7. Randomize node names, agent identities, action ordering, surface wording, and
   irrelevant observations. Preserve the underlying solution.
8. Run a second independent audit that reconstructs every procedural state,
   verifies all targets and solver membership, checks split isolation, and hashes
   the complete dataset.

## Initial scale

- First validate a 128-seed CPU pilot plus targeted coverage cases.
- Scale only after the pilot passes the independent audit and a base-model run.
- Use targeted solver-certified cases to cover rare `WAIT`, `SCAN`, and
  `TRANSFER` mechanics rather than accepting weak or ambiguous random labels.
- Split by generator seed and topology family, not by individual row:
  90% train, 5% validation, 5% synthetic test.
- Keep the current hand-written evaluation suite completely outside training.

## Teacher use

The exact solver supplies action labels and broadcasts use exact structured facts,
so the primary dataset requires no teacher model. Free-form paraphrasing is not
part of the primary SFT protocol and cannot contaminate its score.

## Training

A single 24GB RTX 4090 is sufficient for LoRA SFT of a 1.7–2B model with short
sequences. Begin with a small pilot (about 5,000 examples), evaluate protocol and
mechanics, inspect failures, then train the full set. Keep reasoning disabled and
train only the final JSON response. Multi-agent RL starts only after the model
passes the frozen mechanics and communication gates.

## Leakage controls

- Never include required joint actions, oracle messages, rewards, or scorer errors
  in model inputs.
- Never train on the fixed evaluation scenarios or simple renamings of them.
- Record prompt version, generator version, seed, topology family, solver result,
  and validator status with every row.
- Freeze train/validation/test manifests before comparing checkpoints.
