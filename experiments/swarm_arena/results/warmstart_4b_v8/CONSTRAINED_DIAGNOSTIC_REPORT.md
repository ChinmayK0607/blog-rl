# 4B V8 constrained diagnostic report

Status: diagnostic-only and non-RL-ready. Frozen best diagnostic checkpoint step 256 is noneligible: overall schema 0.998641, broadcast grounded 0.978125, broadcast legal 0.950000. No checkpoint passed all strict gates, so no paired regression or promotion was run.

Mechanics: 24/24 games completed; broadcasts 960/960 strict-valid and grounded; actions 960/960 strict-valid; zero invalid actions/broadcasts. This demonstrates constraint enforcement, not learned protocol competence. Throughput: 45,942 completion tokens in 337.779 seconds = 136.012 tokens/s.

Behavior: adapter/base state change 24/24 each. Adapter used 7 action kinds; base 6. Adapter/base terminal-return SD 3.849/3.173; mean communication spend 31.083/31.708; duplicate-target rates 0.3125/0.2049. Near-budget spending is not evidence of natural communication skill.

Paired side-swapped adapter-v-base: generated focal return +1.668750, dropped +2.187500, generated-minus-dropped -0.518750, 95% CI [-3.0875, 2.88125], p=0.75. The intervention effect is inconclusive and not strategic evidence.

No matched unconstrained 4B 24-game diagnostic exists. Retained old-adapter baseline is engineering two-game and sequential V8 validation evidence.

Constraint-bias artifacts structured_order_ab.json and structured_order_repeat_collision.json: normal versus reversed/random enum order differed on 1/12 prompts, equal to ordinary greedy repeat disagreement 1/12; forced two-fact items.anyOf stress produced zero duplicate facts in 36 outputs. This supports exact enumeration for the bounded diagnostic but does not prove zero content distortion or collision risk.
