# Compaction-aware RL implementation plan

The immutable data path is:

`raw rollout -> compacted segments -> objective-specific training examples -> optimizer`

Implemented now:

- fixed-token compaction with exact carried environment state;
- inherited prompt-group advantage for compacted GRPO;
- `1 / num_segments` policy weights for the rollout-normalized GRPO ablation;
- terminal-only transition rewards and segment-level GAE for compacted PPO;
- explicit zero-baseline and external causal critic-value modes;
- versioned, checksummed training-example artifacts.

The PPO critic must estimate `V(carried_state_summary)` before observing the segment's actions or
terminal outcome. For segment `i`, the transform computes
`delta_i = r_i + gamma * (1-done_i) * V(s_{i+1}) - V(s_i)` and reverse-recursive GAE. Only the final
segment receives the environment's terminal reward. The critic target is `V(s_i) + A_i`.

Next framework slice:

1. Add critic values/targets and critic-loss weights to Prime-RL's transport structures.
2. Add a scalar value head to the policy backbone and checkpoint it with trainer/optimizer state.
3. Compute clipped PPO policy loss plus masked value loss; log explained variance and value error.
4. Export causal critic predictions during collection, then rebuild the cheap training transform.
5. Run deterministic transport/loss/checkpoint tests before the first PPO GPU pilot.

The initial 10-step online pilot remains GRPO because current Prime-RL has no value head. It uses all
eight H100s as six data-parallel inference replicas and two trainer GPUs. PPO must not be labeled as
implemented until the value head, critic optimizer path, and checkpoint/resume tests pass.
