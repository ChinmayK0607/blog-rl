from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import CompactedGRPOAlgoConfig, CompactedPPOAlgoConfig, SegmentNormalizedGRPOAlgoConfig
from prime_rl.orchestrator.algo.routing import stamp_ppo_streams
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class CompactedGRPOAlgorithm(GRPOAlgorithm):
    """Online message-boundary compaction with inherited GRPO credit."""

    def __init__(self, config: CompactedGRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.token_budget = config.token_budget
        self.normalize_by_segment_count = isinstance(config, SegmentNormalizedGRPOAlgoConfig)

    async def finalize_group(self, rollouts: list[Rollout]) -> None:
        await super().finalize_group(rollouts)
        for rollout in rollouts:
            count = len(rollout.samples)
            rollout.num_compaction_segments = count
            rollout.compaction_segment_lengths = [sum(sample.mask) for sample in rollout.samples]
            for sample in rollout.samples:
                action_tokens = sum(sample.mask)
                if action_tokens == 0:
                    continue
                segment_mass = 1.0 / count if self.normalize_by_segment_count else 1.0
                per_token_weight = segment_mass / action_tokens
                sample.rl_weights = [per_token_weight if action else 0.0 for action in sample.mask]


class SegmentNormalizedGRPOAlgorithm(CompactedGRPOAlgorithm):
    pass


class CompactedPPOAlgorithm(CompactedGRPOAlgorithm):
    """Compacted actor-critic pilot with GRPO policy credit and PPO critic streams."""

    def __init__(self, config: CompactedPPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.critic_old_value = config.critic_old_value

    async def finalize_group(self, rollouts: list[Rollout]) -> None:
        await super().finalize_group(rollouts)
        for rollout in rollouts:
            # Compact GRPO has already stamped the per-token advantage stream
            # and actor weights. Add one value target per segment at the causal
            # state immediately before the segment's first action token.
            for sample in rollout.samples:
                if not any(sample.mask):
                    continue
                advantages = sample.advantages or [0.0] * len(sample.token_ids)
                first_action = sample.mask.index(True)
                policy_advantage = advantages[first_action]
                actor_weights = sample.rl_weights
                policy_weight = actor_weights[first_action] if actor_weights is not None else 1.0
                stamp_ppo_streams(
                    sample,
                    policy_advantage=policy_advantage,
                    old_value=self.critic_old_value,
                    value_target=rollout.reward,
                    policy_weight=policy_weight,
                )
