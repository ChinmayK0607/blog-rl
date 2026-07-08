from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import VanillaGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class VanillaGRPOAlgorithm(Algorithm):
    """Vanilla GRPO (DeepSeekMath, https://arxiv.org/abs/2402.03300): sample a
    group of rollouts per example; credit = (reward − group mean) / group std —
    the paper's whitened advantage, where :class:`GRPOAlgorithm` is the Dr.GRPO
    revision that keeps only the mean baseline. Action tokens feed the ``rl``
    loss.

    A group with zero reward variance gets all-zero advantages (nothing to
    whiten — same signal the mean-only form gives a uniform group), which the
    zero-advantage filter then drops.
    """

    def __init__(self, config: VanillaGRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        centered = rewards - rewards.mean()
        std = rewards.std() if len(group) > 1 else rewards.new_zeros(())
        if std > 0:
            advantages = centered / std
        else:
            advantages = torch.zeros_like(centered)
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
