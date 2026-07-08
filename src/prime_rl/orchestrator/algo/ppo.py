from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import PPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class PPOAlgorithm(Algorithm):
    """Vanilla PPO (Schulman et al. 2017): actor-critic with clipped-surrogate
    policy loss and GAE advantages from a learned value head.

    Credit is per-rollout — no group baseline, so scoring happens at arrival.
    The critic lives in the trainer (``trainer.model.ppo_value_head``), so the
    orchestrator's half is only the reward: each sample ships a per-token
    ``rewards`` stream carrying the rollout's terminal reward on its last
    action token (every branch is a full alternative path through the episode,
    so each terminates with the rollout's reward). The trainer turns rewards
    plus its own value predictions into GAE advantages and lambda-return value
    targets — see ``compute_token_gae`` — and trains policy and critic in one
    pass, where the pre-update forward values are the behavior values.

    ``rollout.advantages`` stays ``None``: per-token credit is not known until
    the trainer's forward, so no advantage stream ships (the samples carry a
    zero placeholder the trainer overwrites) and advantage-based filters —
    built for group-relative credit — never fire.
    """

    def __init__(self, config: PPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)

    async def score_rollout(self, rollout: Rollout) -> None:
        for sample in rollout.samples:
            if not any(sample.mask):
                continue
            rewards = [0.0] * len(sample.token_ids)
            last_action = len(sample.mask) - 1 - sample.mask[::-1].index(True)
            rewards[last_action] = float(rollout.reward)
            sample.rewards = rewards
            sample.advantages = [0.0] * len(sample.token_ids)
