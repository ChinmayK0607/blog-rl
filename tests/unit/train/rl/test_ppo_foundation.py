import pytest
import torch

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.orchestrator.algo.routing import stamp_ppo_streams
from prime_rl.trainer.batch import pad_micro_batch, prepare_sample
from prime_rl.trainer.rl.loss import PPOLossInputs, ppo_actor_critic_loss
from prime_rl.trainer.rl.value_head import PPOValueHead
from prime_rl.transport import TrainingSample


def test_critic_streams_survive_truncation_and_padding_aligned():
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4, 5],
        mask=[False, True, True, True, True],
        logprobs=[0.0] * 5,
        temperatures=[1.0] * 5,
        env_name="ppo-test",
        advantages=[0.0, 1.0, 1.0, 1.0, 1.0],
        old_values=[0.0, 0.1, 0.2, 0.3, 0.4],
        value_targets=[0.0, 0.4, 0.5, 0.6, 0.7],
        value_weights=[0.0, 1.0, 0.0, 1.0, 0.0],
    )
    micro = prepare_sample(sample, seq_len=4)
    assert micro.old_values == [0.0, 0.1, 0.2, 0.3]
    assert micro.value_targets == [0.0, 0.4, 0.5, 0.6]
    assert micro.value_weights == [0.0, 1.0, 0.0, 1.0]
    padded = pad_micro_batch(micro, pad_to_multiple_of=8)
    assert len(padded.input_ids) == 8
    assert padded.old_values[-4:] == [0.0] * 4
    assert padded.value_targets[-4:] == [0.0] * 4
    assert padded.value_weights[-4:] == [0.0] * 4


def test_ppo_clipping_uses_behavior_logprobs_and_old_values():
    inputs = PPOLossInputs(
        trainer_logprobs=torch.log(torch.tensor([1.5, 0.5])),
        inference_logprobs=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        values=torch.tensor([1.0, -1.0], requires_grad=True),
        old_values=torch.zeros(2),
        value_targets=torch.tensor([0.5, -0.5]),
        policy_mask=torch.ones(2, dtype=torch.bool),
        value_mask=torch.ones(2, dtype=torch.bool),
    )
    result = ppo_actor_critic_loss(inputs, policy_clip=0.2, value_clip=0.2, value_coef=1.0)
    assert result.metrics["ppo/clip_fraction"] == 1.0
    assert result.metrics["ppo/value_loss"] > 0
    result.loss.backward()
    assert inputs.values.grad is not None


def test_value_head_is_scalar_zero_initialized_and_trainable():
    head = PPOValueHead(8)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    values = head(hidden)
    assert values.shape == (2, 5)
    assert torch.equal(values, torch.zeros_like(values))
    values.sum().backward()
    assert head.proj.weight.grad is not None


def test_ppo_configuration_requires_value_head_both_ways():
    with pytest.raises(ValueError, match="requires trainer.model.ppo_value_head"):
        TrainerConfig(loss={"type": "ppo"})
    with pytest.raises(ValueError, match="requires trainer.loss.type='ppo'"):
        TrainerConfig(model={"ppo_value_head": True})
    config = TrainerConfig(loss={"type": "ppo"}, model={"ppo_value_head": True})
    assert config.model.ppo_value_head


def test_ppo_stamping_places_critic_target_before_first_action():
    sample = TrainingSample(
        token_ids=[10, 11, 12, 13],
        mask=[False, False, True, True],
        logprobs=[0.0] * 4,
        temperatures=[1.0] * 4,
        env_name="ppo-test",
    )
    stamp_ppo_streams(sample, policy_advantage=0.75, old_value=0.2, value_target=0.8, policy_weight=0.5)
    assert sample.advantages == [0.0, 0.0, 0.75, 0.75]
    assert sample.rl_weights == [0.0, 0.0, 0.5, 0.5]
    assert sample.old_values == [0.0, 0.2, 0.0, 0.0]
    assert sample.value_targets == [0.0, 0.8, 0.0, 0.0]
    assert sample.value_weights == [0.0, 1.0, 0.0, 0.0]
