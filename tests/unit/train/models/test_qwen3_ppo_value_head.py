import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.weights import policy_only_state_dict


def tiny_config(*, value_head: bool) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.prime_rl_ppo_value_head = value_head
    return config


def test_value_head_is_opt_in_zero_initialized_and_checkpointed():
    plain = Qwen3ForCausalLM(tiny_config(value_head=False))
    critic = Qwen3ForCausalLM(tiny_config(value_head=True))
    assert plain.value_head is None
    assert critic.value_head is not None
    assert torch.equal(critic.value_head.weight, torch.zeros_like(critic.value_head.weight))
    assert "value_head.weight" in critic.state_dict()


def test_policy_broadcast_excludes_trainer_only_value_head():
    critic = Qwen3ForCausalLM(tiny_config(value_head=True))
    policy_state = policy_only_state_dict(critic.state_dict())
    assert "value_head.weight" not in policy_state
    assert "lm_head.weight" in policy_state
    wrapped = policy_only_state_dict({"_orig_mod.value_head.weight": torch.ones(1), "model.weight": torch.ones(1)})
    assert set(wrapped) == {"model.weight"}
