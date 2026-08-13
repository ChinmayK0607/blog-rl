import pytest
import torch

from prime_rl.configs.trainer import (
    CustomLossConfig,
    DefaultLossConfig,
    RolloutParityGateConfig,
)
from prime_rl.trainer.rl.loss import (
    LossInputs,
    LossOutputs,
    compute_constrained_entropy,
    compute_entropy,
    compute_loss,
    rollout_parity_metrics,
    selective_constrained_log_softmax,
    setup_loss_fns,
    validate_rollout_parity_metrics,
)

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_constrained_logprobs_and_entropy_renormalize_over_legal_tokens():
    logits = torch.tensor([[[0.0, 1.0, 2.0, 7.0]]], device="cuda")
    selected = torch.tensor([[2]], device="cuda")
    allowed = torch.tensor([[[1, 2, -1]]], device="cuda")
    logprob = selective_constrained_log_softmax(logits, selected, allowed)
    expected = torch.tensor(2.0, device="cuda") - torch.logsumexp(
        torch.tensor([1.0, 2.0], device="cuda"), dim=0
    )
    assert torch.allclose(logprob, expected.reshape(1, 1))
    probabilities = torch.softmax(torch.tensor([1.0, 2.0], device="cuda"), dim=0)
    expected_entropy = -(probabilities * probabilities.log()).sum()
    entropy = compute_constrained_entropy(logits, allowed)
    assert torch.allclose(entropy, expected_entropy.reshape(1, 1))


def test_rollout_parity_gate_fails_before_an_out_of_envelope_update():
    metrics = rollout_parity_metrics(
        torch.tensor([0.001, 0.002, 0.003], device="cuda"),
        torch.tensor([0.001, 0.002, 0.08], device="cuda"),
        torch.tensor([0.0, 0.0, 0.01], device="cuda"),
        probability_tail_threshold=0.05,
    )
    permissive = RolloutParityGateConfig(
        max_p99_probability_error=0.1,
        max_probability_tail_fraction=0.5,
        max_mean_mismatch_kl=0.01,
    )
    validate_rollout_parity_metrics(metrics, permissive)
    with pytest.raises(RuntimeError, match="numerical-parity gate failed"):
        validate_rollout_parity_metrics(metrics, RolloutParityGateConfig())


def test_setup_loss_fns_with_custom_config():
    """Test setup_loss_fns with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    loss_fns = setup_loss_fns(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        teacher_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = loss_fns["rl"](inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_sft_loss_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=2,
        training_mode="sft",
    )

    # loss = -sum(masked logprobs) / loss_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics


def test_sft_loss_override_uses_masked_nll_with_default_loss_config():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.ones(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=2,
        training_mode="sft",
    )

    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
