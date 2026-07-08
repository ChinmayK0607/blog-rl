import pytest
from symbolic_tool_calling_v1.collection import (
    RolloutCollectionConfig,
    TaskDatasetConfig,
    build_tasks,
    collect_rollouts,
)
from symbolic_tool_calling_v1.compaction import CompactionConfig, compact_rollouts
from symbolic_tool_calling_v1.training import (
    TrainingTransformConfig,
    build_training_examples,
)


def sample_segments():
    tasks = build_tasks(
        TaskDatasetConfig(tasks_per_condition=1, horizon_buckets=("long",), imbalance_settings=("high",))
    )
    rollouts = collect_rollouts(tasks, RolloutCollectionConfig())
    return compact_rollouts(rollouts, CompactionConfig(token_budget=80))[0]


def test_objective_transform_is_deterministic_and_preserves_segment_identity():
    segments = sample_segments()
    config = TrainingTransformConfig()
    first = build_training_examples(segments, config)
    second = build_training_examples(list(reversed(segments)), config)
    assert first == second
    assert len(first) == 3 * len(segments)
    for objective in config.objectives:
        objective_examples = [example for example in first if example.objective == objective]
        assert {example.segment_id for example in objective_examples} == {segment.segment_id for segment in segments}


def test_segment_normalization_equalizes_total_policy_weight_per_rollout():
    examples = build_training_examples(sample_segments(), TrainingTransformConfig())
    normalized = [example for example in examples if example.objective == "segment_normalized_grpo"]
    rollout_ids = {example.rollout_id for example in normalized}
    for rollout_id in rollout_ids:
        assert sum(example.policy_weight for example in normalized if example.rollout_id == rollout_id) == pytest.approx(1.0)
    plain = [example for example in examples if example.objective == "compacted_grpo"]
    for rollout_id in rollout_ids:
        group = [example for example in plain if example.rollout_id == rollout_id]
        assert sum(example.policy_weight for example in group) == len(group)


def test_ppo_uses_terminal_only_reward_and_segment_level_gae():
    segments = sample_segments()
    config = TrainingTransformConfig(objectives=("compacted_ppo",), gamma=1.0, gae_lambda=1.0)
    ppo = build_training_examples(segments, config)
    for rollout_id in {example.rollout_id for example in ppo}:
        group = [example for example in ppo if example.rollout_id == rollout_id]
        terminal_reward = group[-1].transition_reward
        assert [example.transition_reward for example in group[:-1]] == [0.0] * (len(group) - 1)
        assert group[-1].done
        assert not any(example.done for example in group[:-1])
        assert all(example.critic_value == 0.0 for example in group)
        assert all(example.policy_advantage == terminal_reward for example in group)
        assert all(example.critic_target == terminal_reward for example in group)


def test_external_critic_bootstraps_gae_without_future_reward_leakage():
    segments = sample_segments()
    rollout_id = segments[0].rollout_id
    group = sorted((segment for segment in segments if segment.rollout_id == rollout_id), key=lambda item: item.segment_index)
    values = {segment.segment_id: 0.25 + 0.1 * index for index, segment in enumerate(group)}
    config = TrainingTransformConfig(
        objectives=("compacted_ppo",),
        gamma=0.9,
        gae_lambda=0.8,
        critic_source="external",
        critic_values=values,
    )
    with pytest.raises(ValueError, match="missing external critic values"):
        build_training_examples(segments, config)

    only_group = build_training_examples(group, config)
    final = only_group[-1]
    assert final.policy_advantage == pytest.approx(group[-1].inherited_reward - values[group[-1].segment_id])
    assert final.critic_target == pytest.approx(group[-1].inherited_reward)


def test_rejects_incomplete_compacted_rollout():
    segments = sample_segments()
    group = [segment for segment in segments if segment.rollout_id == segments[0].rollout_id]
    assert len(group) > 1
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        build_training_examples(group[:-1], TrainingTransformConfig())
