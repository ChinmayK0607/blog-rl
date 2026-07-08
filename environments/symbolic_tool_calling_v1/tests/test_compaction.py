from symbolic_tool_calling_v1.artifacts import read_jsonl, verify_artifact
from symbolic_tool_calling_v1.collection import (
    RolloutCollectionConfig,
    TaskDatasetConfig,
    build_tasks,
    collect_rollouts,
)
from symbolic_tool_calling_v1.compaction import (
    CompactionConfig,
    compact_rollout,
    compact_rollouts,
    group_relative_advantages,
    write_compaction_dataset,
)
from symbolic_tool_calling_v1.schemas import CompactionSegment


def sample_rollouts():
    task = build_tasks(
        TaskDatasetConfig(
            tasks_per_condition=1,
            horizon_buckets=("long",),
            imbalance_settings=("high",),
            recovery_cost=3,
        )
    )[0]
    return collect_rollouts([task], RolloutCollectionConfig())


def test_compaction_is_deterministic_lossless_and_ordered():
    rollout = sample_rollouts()[1]
    config = CompactionConfig(token_budget=80)
    first, first_summary = compact_rollout(rollout, config)
    second, second_summary = compact_rollout(rollout, config)
    assert first == second
    assert first_summary == second_summary
    rebuilt = [step for segment in first for step in segment.steps]
    assert rebuilt == list(rollout.steps)
    assert [segment.segment_index for segment in first] == list(range(len(first)))
    assert all(segment.num_segments_total == len(first) for segment in first)
    assert sum(segment.turn_count_segment for segment in first) == rollout.total_turns
    assert sum(segment.token_count_segment for segment in first) == rollout.total_tokens
    assert first[0].carried_state_summary == rollout.initial_environment_state
    for previous, current in zip(first, first[1:]):
        assert current.carried_state_summary == previous.steps[-1].environment_state


def test_budget_is_respected_except_for_indivisible_oversized_steps():
    rollout = sample_rollouts()[0]
    budget = 10
    segments, _ = compact_rollout(rollout, CompactionConfig(token_budget=budget))
    assert all(segment.token_count_segment <= budget or len(segment.steps) == 1 for segment in segments)
    assert all(segment.steps for segment in segments)


def test_compaction_exposes_segment_count_imbalance_within_prompt_group():
    rollouts = sample_rollouts()
    segments, summaries = compact_rollouts(rollouts, CompactionConfig(token_budget=80))
    assert len({summary.num_segments for summary in summaries}) > 1
    assert sum(summary.num_segments for summary in summaries) == len(segments)
    assert sum(segment.terminal_containing for segment in segments) == len(rollouts)


def test_group_advantages_are_prompt_local_normalized_and_inherited():
    rollouts = sample_rollouts()
    advantages = group_relative_advantages(rollouts)
    values = [advantages[rollout.rollout_id] for rollout in rollouts]
    assert abs(sum(values)) < 1e-12
    assert values[-1] < 0 < values[0]
    segments, _ = compact_rollouts(rollouts, CompactionConfig(token_budget=80))
    for segment in segments:
        assert segment.inherited_group_advantage == advantages[segment.rollout_id]


def test_compaction_artifact_is_valid_and_round_trips(tmp_path, repo_root):
    rollouts = sample_rollouts()
    output = tmp_path / "compact-v1"
    manifest = write_compaction_dataset(output, rollouts, CompactionConfig(token_budget=80), repo_root)
    assert verify_artifact(output) == manifest
    restored = read_jsonl(output / "segments.jsonl", CompactionSegment)
    assert len(restored) == manifest.records
