import json

import pytest
from symbolic_tool_calling_v1.artifacts import read_jsonl, verify_artifact
from symbolic_tool_calling_v1.collection import RolloutCollectionConfig, TaskDatasetConfig
from symbolic_tool_calling_v1.compaction import CompactionConfig
from symbolic_tool_calling_v1.pipeline import PipelineConfig, run_pipeline
from symbolic_tool_calling_v1.schemas import CompactionSegment, RolloutRecord, TrainingExample


def test_end_to_end_pipeline_uses_each_frozen_stage_as_the_next_input(tmp_path, repo_root):
    output = tmp_path / "pilot-v1"
    config = PipelineConfig(
        tasks=TaskDatasetConfig(
            tasks_per_condition=1,
            horizon_buckets=("short", "long"),
            imbalance_settings=("low", "high"),
        ),
        rollouts=RolloutCollectionConfig(group_size=4),
        compaction=CompactionConfig(token_budget=80),
    )
    index = run_pipeline(output, config, repo_root)
    assert set(index) == {"tasks", "rollouts", "compaction", "training"}
    assert json.loads((output / "pipeline_config.json").read_text()) == config.model_dump(mode="json")
    for stage in index:
        assert verify_artifact(output / stage).artifact_id == index[stage]["artifact_id"]

    rollouts = read_jsonl(output / "rollouts" / "rollouts.jsonl", RolloutRecord)
    segments = read_jsonl(output / "compaction" / "segments.jsonl", CompactionSegment)
    examples = read_jsonl(output / "training" / "examples.jsonl", TrainingExample)
    assert len(rollouts) == 16
    assert {sum(rollout.prompt_id == prompt for rollout in rollouts) for prompt in {r.prompt_id for r in rollouts}} == {
        4
    }
    assert {segment.rollout_id for segment in segments} == {rollout.rollout_id for rollout in rollouts}
    assert any(segment.num_segments_total > 1 for segment in segments)
    assert len(examples) == 3 * len(segments)
    assert {example.objective for example in examples} == {
        "compacted_grpo",
        "segment_normalized_grpo",
        "compacted_ppo",
    }

    with pytest.raises(FileExistsError):
        run_pipeline(output, config, repo_root)
