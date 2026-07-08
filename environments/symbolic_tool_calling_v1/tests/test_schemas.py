from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from symbolic_tool_calling_v1 import Action, generate_task, initial_state
from symbolic_tool_calling_v1.schemas import (
    ArtifactManifest,
    ChecksumEntry,
    CompactionSegment,
    ExperimentManifest,
    RolloutRecord,
    RolloutStep,
)


def make_step(turn_index=0):
    task = generate_task(1)
    return RolloutStep(
        turn_index=turn_index,
        model_output='{"tool":"inspect"}',
        tool_call=Action(tool="inspect"),
        tool_response='{"room":"room_00"}',
        environment_state=initial_state(task),
        input_token_count=2,
        output_token_count=1,
        response_token_count=1,
    )


def make_rollout():
    step = make_step()
    return RolloutRecord(
        rollout_id="rollout-1",
        task_id="task-1",
        prompt_id="prompt-1",
        policy_id="policy-1",
        policy_checkpoint="scripted",
        sampling_config_hash="abc",
        environment_version="1.0.0",
        rollout_seed=3,
        full_prompt_text="prompt",
        initial_environment_state=step.environment_state,
        steps=(step,),
        success=False,
        terminal_reward=0.0,
        total_turns=1,
        total_tokens=4,
        stop_reason="policy_stopped",
    )


def test_rollout_round_trips_as_json_without_losing_types():
    rollout = make_rollout()
    restored = RolloutRecord.model_validate_json(rollout.model_dump_json())
    assert restored == rollout


@pytest.mark.parametrize(
    "updates",
    [
        {"total_turns": 2},
        {"total_tokens": 99},
        {"success": True},
        {"steps": (make_step(turn_index=1),)},
    ],
)
def test_rollout_rejects_inconsistent_derived_fields(updates):
    with pytest.raises(ValidationError):
        RolloutRecord.model_validate({**make_rollout().model_dump(), **updates})


def test_compaction_segment_enforces_bounds_and_counts():
    step = make_step()
    segment = CompactionSegment(
        rollout_id="rollout-1",
        compaction_version="1.0.0",
        compaction_method="fixed_token_budget",
        segment_id="segment-1",
        segment_index=0,
        num_segments_total=1,
        token_count_segment=4,
        turn_count_segment=1,
        segment_position_bucket="late",
        carried_state_summary=step.environment_state,
        inherited_reward=0.0,
        segment_start_turn=0,
        segment_end_turn=1,
        terminal_containing=False,
        steps=(step,),
    )
    assert CompactionSegment.model_validate_json(segment.model_dump_json()) == segment
    with pytest.raises(ValidationError):
        CompactionSegment.model_validate({**segment.model_dump(), "segment_end_turn": 2})


def test_manifests_are_strict_and_json_serializable():
    now = datetime.now(UTC)
    experiment = ExperimentManifest(
        experiment_id="exp-1",
        git_commit="deadbeef",
        config_files_used=("config.toml",),
        model_name="model",
        base_checkpoint="checkpoint",
        environment_dataset_version="tasks-v1",
        rollout_dataset_version="rollouts-v1",
        compaction_dataset_version="compact-v1",
        training_code_version="code-v1",
        exact_command_line=("rl", "@", "config.toml"),
        wall_clock_start=now,
        output_paths=("outputs/exp-1",),
    )
    artifact = ArtifactManifest(
        artifact_id="tasks-v1",
        artifact_type="tasks",
        artifact_version="1",
        created_at=now,
        git_commit="deadbeef",
        config_snapshot={"seed": 1},
        records=1,
        summary_stats_path="summary.json",
        checksums=(ChecksumEntry(path="tasks.jsonl", sha256="a" * 64, bytes=10),),
    )
    assert ExperimentManifest.model_validate_json(experiment.model_dump_json()) == experiment
    assert ArtifactManifest.model_validate_json(artifact.model_dump_json()) == artifact
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate({**artifact.model_dump(), "unknown": True})
