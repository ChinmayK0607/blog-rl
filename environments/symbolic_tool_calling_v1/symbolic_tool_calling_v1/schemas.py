from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from symbolic_tool_calling_v1.models import Action, BenchmarkState, FrozenModel

ROLLOUT_SCHEMA_VERSION = "1.0.0"
COMPACTION_SCHEMA_VERSION = "1.0.0"
TRAINING_EXAMPLE_SCHEMA_VERSION = "1.0.0"
EXPERIMENT_MANIFEST_VERSION = "1.0.0"
ARTIFACT_MANIFEST_VERSION = "1.0.0"


class SamplingConfig(FrozenModel):
    temperature: float
    top_p: float
    top_k: int | None = None
    max_turns: int
    max_tokens: int
    stop: tuple[str, ...] = ()


class RolloutStep(FrozenModel):
    turn_index: int = Field(ge=0)
    model_output: str
    tool_call: Action
    tool_response: str
    environment_state: BenchmarkState
    input_token_count: int = Field(ge=0)
    output_token_count: int = Field(ge=0)
    response_token_count: int = Field(ge=0)

    @property
    def token_count(self) -> int:
        return self.input_token_count + self.output_token_count + self.response_token_count


class RolloutRecord(FrozenModel):
    schema_version: Literal["1.0.0"] = ROLLOUT_SCHEMA_VERSION
    rollout_id: str
    task_id: str
    prompt_id: str
    policy_id: str
    policy_checkpoint: str
    sampling_config_hash: str
    environment_version: str
    rollout_seed: int
    full_prompt_text: str
    initial_environment_state: BenchmarkState
    steps: tuple[RolloutStep, ...]
    success: bool
    terminal_reward: float
    total_turns: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    stop_reason: Literal["submitted", "max_turns", "policy_stopped", "error"]
    error: str | None = None

    @model_validator(mode="after")
    def _validate_derived_fields(self) -> "RolloutRecord":
        if self.total_turns != len(self.steps):
            raise ValueError("total_turns must equal the number of steps")
        if self.total_tokens != sum(step.token_count for step in self.steps):
            raise ValueError("total_tokens must equal the sum of step token counts")
        if self.success != (self.terminal_reward == 1.0):
            raise ValueError("success and terminal_reward disagree")
        if tuple(step.turn_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("turn indexes must be contiguous and zero-based")
        return self


class CompactionSegment(FrozenModel):
    schema_version: Literal["1.0.0"] = COMPACTION_SCHEMA_VERSION
    rollout_id: str
    compaction_version: str
    compaction_method: Literal["fixed_token_budget"]
    segment_id: str
    segment_index: int = Field(ge=0)
    num_segments_total: int = Field(ge=1)
    token_count_segment: int = Field(ge=0)
    turn_count_segment: int = Field(ge=0)
    segment_position_bucket: Literal["early", "middle", "late"]
    carried_state_summary: BenchmarkState
    inherited_reward: float
    inherited_group_advantage: float | None = None
    segment_start_turn: int = Field(ge=0)
    segment_end_turn: int = Field(ge=0)
    terminal_containing: bool
    steps: tuple[RolloutStep, ...]

    @model_validator(mode="after")
    def _validate_segment(self) -> "CompactionSegment":
        if self.segment_index >= self.num_segments_total:
            raise ValueError("segment_index must be smaller than num_segments_total")
        if self.segment_end_turn < self.segment_start_turn:
            raise ValueError("segment turn range is reversed")
        if self.turn_count_segment != len(self.steps):
            raise ValueError("turn_count_segment must equal the number of steps")
        if self.segment_end_turn - self.segment_start_turn != self.turn_count_segment:
            raise ValueError("segment turn bounds must be half-open")
        if self.token_count_segment != sum(step.token_count for step in self.steps):
            raise ValueError("token_count_segment must equal the sum of step tokens")
        return self


class CompactionSummary(FrozenModel):
    rollout_id: str
    num_segments: int = Field(ge=1)
    segment_token_lengths: tuple[int, ...]
    segment_turn_lengths: tuple[int, ...]
    first_segment_id: str
    last_segment_id: str
    terminal_segment_id: str | None


class TrainingExample(FrozenModel):
    """Objective-specific credit attached to one immutable compacted segment."""

    schema_version: Literal["1.0.0"] = TRAINING_EXAMPLE_SCHEMA_VERSION
    example_id: str
    objective: Literal["compacted_grpo", "segment_normalized_grpo", "compacted_ppo"]
    rollout_id: str
    segment_id: str
    segment_index: int = Field(ge=0)
    num_segments_total: int = Field(ge=1)
    policy_advantage: float
    policy_weight: float = Field(gt=0)
    transition_reward: float
    done: bool
    critic_value: float | None = None
    critic_target: float | None = None

    @model_validator(mode="after")
    def _validate_objective_fields(self) -> "TrainingExample":
        has_critic = self.critic_value is not None or self.critic_target is not None
        if self.objective == "compacted_ppo" and not (
            self.critic_value is not None and self.critic_target is not None
        ):
            raise ValueError("compacted PPO examples require both critic value and target")
        if self.objective != "compacted_ppo" and has_critic:
            raise ValueError("critic fields are only valid for compacted PPO")
        if self.done != (self.segment_index == self.num_segments_total - 1):
            raise ValueError("done must identify the final compacted segment")
        return self


class ExperimentManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = EXPERIMENT_MANIFEST_VERSION
    experiment_id: str
    git_commit: str
    config_files_used: tuple[str, ...]
    model_name: str
    base_checkpoint: str
    tokenizer_hash: str | None = None
    environment_dataset_version: str
    rollout_dataset_version: str
    compaction_dataset_version: str
    training_code_version: str
    exact_command_line: tuple[str, ...]
    wall_clock_start: datetime
    wall_clock_end: datetime | None = None
    output_paths: tuple[str, ...]


class ChecksumEntry(FrozenModel):
    path: str
    sha256: str
    bytes: int = Field(ge=0)


class ArtifactManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = ARTIFACT_MANIFEST_VERSION
    artifact_id: str
    artifact_type: Literal["tasks", "rollouts", "compaction", "training_examples"]
    artifact_version: str
    created_at: datetime
    git_commit: str
    config_snapshot: dict[str, Any]
    records: int = Field(ge=0)
    summary_stats_path: str
    checksums: tuple[ChecksumEntry, ...]
