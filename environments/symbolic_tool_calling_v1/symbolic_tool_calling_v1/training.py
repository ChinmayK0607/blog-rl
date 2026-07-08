import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from symbolic_tool_calling_v1.artifacts import read_jsonl, stable_hash, verify_artifact, write_artifact
from symbolic_tool_calling_v1.models import FrozenModel
from symbolic_tool_calling_v1.schemas import CompactionSegment, TrainingExample

TRAINING_TRANSFORM_VERSION = "1.0.0"
Objective = Literal["compacted_grpo", "segment_normalized_grpo", "compacted_ppo"]


class TrainingTransformConfig(FrozenModel):
    transform_version: Literal["1.0.0"] = TRAINING_TRANSFORM_VERSION
    objectives: tuple[Objective, ...] = (
        "compacted_grpo",
        "segment_normalized_grpo",
        "compacted_ppo",
    )
    gamma: float = Field(default=1.0, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    critic_source: Literal["zero_baseline", "external"] = "zero_baseline"
    critic_values: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_critic_config(self) -> "TrainingTransformConfig":
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("objectives must be unique")
        if self.critic_source == "zero_baseline" and self.critic_values:
            raise ValueError("zero_baseline cannot carry external critic values")
        return self


def _group_segments(segments: list[CompactionSegment]) -> dict[str, list[CompactionSegment]]:
    grouped: dict[str, list[CompactionSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.rollout_id].append(segment)
    for rollout_id, group in grouped.items():
        group.sort(key=lambda segment: segment.segment_index)
        expected = list(range(group[0].num_segments_total))
        actual = [segment.segment_index for segment in group]
        if actual != expected or any(segment.num_segments_total != len(group) for segment in group):
            raise ValueError(f"rollout {rollout_id} has incomplete or inconsistent segments")
    return grouped


def _critic_values(group: list[CompactionSegment], config: TrainingTransformConfig) -> list[float]:
    if config.critic_source == "zero_baseline":
        return [0.0] * len(group)
    missing = [segment.segment_id for segment in group if segment.segment_id not in config.critic_values]
    if missing:
        raise ValueError(f"missing external critic values for segment ids: {missing}")
    return [config.critic_values[segment.segment_id] for segment in group]


def _gae(
    group: list[CompactionSegment], values: list[float], gamma: float, gae_lambda: float
) -> tuple[list[float], list[float]]:
    """Segment-level GAE with reward only on the final environment transition."""
    advantages = [0.0] * len(group)
    next_advantage = 0.0
    for index in range(len(group) - 1, -1, -1):
        done = index == len(group) - 1
        reward = group[index].inherited_reward if done else 0.0
        next_value = 0.0 if done else values[index + 1]
        delta = reward + gamma * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * (0.0 if done else next_advantage)
        advantages[index] = next_advantage
    targets = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, targets


def build_training_examples(
    segments: list[CompactionSegment], config: TrainingTransformConfig
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    grouped = _group_segments(segments)
    for rollout_id in sorted(grouped):
        group = grouped[rollout_id]
        values = _critic_values(group, config)
        ppo_advantages, critic_targets = _gae(group, values, config.gamma, config.gae_lambda)
        for objective in config.objectives:
            for index, segment in enumerate(group):
                done = index == len(group) - 1
                if objective == "compacted_ppo":
                    advantage = ppo_advantages[index]
                    weight = 1.0
                    critic_value = values[index]
                    critic_target = critic_targets[index]
                else:
                    if segment.inherited_group_advantage is None:
                        raise ValueError(f"segment {segment.segment_id} has no inherited group advantage")
                    advantage = segment.inherited_group_advantage
                    weight = 1.0 / len(group) if objective == "segment_normalized_grpo" else 1.0
                    critic_value = None
                    critic_target = None
                identity = {"segment_id": segment.segment_id, "objective": objective, "version": config.transform_version}
                examples.append(
                    TrainingExample(
                        example_id=stable_hash(identity, prefix="train-example-")[:31],
                        objective=objective,
                        rollout_id=segment.rollout_id,
                        segment_id=segment.segment_id,
                        segment_index=segment.segment_index,
                        num_segments_total=segment.num_segments_total,
                        policy_advantage=advantage,
                        policy_weight=weight,
                        transition_reward=segment.inherited_reward if done else 0.0,
                        done=done,
                        critic_value=critic_value,
                        critic_target=critic_target,
                    )
                )
    return examples


def write_training_dataset(
    output_dir: Path,
    segments: list[CompactionSegment],
    config: TrainingTransformConfig,
    repo: Path,
):
    examples = build_training_examples(segments, config)
    by_objective = {objective: sum(example.objective == objective for example in examples) for objective in config.objectives}
    return write_artifact(
        output_dir,
        artifact_id=f"training-examples-{stable_hash(config.model_dump(mode='json'))[:12]}",
        artifact_type="training_examples",
        artifact_version=config.transform_version,
        config=config,
        records=examples,
        records_filename="examples.jsonl",
        summary={"num_examples": len(examples), "by_objective": by_objective},
        repo=repo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach GRPO/PPO credit to a frozen compaction artifact.")
    parser.add_argument("compaction_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = verify_artifact(args.compaction_dir)
    if manifest.artifact_type != "compaction":
        raise ValueError(f"expected a compaction artifact, got {manifest.artifact_type}")
    config = TrainingTransformConfig()
    if args.config:
        config = TrainingTransformConfig.model_validate(json.loads(args.config.read_text()))
    segments = read_jsonl(args.compaction_dir / "segments.jsonl", CompactionSegment)
    result = write_training_dataset(args.output_dir, segments, config, args.repo.resolve())
    print(json.dumps({"artifact_id": result.artifact_id, "records": result.records}, indent=2))
