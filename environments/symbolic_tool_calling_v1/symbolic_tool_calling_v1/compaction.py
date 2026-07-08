import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from symbolic_tool_calling_v1.artifacts import stable_hash, write_artifact
from symbolic_tool_calling_v1.models import FrozenModel
from symbolic_tool_calling_v1.schemas import CompactionSegment, CompactionSummary, RolloutRecord

COMPACTION_VERSION = "1.0.0"


class CompactionConfig(FrozenModel):
    compaction_version: Literal["1.0.0"] = COMPACTION_VERSION
    token_budget: int = Field(default=512, ge=1)
    inherit_group_advantage: bool = True


def group_relative_advantages(rollouts: list[RolloutRecord]) -> dict[str, float]:
    by_prompt: dict[str, list[RolloutRecord]] = {}
    for rollout in rollouts:
        by_prompt.setdefault(rollout.prompt_id, []).append(rollout)
    advantages: dict[str, float] = {}
    for group in by_prompt.values():
        rewards = [rollout.terminal_reward for rollout in group]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        scale = math.sqrt(variance)
        for rollout in group:
            advantages[rollout.rollout_id] = (rollout.terminal_reward - mean) / scale if scale else 0.0
    return advantages


def _position(index: int, total: int) -> str:
    if total == 1:
        return "late"
    fraction = index / (total - 1)
    if fraction < 1 / 3:
        return "early"
    if fraction > 2 / 3:
        return "late"
    return "middle"


def compact_rollout(
    rollout: RolloutRecord,
    config: CompactionConfig,
    *,
    inherited_group_advantage: float | None = None,
) -> tuple[list[CompactionSegment], CompactionSummary]:
    chunks: list[list] = []
    current: list = []
    current_tokens = 0
    for step in rollout.steps:
        if current and current_tokens + step.token_count > config.token_budget:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(step)
        current_tokens += step.token_count
    if current or not chunks:
        chunks.append(current)

    segments: list[CompactionSegment] = []
    total = len(chunks)
    start_turn = 0
    for index, chunk in enumerate(chunks):
        end_turn = start_turn + len(chunk)
        carried_state = (
            rollout.initial_environment_state if start_turn == 0 else rollout.steps[start_turn - 1].environment_state
        )
        segment_id = stable_hash(
            {
                "rollout_id": rollout.rollout_id,
                "compaction_version": config.compaction_version,
                "token_budget": config.token_budget,
                "segment_index": index,
                "start_turn": start_turn,
                "end_turn": end_turn,
            },
            prefix="segment-",
        )[:25]
        segments.append(
            CompactionSegment(
                rollout_id=rollout.rollout_id,
                compaction_version=config.compaction_version,
                compaction_method="fixed_token_budget",
                segment_id=segment_id,
                segment_index=index,
                num_segments_total=total,
                token_count_segment=sum(step.token_count for step in chunk),
                turn_count_segment=len(chunk),
                segment_position_bucket=_position(index, total),
                carried_state_summary=carried_state.model_copy(deep=True),
                inherited_reward=rollout.terminal_reward,
                inherited_group_advantage=inherited_group_advantage,
                segment_start_turn=start_turn,
                segment_end_turn=end_turn,
                terminal_containing=bool(chunk and chunk[-1].environment_state.terminal),
                steps=tuple(chunk),
            )
        )
        start_turn = end_turn

    terminal = next((segment.segment_id for segment in segments if segment.terminal_containing), None)
    summary = CompactionSummary(
        rollout_id=rollout.rollout_id,
        num_segments=len(segments),
        segment_token_lengths=tuple(segment.token_count_segment for segment in segments),
        segment_turn_lengths=tuple(segment.turn_count_segment for segment in segments),
        first_segment_id=segments[0].segment_id,
        last_segment_id=segments[-1].segment_id,
        terminal_segment_id=terminal,
    )
    return segments, summary


def compact_rollouts(
    rollouts: list[RolloutRecord], config: CompactionConfig
) -> tuple[list[CompactionSegment], list[CompactionSummary]]:
    advantages = group_relative_advantages(rollouts) if config.inherit_group_advantage else {}
    all_segments: list[CompactionSegment] = []
    summaries: list[CompactionSummary] = []
    for rollout in rollouts:
        segments, summary = compact_rollout(
            rollout,
            config,
            inherited_group_advantage=advantages.get(rollout.rollout_id),
        )
        all_segments.extend(segments)
        summaries.append(summary)
    return all_segments, summaries


def write_compaction_dataset(
    output_dir: Path,
    rollouts: list[RolloutRecord],
    config: CompactionConfig,
    repo: Path,
):
    segments, summaries = compact_rollouts(rollouts, config)
    summary_by_rollout = {summary.rollout_id: summary.model_dump(mode="json") for summary in summaries}
    segment_counts = [summary.num_segments for summary in summaries]
    dashboard = {
        "num_rollouts": len(rollouts),
        "num_segments": len(segments),
        "segment_count": {
            "min": min(segment_counts),
            "max": max(segment_counts),
            "mean": sum(segment_counts) / len(segment_counts),
        },
        "empty_segments": sum(not segment.steps for segment in segments),
        "rollouts": summary_by_rollout,
    }
    return write_artifact(
        output_dir,
        artifact_id=f"compaction-{stable_hash(config.model_dump(mode='json'))[:12]}",
        artifact_type="compaction",
        artifact_version=config.compaction_version,
        config=config,
        records=segments,
        records_filename="segments.jsonl",
        summary=dashboard,
        repo=repo,
    )
