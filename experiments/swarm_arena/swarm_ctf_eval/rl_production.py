from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .async_admission import AsyncAdmissionLimits

ScenarioKind = Literal["ordinary", "critical", "decoy"]
HandoffFocusRole = Literal["sender", "receiver"]
OpponentFamily = Literal["base", "sft", "historical", "current"]

RL_PRODUCTION_PLAN_VERSION = "arena-rl-v4-production-plan-v1"
STAGED_RL_PRODUCTION_PLAN_VERSION = "arena-rl-v4-staged-production-plan-v1"


@dataclass(frozen=True)
class AdaptiveCurriculumConfig:
    version: str = "arena-rl-adaptive-curriculum-v1"
    stage_updates: int = 10
    positive_epsilon: float = 1e-12
    minimum_replicas: int = 4
    mastered_pass_rate: float = 1.0
    stalled_pass_rate: float = 0.0
    mastered_anchor_fraction: float = 0.1
    stalled_anchor_fraction: float = 0.1
    selection_seed: int = 20_261_101
    candidate_cases: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.version != "arena-rl-adaptive-curriculum-v1":
            raise ValueError("unsupported adaptive curriculum version")
        if self.stage_updates < 1 or self.minimum_replicas < 2:
            raise ValueError("adaptive curriculum requires positive stages and replica coverage")
        if self.positive_epsilon < 0 or self.selection_seed < 0:
            raise ValueError("adaptive curriculum epsilon and seed cannot be negative")
        if not 0 <= self.stalled_pass_rate < self.mastered_pass_rate <= 1:
            raise ValueError("adaptive curriculum pass-rate thresholds are invalid")
        if min(self.mastered_anchor_fraction, self.stalled_anchor_fraction) < 0:
            raise ValueError("adaptive curriculum anchor fractions cannot be negative")
        if self.mastered_anchor_fraction + self.stalled_anchor_fraction >= 1:
            raise ValueError("adaptive curriculum must reserve most slots for the frontier")
        if len(self.candidate_cases) != len(set(self.candidate_cases)):
            raise ValueError("adaptive curriculum candidate cases must be unique")
        for value in self.candidate_cases:
            parts = value.split(":")
            if (
                len(parts) != 3
                or not parts[0].isdigit()
                or parts[1] not in {"left_exposed", "right_exposed"}
                or parts[2] not in {"blue-0", "blue-1", "blue-2", "blue-3"}
            ):
                raise ValueError(f"invalid adaptive curriculum candidate case: {value}")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def logical_update_has_signal(step_groups: list[dict[str, Any]]) -> bool:
    """Return whether an approved shared-return logical update can change weights."""
    return any(
        abs(float(value)) > 1e-12
        for group in step_groups
        for replica in group.get("replicas", [])
        for value in replica.get("advantages", {}).values()
    )


@dataclass(frozen=True)
class CurriculumMix:
    ordinary: int
    critical: int
    decoy: int

    def validate(self) -> None:
        if self.ordinary < 0 or self.critical < 1 or self.decoy < 0:
            raise ValueError(
                "curriculum mix requires non-negative ordinary/decoy and positive critical groups"
            )
        if self.decoy > self.critical:
            raise ValueError("training decoys must be a matched subset of critical groups")

    @property
    def reduced_counts(self) -> tuple[int, int, int]:
        self.validate()
        divisor = math.gcd(self.ordinary, math.gcd(self.critical, self.decoy))
        return (
            self.ordinary // divisor,
            self.critical // divisor,
            self.decoy // divisor,
        )

    @property
    def block_size(self) -> int:
        return sum(self.reduced_counts)


@dataclass(frozen=True)
class ScenarioAssignment:
    ordinal: int
    kind: ScenarioKind
    pair_index: int | None
    ordinary_seed: int | None
    stage: str | None = None
    ordinary_size: int | None = None
    ordinary_horizon: int | None = None
    handoff_focus_role: HandoffFocusRole | None = None
    handoff_world: str | None = None
    handoff_remaining_turns: int | None = None
    handoff_trainable_turn_offsets: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    updates: int
    update_pattern: tuple[CurriculumMix, ...]
    ordinary_sizes: tuple[int, ...]
    ordinary_horizons: tuple[int, ...]
    handoff_focus_roles: tuple[HandoffFocusRole, ...] = ("receiver",)
    handoff_cases: tuple[tuple[int, str], ...] = ()
    handoff_remaining_turns: int | None = None
    handoff_trainable_turn_offsets: tuple[int, ...] | None = None

    def validate(self, *, groups_per_update: int) -> None:
        if not self.name or self.updates < 1 or not self.update_pattern:
            raise ValueError("curriculum stages require a name, updates, and an update pattern")
        for mix in self.update_pattern:
            mix.validate()
            if mix.ordinary + mix.critical + mix.decoy != groups_per_update:
                raise ValueError(f"stage {self.name} update mix must contain exactly {groups_per_update} groups")
        if not self.ordinary_sizes or min(self.ordinary_sizes) < 4:
            raise ValueError(f"stage {self.name} requires valid ordinary graph sizes")
        if not self.ordinary_horizons or min(self.ordinary_horizons) < 2:
            raise ValueError(f"stage {self.name} requires valid ordinary horizons")
        if not self.handoff_focus_roles or any(
            role not in {"sender", "receiver"} for role in self.handoff_focus_roles
        ):
            raise ValueError(f"stage {self.name} requires sender/receiver handoff focus roles")
        if self.handoff_remaining_turns is not None and self.handoff_remaining_turns < 1:
            raise ValueError(f"stage {self.name} must retain at least one handoff turn")
        if self.handoff_trainable_turn_offsets is not None:
            offsets = self.handoff_trainable_turn_offsets
            if not offsets or len(set(offsets)) != len(offsets):
                raise ValueError(
                    f"stage {self.name} handoff trainable turn offsets must be non-empty and unique"
                )
            if any(offset < 0 for offset in offsets):
                raise ValueError(
                    f"stage {self.name} handoff trainable turn offsets cannot be negative"
                )
            if (
                self.handoff_remaining_turns is not None
                and max(offsets) >= self.handoff_remaining_turns
            ):
                raise ValueError(
                    f"stage {self.name} handoff trainable turn offsets must fit within remaining turns"
                )
        if any(
            pair_index < 0 or world not in {"left_exposed", "right_exposed"}
            for pair_index, world in self.handoff_cases
        ):
            raise ValueError(f"stage {self.name} contains an invalid selected handoff case")


def exact_staged_curriculum_schedule(
    stages: tuple[CurriculumStage, ...],
    *,
    groups_per_update: int,
    pair_offset: int,
    ordinary_seed_base: int,
    shuffle_seed: int,
) -> tuple[ScenarioAssignment, ...]:
    """Build a predeclared staged schedule with optional matched training decoys."""
    if not stages:
        raise ValueError("staged curriculum requires at least one stage")
    if min(groups_per_update, pair_offset, ordinary_seed_base, shuffle_seed) < 0:
        raise ValueError("staged curriculum counts, offsets, and seeds cannot be negative")
    if groups_per_update < 1:
        raise ValueError("groups per update must be positive")

    assignments: list[ScenarioAssignment] = []
    pair_cursor = pair_offset
    ordinary_cursor = 0
    update_cursor = 0
    handoff_focus_cursor = 0
    handoff_case_cursor = 0
    for stage in stages:
        stage.validate(groups_per_update=groups_per_update)
        for stage_update in range(stage.updates):
            mix = stage.update_pattern[stage_update % len(stage.update_pattern)]
            if stage.handoff_cases:
                selected_cases = tuple(
                    stage.handoff_cases[(handoff_case_cursor + index) % len(stage.handoff_cases)]
                    for index in range(mix.critical)
                )
                pair_indices = tuple(pair_index for pair_index, _ in selected_cases)
                pair_worlds = tuple(world for _, world in selected_cases)
                handoff_case_cursor += mix.critical
            else:
                pair_indices = tuple(pair_cursor + index for index in range(mix.critical))
                pair_worlds = (None,) * mix.critical
                pair_cursor += mix.critical
            pair_roles = tuple(
                stage.handoff_focus_roles[
                    (handoff_focus_cursor + index) % len(stage.handoff_focus_roles)
                ]
                for index in range(mix.critical)
            )
            block: list[
                tuple[
                    ScenarioKind,
                    int | None,
                    int | None,
                    HandoffFocusRole | None,
                    str | None,
                ]
            ] = []
            block.extend(
                ("ordinary", None, ordinary_seed_base + ordinary_cursor + index, None, None)
                for index in range(mix.ordinary)
            )
            ordinary_cursor += mix.ordinary
            block.extend(
                ("critical", pair_index, None, focus_role, world)
                for pair_index, focus_role, world in zip(
                    pair_indices, pair_roles, pair_worlds, strict=True
                )
            )
            block.extend(
                ("decoy", pair_index, None, focus_role, world)
                for pair_index, focus_role, world in zip(
                    pair_indices[: mix.decoy],
                    pair_roles[: mix.decoy],
                    pair_worlds[: mix.decoy],
                    strict=True,
                )
            )
            handoff_focus_cursor += mix.critical
            random.Random(f"{shuffle_seed}:{update_cursor}").shuffle(block)
            for within_update, (
                kind,
                pair_index,
                ordinary_seed,
                focus_role,
                handoff_world,
            ) in enumerate(block):
                assignments.append(
                    ScenarioAssignment(
                        ordinal=len(assignments),
                        kind=kind,
                        pair_index=pair_index,
                        ordinary_seed=ordinary_seed,
                        stage=stage.name,
                        ordinary_size=(
                            stage.ordinary_sizes[(stage_update + within_update) % len(stage.ordinary_sizes)]
                            if kind == "ordinary"
                            else None
                        ),
                        ordinary_horizon=(
                            stage.ordinary_horizons[(stage_update + within_update) % len(stage.ordinary_horizons)]
                            if kind == "ordinary"
                            else None
                        ),
                        handoff_focus_role=focus_role,
                        handoff_world=handoff_world,
                        handoff_remaining_turns=(
                            stage.handoff_remaining_turns if kind != "ordinary" else None
                        ),
                        handoff_trainable_turn_offsets=(
                            stage.handoff_trainable_turn_offsets if kind != "ordinary" else None
                        ),
                    )
                )
            update_cursor += 1
    return tuple(assignments)


def exact_curriculum_schedule(
    mix: CurriculumMix,
    *,
    total_groups: int,
    pair_offset: int,
    ordinary_seed_base: int,
    shuffle_seed: int,
) -> tuple[ScenarioAssignment, ...]:
    """Build an exact, paired curriculum schedule before any rollout is sampled."""
    ordinary, critical, decoy = mix.reduced_counts
    block_size = ordinary + critical + decoy
    if total_groups < 1 or total_groups % block_size:
        raise ValueError(f"total groups must be a positive multiple of curriculum block size {block_size}")
    if min(pair_offset, ordinary_seed_base, shuffle_seed) < 0:
        raise ValueError("curriculum offsets and seeds cannot be negative")

    assignments: list[ScenarioAssignment] = []
    pair_cursor = pair_offset
    ordinary_cursor = 0
    for block_index in range(total_groups // block_size):
        block: list[tuple[ScenarioKind, int | None, int | None]] = []
        block.extend(("ordinary", None, ordinary_seed_base + ordinary_cursor + index) for index in range(ordinary))
        ordinary_cursor += ordinary
        pair_indices = tuple(pair_cursor + index for index in range(critical))
        block.extend(("critical", pair_index, None) for pair_index in pair_indices)
        block.extend(("decoy", pair_index, None) for pair_index in pair_indices[:decoy])
        pair_cursor += critical
        random.Random(f"{shuffle_seed}:{block_index}").shuffle(block)
        for kind, pair_index, ordinary_seed in block:
            assignments.append(
                ScenarioAssignment(
                    ordinal=len(assignments),
                    kind=kind,
                    pair_index=pair_index,
                    ordinary_seed=ordinary_seed,
                )
            )
    return tuple(assignments)


def scenario_sampling_namespace(
    assignment: ScenarioAssignment | None,
    *,
    run_id: str,
    step: int,
    fallback_pair_index: int | None = None,
) -> str | None:
    """Return a shared sampling namespace only for matched curriculum pairs.

    Production ordinary assignments deliberately ignore the legacy alternating
    fallback. They have no matched pair and must keep independent sampling.
    """
    if not run_id or step < 0:
        raise ValueError("sampling namespaces require a run ID and non-negative step")
    if assignment is not None:
        if assignment.kind == "ordinary":
            return None
        pair_index = assignment.pair_index
    else:
        pair_index = fallback_pair_index
    if pair_index is None:
        return None
    if pair_index < 0:
        raise ValueError("sampling namespace pair index cannot be negative")
    return f"{run_id}:step-{step}:pair-{pair_index}"


@dataclass(frozen=True)
class OpponentSnapshot:
    opponent_id: str
    family: OpponentFamily
    model_name: str
    revision: str
    adapter_sha256: str | None
    update_index: int

    def validate(self) -> None:
        if not self.opponent_id or not self.model_name or not self.revision:
            raise ValueError("opponent snapshots require immutable identities")
        if self.adapter_sha256 is not None and not _is_sha256(self.adapter_sha256):
            raise ValueError(f"invalid adapter SHA-256 for opponent {self.opponent_id}")
        if self.family != "base" and self.adapter_sha256 is None:
            raise ValueError(f"non-base opponent {self.opponent_id} requires adapter bytes")
        if self.family == "base" and self.adapter_sha256 is not None:
            raise ValueError("the base opponent cannot bind LoRA adapter bytes")
        if self.update_index < 0:
            raise ValueError(f"negative update index for opponent {self.opponent_id}")


@dataclass(frozen=True)
class OpponentPool:
    snapshots: tuple[OpponentSnapshot, ...]
    rotation_seed: int

    def validate(self) -> None:
        if self.rotation_seed < 0 or not self.snapshots:
            raise ValueError("opponent pool requires snapshots and a non-negative seed")
        for snapshot in self.snapshots:
            snapshot.validate()
        ids = [snapshot.opponent_id for snapshot in self.snapshots]
        if len(ids) != len(set(ids)):
            raise ValueError("opponent IDs must be unique")
        model_names = [snapshot.model_name for snapshot in self.snapshots]
        if len(model_names) != len(set(model_names)):
            raise ValueError("opponent serving names must be unique")
        families = {snapshot.family for snapshot in self.snapshots}
        required = {"base", "sft", "historical", "current"}
        if families != required:
            raise ValueError("opponent pool must contain base, SFT, historical, and current families")

    @property
    def sha256(self) -> str:
        self.validate()
        return _canonical_sha256(asdict(self))

    def schedule(self, total_groups: int) -> tuple[OpponentSnapshot, ...]:
        self.validate()
        if total_groups < 1 or total_groups % len(self.snapshots):
            raise ValueError("total groups must be a positive multiple of the opponent-pool size")
        result = []
        for cycle in range(total_groups // len(self.snapshots)):
            block = list(self.snapshots)
            random.Random(f"{self.rotation_seed}:{cycle}").shuffle(block)
            result.extend(block)
        return tuple(result)


@dataclass(frozen=True)
class AsyncBackendIdentity:
    name: str
    version: str
    kernel_config_sha256: str
    calibration_sha256: str

    def validate(self) -> None:
        if not self.name or not self.version:
            raise ValueError("async backend identity cannot be empty")
        if not _is_sha256(self.kernel_config_sha256) or not _is_sha256(self.calibration_sha256):
            raise ValueError("async backend identity requires kernel and calibration hashes")


@dataclass(frozen=True)
class ProductionPlan:
    version: str
    mix: CurriculumMix
    trainable_phases: tuple[Literal["BROADCAST", "ACT"], ...]
    trainable_turn_offsets: tuple[int, ...] | None
    opponent_pool: OpponentPool
    backend: AsyncBackendIdentity
    admission_limits: AsyncAdmissionLimits
    groups_per_update: int
    rollout_queue_capacity: int
    curriculum_shuffle_seed: int
    pair_offset: int
    ordinary_seed_base: int
    ordinary_sizes: tuple[int, ...]
    ordinary_horizons: tuple[int, ...]
    stages: tuple[CurriculumStage, ...] = ()
    shared_return_replicas: int = 4
    action_prompt_profile: Literal["full", "focused_handoff_compact"] = "full"
    shared_return_baseline: Literal[
        "leave_one_out_mean",
        "paired_message_drop",
        "paired_target_swap",
        "paired_receiver_target_swap",
        "paired_receiver_target_swap_challenge",
    ] = "leave_one_out_mean"
    decoy_shared_return_baseline: Literal[
        "paired_receiver_target_swap_challenge",
    ] | None = None
    paired_contrast_centering: Literal["replica_mean", "none"] = "replica_mean"
    monitor_logical_update_signal: bool = False
    adaptive_curriculum: AdaptiveCurriculumConfig | None = None

    def validate(self) -> None:
        if self.version not in {
            RL_PRODUCTION_PLAN_VERSION,
            STAGED_RL_PRODUCTION_PLAN_VERSION,
        }:
            raise ValueError(f"unsupported production plan: {self.version}")
        self.mix.validate()
        if not self.trainable_phases or len(set(self.trainable_phases)) != len(self.trainable_phases):
            raise ValueError("trainable phases must be non-empty and unique")
        if any(phase not in {"BROADCAST", "ACT"} for phase in self.trainable_phases):
            raise ValueError("production plan contains an unknown trainable phase")
        if self.trainable_turn_offsets is not None:
            if not self.trainable_turn_offsets or len(set(self.trainable_turn_offsets)) != len(
                self.trainable_turn_offsets
            ):
                raise ValueError("trainable turn offsets must be non-empty and unique")
            if any(offset < 0 for offset in self.trainable_turn_offsets):
                raise ValueError("trainable turn offsets cannot be negative")
        self.opponent_pool.validate()
        self.backend.validate()
        self.admission_limits.validate()
        if self.groups_per_update < 1 or self.rollout_queue_capacity < self.groups_per_update:
            raise ValueError("rollout queue must hold at least one complete logical update")
        if self.version == RL_PRODUCTION_PLAN_VERSION:
            if self.stages:
                raise ValueError("legacy production plans cannot contain staged curricula")
            if self.groups_per_update % self.mix.block_size:
                raise ValueError("each logical update must contain an exact curriculum block")
        else:
            if not self.stages:
                raise ValueError("staged production plans require curriculum stages")
            for stage in self.stages:
                stage.validate(groups_per_update=self.groups_per_update)
        if self.groups_per_update % len(self.opponent_pool.snapshots):
            raise ValueError("each logical update must contain one exact opponent-pool rotation")
        if (
            min(
                self.curriculum_shuffle_seed,
                self.pair_offset,
                self.ordinary_seed_base,
            )
            < 0
        ):
            raise ValueError("production curriculum seeds and offsets cannot be negative")
        if not self.ordinary_sizes or min(self.ordinary_sizes) < 4:
            raise ValueError("production plan requires valid ordinary graph sizes")
        if not self.ordinary_horizons or min(self.ordinary_horizons) < 2:
            raise ValueError("production plan requires valid ordinary horizons")
        if self.shared_return_replicas < 2 or self.shared_return_replicas > 32:
            raise ValueError("production plan requires 2--32 shared-return replicas")
        if self.action_prompt_profile not in {"full", "focused_handoff_compact"}:
            raise ValueError("production plan contains an unknown action prompt profile")
        if self.action_prompt_profile == "focused_handoff_compact" and "ACT" not in self.trainable_phases:
            raise ValueError("compact handoff prompts require ACT to be a trainable phase")
        if self.shared_return_baseline not in {
            "leave_one_out_mean",
            "paired_message_drop",
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        }:
            raise ValueError("production plan contains an unknown shared-return baseline")
        if self.shared_return_baseline in {
            "paired_message_drop",
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        } and (
            self.trainable_phases != ("ACT",)
        ):
            raise ValueError("paired message-intervention plans require receiver ACT-only training")
        if self.decoy_shared_return_baseline not in {
            None,
            "paired_receiver_target_swap_challenge",
        }:
            raise ValueError("production plan contains an unknown decoy baseline")
        if self.decoy_shared_return_baseline is not None and not any(
            mix.decoy for stage in self.stages for mix in stage.update_pattern
        ):
            raise ValueError("a decoy challenge baseline requires scheduled decoy groups")
        if self.paired_contrast_centering not in {"replica_mean", "none"}:
            raise ValueError("production plan contains an unknown paired-contrast centering")
        if self.paired_contrast_centering == "none" and self.shared_return_baseline not in {
            "paired_message_drop",
            "paired_target_swap",
            "paired_receiver_target_swap",
            "paired_receiver_target_swap_challenge",
        }:
            raise ValueError("uncentered paired contrast requires a paired intervention baseline")
        if not isinstance(self.monitor_logical_update_signal, bool):
            raise ValueError("logical-update signal monitor must be boolean")
        if self.adaptive_curriculum is not None:
            self.adaptive_curriculum.validate()
            if self.version != STAGED_RL_PRODUCTION_PLAN_VERSION:
                raise ValueError("adaptive curricula require a staged production plan")
            if any(
                stage.updates != self.adaptive_curriculum.stage_updates
                for stage in self.stages
            ):
                raise ValueError("adaptive curriculum boundaries must match every staged block")
            if not self.adaptive_curriculum.candidate_cases:
                raise ValueError("adaptive curriculum requires a hash-bound candidate pool")

    @property
    def sha256(self) -> str:
        self.validate()
        payload = asdict(self)
        # This stage field was added after the completed V4--V13 runs.  Omitting
        # its legacy default preserves every existing immutable plan identity;
        # an explicit multi-turn schedule remains hash-bound for V14 onward.
        for stage in payload.get("stages", []):
            if stage.get("handoff_trainable_turn_offsets") is None:
                stage.pop("handoff_trainable_turn_offsets")
        if self.shared_return_replicas == 4:
            payload.pop("shared_return_replicas")
        if self.action_prompt_profile == "full":
            payload.pop("action_prompt_profile")
        if self.shared_return_baseline == "leave_one_out_mean":
            payload.pop("shared_return_baseline")
        if self.decoy_shared_return_baseline is None:
            payload.pop("decoy_shared_return_baseline")
        if self.paired_contrast_centering == "replica_mean":
            payload.pop("paired_contrast_centering")
        if not self.monitor_logical_update_signal:
            payload.pop("monitor_logical_update_signal")
        if self.adaptive_curriculum is None:
            payload.pop("adaptive_curriculum")
        if self.version == RL_PRODUCTION_PLAN_VERSION:
            # Preserve byte-for-byte run-lock identity for every completed v4 run.
            payload.pop("stages")
        return _canonical_sha256(payload)

    @property
    def expected_updates(self) -> int | None:
        if not self.stages:
            return None
        return sum(stage.updates for stage in self.stages)

    def curriculum_schedule(self, *, steps: int) -> tuple[ScenarioAssignment, ...]:
        self.validate()
        if self.stages:
            expected = self.expected_updates
            if steps != expected:
                raise ValueError(f"staged plan declares {expected} updates but controller requested {steps}")
            return exact_staged_curriculum_schedule(
                self.stages,
                groups_per_update=self.groups_per_update,
                pair_offset=self.pair_offset,
                ordinary_seed_base=self.ordinary_seed_base,
                shuffle_seed=self.curriculum_shuffle_seed,
            )
        return exact_curriculum_schedule(
            self.mix,
            total_groups=steps * self.groups_per_update,
            pair_offset=self.pair_offset,
            ordinary_seed_base=self.ordinary_seed_base,
            shuffle_seed=self.curriculum_shuffle_seed,
        )


def load_production_plan(path: Path) -> tuple[ProductionPlan, dict[str, Path | None]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pool_rows = raw["opponent_pool"]["snapshots"]
    snapshots = tuple(
        OpponentSnapshot(
            opponent_id=str(row["opponent_id"]),
            family=str(row["family"]),
            model_name=str(row["model_name"]),
            revision=str(row["revision"]),
            adapter_sha256=(None if row.get("adapter_sha256") is None else str(row["adapter_sha256"])),
            update_index=int(row["update_index"]),
        )
        for row in pool_rows
    )
    runtime_paths = {
        str(row["opponent_id"]): (
            None if row.get("adapter_path") is None else (path.parent / str(row["adapter_path"])).resolve()
        )
        for row in pool_rows
    }
    limit_values = dict(raw["async_admission"]["limits"])
    # Backward-compatible loading for immutable plans produced before the
    # robust mean mismatch-KL diagnostic became configurable.
    limit_values.setdefault("max_mean_mismatch_kl", None)
    limits = AsyncAdmissionLimits(**limit_values)
    backend = AsyncBackendIdentity(**raw["async_admission"]["backend"])
    turns = raw["trainable_spans"]["turn_offsets"]
    stage_rows = raw.get("curriculum_stages", [])
    stages = tuple(
        CurriculumStage(
            name=str(row["name"]),
            updates=int(row["updates"]),
            update_pattern=tuple(CurriculumMix(**mix) for mix in row["update_pattern"]),
            ordinary_sizes=tuple(int(value) for value in row["ordinary_sizes"]),
            ordinary_horizons=tuple(int(value) for value in row["ordinary_horizons"]),
            handoff_focus_roles=tuple(
                str(value) for value in row.get("handoff_focus_roles", ["receiver"])
            ),
            handoff_cases=tuple(
                (int(value["pair_index"]), str(value["world"]))
                for value in row.get("handoff_cases", [])
            ),
            handoff_remaining_turns=(
                None
                if row.get("handoff_remaining_turns") is None
                else int(row["handoff_remaining_turns"])
            ),
            handoff_trainable_turn_offsets=(
                None
                if row.get("handoff_trainable_turn_offsets") is None
                else tuple(int(value) for value in row["handoff_trainable_turn_offsets"])
            ),
        )
        for row in stage_rows
    )
    plan = ProductionPlan(
        version=str(raw["version"]),
        mix=CurriculumMix(**raw["curriculum_mix"]),
        trainable_phases=tuple(raw["trainable_spans"]["phases"]),
        trainable_turn_offsets=(None if turns == "all" else tuple(int(value) for value in turns)),
        opponent_pool=OpponentPool(
            snapshots=snapshots,
            rotation_seed=int(raw["opponent_pool"]["rotation_seed"]),
        ),
        backend=backend,
        admission_limits=limits,
        groups_per_update=int(raw["groups_per_update"]),
        rollout_queue_capacity=int(raw["rollout_queue_capacity"]),
        curriculum_shuffle_seed=int(raw["schedule"]["shuffle_seed"]),
        pair_offset=int(raw["schedule"]["pair_offset"]),
        ordinary_seed_base=int(raw["schedule"]["ordinary_seed_base"]),
        ordinary_sizes=tuple(int(value) for value in raw["schedule"]["ordinary_sizes"]),
        ordinary_horizons=tuple(int(value) for value in raw["schedule"]["ordinary_horizons"]),
        stages=stages,
        shared_return_replicas=int(raw.get("rollout_runtime", {}).get("shared_return_replicas", 4)),
        action_prompt_profile=str(raw.get("rollout_runtime", {}).get("action_prompt_profile", "full")),
        shared_return_baseline=str(
            raw.get("rollout_runtime", {}).get("shared_return_baseline", "leave_one_out_mean")
        ),
        decoy_shared_return_baseline=raw.get("rollout_runtime", {}).get(
            "decoy_shared_return_baseline"
        ),
        paired_contrast_centering=str(
            raw.get("rollout_runtime", {}).get("paired_contrast_centering", "replica_mean")
        ),
        monitor_logical_update_signal=bool(
            raw.get("rollout_runtime", {}).get("monitor_logical_update_signal", False)
        ),
        adaptive_curriculum=(
            None
            if raw.get("adaptive_curriculum") is None
            else AdaptiveCurriculumConfig(
                **{
                    **raw["adaptive_curriculum"],
                    "candidate_cases": tuple(
                        str(value)
                        for value in raw["adaptive_curriculum"].get("candidate_cases", [])
                    ),
                }
            )
        ),
    )
    plan.validate()
    if set(runtime_paths) != {snapshot.opponent_id for snapshot in snapshots}:
        raise ValueError("opponent runtime paths do not cover the immutable pool")
    for snapshot in snapshots:
        path_value = runtime_paths[snapshot.opponent_id]
        if snapshot.family == "base" and path_value is not None:
            raise ValueError("base opponent cannot have an adapter path")
        if snapshot.family in {"sft", "historical"} and path_value is None:
            raise ValueError(f"opponent {snapshot.opponent_id} requires an adapter path")
    return plan, runtime_paths
