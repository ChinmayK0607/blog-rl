from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GENERATOR_VERSION = "1.0.0"
VERIFIER_SPEC_VERSION = "1.0.0"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RoomSpec(FrozenModel):
    room_id: str
    exits: dict[str, str]
    objects: tuple[str, ...] = ()
    terminal_code: str | None = None
    verbosity_text: str = ""


class HiddenGraphSpec(FrozenModel):
    start_room: str
    target_room: str
    target_system: str
    access_code: str
    key_item: str
    switch_id: str
    rooms: dict[str, RoomSpec]


class BenchmarkTask(FrozenModel):
    task_id: str
    generator_version: str = GENERATOR_VERSION
    seed: int
    horizon_bucket: Literal["short", "medium", "long"]
    dependency_depth: int
    branching_factor: int
    distractor_ratio: float
    recovery_cost: int
    verbosity_setting: Literal["low", "high"]
    imbalance_setting: Literal["low", "high"]
    hidden_graph_spec: HiddenGraphSpec
    optimal_plan_length: int
    verifier_spec_version: str = VERIFIER_SPEC_VERSION


class Action(FrozenModel):
    tool: Literal["inspect", "move", "pickup", "use", "query", "submit"]
    arguments: dict[str, Any] = Field(default_factory=dict)


class BenchmarkState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_room: str
    inventory: list[str] = Field(default_factory=list)
    activated_switches: list[str] = Field(default_factory=list)
    unlocked_rooms: list[str] = Field(default_factory=list)
    discovered_code: str | None = None
    submitted: bool = False
    success: bool = False
    terminal: bool = False
    turns: int = 0
    invalid_actions: int = 0


class Transition(FrozenModel):
    action: Action
    observation: str
    state: BenchmarkState
