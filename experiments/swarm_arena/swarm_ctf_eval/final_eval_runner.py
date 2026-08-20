from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .arena import GameState, NodeObservation, Team
from .arena_eval import ArenaModel
from .arena_generation import generate_state
from .communication_curriculum import permute_agent_labels, swap_team_labels
from .crossplay_eval import evaluate_crossplay
from .episode import EMPTY_BROADCAST, EpisodeConfig
from .episode_protocol import ActionPromptProfile
from .final_eval_v3 import COMMUNICATION_CONDITIONS
from .rl_v3 import ArenaRLEnv


@dataclass(frozen=True)
class FinalEvalIdentity:
    case_id: str
    suite: str
    policy_variant: str
    policy_revision: str
    policy_assignment: str
    role_assignment: str
    option_order: str
    opponent_id: str
    opponent_revision: str
    sampling_key: str


def _assignment(value: str) -> tuple[int, int, int, int]:
    if value == "identity":
        return (0, 1, 2, 3)
    prefix = "perm-"
    if not value.startswith(prefix):
        raise ValueError(f"invalid four-agent assignment: {value}")
    digits = tuple(int(character) for character in value.removeprefix(prefix))
    if len(digits) != 4 or sorted(digits) != [0, 1, 2, 3]:
        raise ValueError(f"invalid four-agent assignment: {value}")
    return digits


def _roster(
    models: tuple[ArenaModel, ...],
    team: Team,
    *,
    policy_assignment: tuple[int, int, int, int] = (0, 1, 2, 3),
    role_assignment: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> dict[str, ArenaModel]:
    if len(models) != 4:
        raise ValueError("final evaluation requires exactly four models per team")
    prefix = team.lower()
    return {
        f"{prefix}-{role_assignment[physical_index]}": models[policy_assignment[physical_index]]
        for physical_index in range(4)
    }


def _option_offset(option_order: str) -> int:
    if option_order == "canonical":
        return 0
    prefix = "permuted-"
    if not option_order.startswith(prefix):
        raise ValueError(f"invalid option-order assignment: {option_order}")
    index = int(option_order.removeprefix(prefix))
    if index < 1:
        raise ValueError(f"invalid option-order assignment: {option_order}")
    return index * 1_000_003


def evaluate_final_case(
    focal_models: tuple[ArenaModel, ...],
    opponent_models: tuple[ArenaModel, ...],
    case: tuple[int, int, int],
    identity: FinalEvalIdentity,
    *,
    focal_side: Team,
    condition: str,
    initial_state: GameState | None = None,
    critical_target: str | None = None,
    turn_zero_required_facts: dict[str, NodeObservation] | None = None,
    action_prompt_profiles: dict[str, ActionPromptProfile] | None = None,
    target_swap_sender: str | None = None,
    target_swap_targets: tuple[str, str] | None = None,
    target_swap_active_target: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in {*COMMUNICATION_CONDITIONS, "target_swapped"}:
        raise ValueError(f"unknown final-eval condition: {condition}")
    swap_fields = (target_swap_sender, target_swap_targets, target_swap_active_target)
    if condition == "target_swapped" and not all(value is not None for value in swap_fields):
        raise ValueError("target-swapped evaluation requires complete intervention metadata")
    if condition != "target_swapped" and any(value is not None for value in swap_fields):
        raise ValueError("non-swap evaluation cannot bind a target-swap intervention")
    handoff_suites = {"critical", "decoy", "handoff_critical", "handoff_decoy"}
    if identity.suite in handoff_suites and initial_state is None:
        raise ValueError("critical and decoy evaluation require a frozen initial state")
    if identity.suite in {"critical", "handoff_critical"} and critical_target is None:
        raise ValueError("critical evaluation requires its certified target")

    seed, size, horizon = case
    env = ArenaRLEnv(
        seed,
        size,
        EpisodeConfig(
            horizon=horizon,
            communication_cost=0.0,
            invalid_broadcast_cost=0.0,
            invalid_action_cost=0.0,
        ),
    )
    policy_assignment = _assignment(identity.policy_assignment)
    role_assignment = _assignment(identity.role_assignment)
    focal = _roster(
        focal_models,
        focal_side,
        policy_assignment=policy_assignment,
        role_assignment=role_assignment,
    )
    rival_side: Team = "RED" if focal_side == "BLUE" else "BLUE"
    rival = _roster(opponent_models, rival_side)
    blue = focal if focal_side == "BLUE" else rival
    red = focal if focal_side == "RED" else rival
    crossplay_condition = "generated" if condition == "normal" else condition
    resolved_state = (
        swap_team_labels(initial_state)
        if initial_state is not None and focal_side == "RED"
        else initial_state
    )
    if role_assignment != (0, 1, 2, 3):
        if resolved_state is None:
            resolved_state = generate_state(seed, size)
        resolved_state = permute_agent_labels(resolved_state, focal_side, role_assignment)
    resolved_swap_sender = None
    if target_swap_sender is not None:
        sender_index = int(target_swap_sender.split("-", 1)[1])
        resolved_swap_sender = f"{focal_side.lower()}-{role_assignment[sender_index]}"
    raw = evaluate_crossplay(
        blue,
        red,
        case,
        blue_condition=crossplay_condition if focal_side == "BLUE" else "generated",
        red_condition=crossplay_condition if focal_side == "RED" else "generated",
        initial_state=resolved_state,
        env=env,
        action_permutation_offset=_option_offset(identity.option_order),
        turn_zero_required_facts=turn_zero_required_facts,
        action_prompt_profiles=action_prompt_profiles,
        target_swap_interventions=(
            {
                focal_side: (
                    resolved_swap_sender,
                    target_swap_targets,
                    target_swap_active_target,
                )
            }
            if condition == "target_swapped"
            else None
        ),
    )
    nonempty = 0
    for turn in raw["turns"]:
        for broadcast in turn["broadcasts"]:
            if broadcast["team"] != focal_side:
                continue
            if broadcast["accepted_message"] != EMPTY_BROADCAST.to_dict():
                nonempty += 1
    final_nodes = raw["turns"][-1]["post_state"]["nodes"]
    captured = (
        critical_target is not None
        and final_nodes[critical_target]["owner"] == focal_side
    )
    row = {
        "case_id": identity.case_id,
        "suite": identity.suite,
        "opponent_id": identity.opponent_id,
        "opponent_revision": identity.opponent_revision,
        "side": focal_side,
        "policy_variant": identity.policy_variant,
        "policy_revision": identity.policy_revision,
        "policy_assignment": identity.policy_assignment,
        "role_assignment": identity.role_assignment,
        "option_order": identity.option_order,
        "condition": condition,
        "sampling_key": identity.sampling_key,
        "terminal_return": raw["metrics"][focal_side]["terminal_return"],
        "messages_nonempty": nonempty,
        "critical_capture": captured,
        "broadcast_protocol_rate": raw["metrics"][focal_side]["broadcast_protocol_rate"],
        "broadcast_grounded_rate": raw["metrics"][focal_side]["broadcast_grounded_rate"],
        "action_protocol_rate": raw["metrics"][focal_side]["action_protocol_rate"],
        "communication_spend": raw["metrics"][focal_side]["communication_spend"],
        "invalid_broadcasts": raw["metrics"][focal_side]["invalid_broadcasts"],
        "invalid_actions": raw["metrics"][focal_side]["invalid_actions"],
        "duplicate_target_turn_rate": raw["metrics"][focal_side][
            "duplicate_target_turn_rate"
        ],
        "seed": seed,
        "size": size,
        "horizon": horizon,
    }
    return row, raw
