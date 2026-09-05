from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    instruction: str
    observation: dict[str, Any]
    legal_actions: tuple[dict[str, Any], ...]
    acceptable_actions: tuple[dict[str, Any], ...]
    required_message_facts: tuple[tuple[str, ...], ...] = ()
    forbidden_message_facts: tuple[tuple[str, ...], ...] = ()
    message_required: bool = False
    pair_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
