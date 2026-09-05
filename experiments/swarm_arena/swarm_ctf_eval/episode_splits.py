from __future__ import annotations

import hashlib
import json

# Frozen OOD cases: training should use sizes 12-13 and horizons 4-6. Evaluation
# holds out larger graphs, longer credit horizons, and opponent-style switches.
_SEEDS = tuple(2_000_003 + 193 * index for index in range(72))
_STYLES = ("balanced", "aggressive", "defensive")
EPISODE_EVAL_CASES: tuple[tuple[int, int, int, str, str], ...] = tuple(
    (
        seed,
        14 if index < 36 else 16,
        6 if index % 2 == 0 else 8,
        _STYLES[index % 3],
        _STYLES[(index + 1) % 3],
    )
    for index, seed in enumerate(_SEEDS)
)
EPISODE_EVAL_SEEDS = frozenset(case[0] for case in EPISODE_EVAL_CASES)
EPISODE_EVAL_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(EPISODE_EVAL_CASES, separators=(",", ":")).encode()
).hexdigest()
