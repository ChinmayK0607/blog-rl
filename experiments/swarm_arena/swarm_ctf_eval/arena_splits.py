from __future__ import annotations

import hashlib
import json

# Frozen before model comparison. These seeds are never valid SFT seeds. Twenty
# cases at each topology size give 60 total cases, balanced across three unseen
# opponent policies. The tuple construction is deterministic and its hash is
# recorded in every report.
_FROZEN_SEEDS = tuple(1_000_003 + 137 * index for index in range(60))
_STYLES = ("balanced", "aggressive", "defensive")
FROZEN_EVAL_CASES: tuple[tuple[int, int, str], ...] = tuple(
    (seed, 12 + index // 20, _STYLES[index % len(_STYLES)]) for index, seed in enumerate(_FROZEN_SEEDS)
)

FROZEN_EVAL_SEEDS = frozenset(seed for seed, _, _ in FROZEN_EVAL_CASES)
FROZEN_EVAL_MANIFEST_SHA256 = hashlib.sha256(json.dumps(FROZEN_EVAL_CASES, separators=(",", ":")).encode()).hexdigest()
