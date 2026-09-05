from __future__ import annotations

from swarm_ctf_eval.episode_baselines import POLICIES, run_case
from swarm_ctf_eval.episode_splits import EPISODE_EVAL_CASES


def test_all_episode_baselines_finish_without_protocol_errors() -> None:
    case = EPISODE_EVAL_CASES[0]
    for policy in POLICIES:
        row = run_case(case, policy)
        assert row["invalid_actions"] == 0
        assert row["invalid_broadcasts"] == 0
        assert row["communication_spend"] == 0
