from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).with_name("assess_v12_repair_pulses.py")
spec = importlib.util.spec_from_file_location("v12_pulses", MODULE)
pulses = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pulses)


def summary(semantic: float, specificity: float, legacy: float, hard: float):
    return {
        "semantic": {
            "candidate_critical_normal_minus_target_swapped": {"mean_difference": semantic},
            "critical_minus_decoy_semantic_specificity": {"mean_difference": specificity},
        },
        "capability_rl_minus_sft": {
            "ordinary_legacy": {"mean_difference": legacy},
            "ordinary_hard": {"mean_difference": hard},
        },
    }


def test_stops_only_after_two_three_way_failures() -> None:
    baseline = summary(.04, .03, -.01, .01)
    failed = summary(0, .02, -.02, 0)
    assert pulses.assess(baseline, [(20, failed), (40, failed)])["decision"] == "stop"


def test_any_repair_signal_continues_without_promoting_a_checkpoint() -> None:
    baseline = summary(.04, .03, -.01, .01)
    failed = summary(0, .02, -.02, 0)
    repaired = summary(-.01, .02, -.005, .02)
    result = pulses.assess(baseline, [(20, failed), (40, repaired)])
    assert result["decision"] == "continue"
    assert result["frozen_data_opened"] is False
