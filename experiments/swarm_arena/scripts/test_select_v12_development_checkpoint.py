from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("select_v12_development_checkpoint.py")
spec = importlib.util.spec_from_file_location("v12_selector", MODULE)
selector = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(selector)


def summary(*, semantic=.1, specificity=.1, legacy=(-.01, .01), hard=(-.01, .01)):
    return {
        "tier": "selection",
        "semantic": {
            "candidate_critical_normal_minus_target_swapped": {
                "mean_difference": semantic,
                "mean_difference_95": [semantic - .01, semantic + .01],
            },
            "critical_minus_decoy_semantic_specificity": {
                "mean_difference": specificity,
                "mean_difference_95": [specificity - .01, specificity + .01],
            },
        },
        "capability_rl_minus_sft": {
            "ordinary_legacy": {"mean_difference": 0, "mean_difference_95": list(legacy)},
            "ordinary_hard": {"mean_difference": 0, "mean_difference_95": list(hard)},
        },
    }


def write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_selects_earliest_noninferior_candidate(tmp_path: Path) -> None:
    design = write(tmp_path / "design.json", {})
    first = write(tmp_path / "20.json", summary(legacy=(-.021, .01)))
    second = write(tmp_path / "40.json", summary(legacy=(-.02, .01)))
    result = selector.select([(20, first), (40, second)], design)
    assert result["selected_step"] == 40
    assert result["frozen_launch_authorized"]


def test_exact_margin_passes_but_point_estimate_without_interval_fails(tmp_path: Path) -> None:
    design = write(tmp_path / "design.json", {})
    good = write(tmp_path / "20.json", summary(legacy=(-.02, -.01), hard=(-.02, .02)))
    assert selector.select([(20, good)], design)["selected_step"] == 20
    bad_payload = summary()
    del bad_payload["capability_rl_minus_sft"]["ordinary_legacy"]["mean_difference_95"]
    bad = write(tmp_path / "bad.json", bad_payload)
    with pytest.raises(ValueError, match="mean_difference_95"):
        selector.select([(20, bad)], design)


def test_order_and_no_fallback_are_fail_closed(tmp_path: Path) -> None:
    design = write(tmp_path / "design.json", {})
    failed = write(tmp_path / "20.json", summary(specificity=0))
    result = selector.select([(20, failed)], design)
    assert result["selected_step"] is None
    assert not result["frozen_launch_authorized"]
    with pytest.raises(ValueError, match="ordered prefix"):
        selector.select([(40, failed)], design)
