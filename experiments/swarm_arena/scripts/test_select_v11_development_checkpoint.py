import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).with_name("select_v11_development_checkpoint.py")
spec = importlib.util.spec_from_file_location("selector", MODULE)
selector = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(selector)


def summary(semantic=0.1, specificity=0.1, legacy=0.0, hard=0.0):
    return {
        "tier": "selection",
        "semantic": {
            "candidate_critical_normal_minus_target_swapped": {"mean_difference": semantic},
            "critical_minus_decoy_semantic_specificity": {"mean_difference": specificity},
        },
        "capability_rl_minus_sft": {
            "ordinary_legacy": {"mean_difference": legacy},
            "ordinary_hard": {"mean_difference": hard},
        },
    }


def write(path, content):
    path.write_text(json.dumps(content))
    return path


def test_earliest_pass_and_later_marked_skipped(tmp_path):
    design = write(tmp_path / "design.json", {})
    clarification = write(tmp_path / "clarification.json", {})
    paths = [write(tmp_path / f"{step}.json", summary(semantic=(-.1 if step == 60 else .1))) for step in (60, 120)]
    result = selector.select(list(zip((60, 120), paths)), design, clarification)
    assert result["selected_step"] == 120
    assert result["candidates"][2]["skipped_after_earliest_selection"] is True


def test_tie_selects_first_and_zero_ordinary_is_allowed(tmp_path):
    design = write(tmp_path / "design.json", {})
    clarification = write(tmp_path / "clarification.json", {})
    paths = [write(tmp_path / f"{step}.json", summary()) for step in (60, 120, 180)]
    assert selector.select(list(zip((60, 120, 180), paths)), design, clarification)["selected_step"] == 60


def test_no_fallback_when_none_pass(tmp_path):
    design = write(tmp_path / "design.json", {})
    clarification = write(tmp_path / "clarification.json", {})
    paths = [write(tmp_path / f"{step}.json", summary(specificity=0)) for step in (60, 120, 180)]
    result = selector.select(list(zip((60, 120, 180), paths)), design, clarification)
    assert result["selected_step"] is None
    assert result["frozen_launch_authorized"] is False


def test_missing_key_fails_closed(tmp_path):
    design = write(tmp_path / "design.json", {})
    clarification = write(tmp_path / "clarification.json", {})
    bad = write(tmp_path / "60.json", {"tier": "selection"})
    good = write(tmp_path / "120.json", summary())
    try:
        selector.select([(60, bad), (120, good), (180, good)], design, clarification)
    except ValueError as error:
        assert "missing or ambiguous selector key" in str(error)
    else:
        raise AssertionError("missing key should fail closed")
