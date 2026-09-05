import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.freeze_v14_8_measurement_repair import build_bundle


def test_measurement_bundle_reproduces_and_preserves_science():
    repo = Path(__file__).resolve().parents[3]
    actual = build_bundle(repo)
    expected = json.loads((repo / "experiments/swarm_arena/data/rl_v14_8/cpu_bundle.json").read_text())
    parent = json.loads((repo / "experiments/swarm_arena/data/rl_v14_7/cpu_bundle.json").read_text())
    assert actual == expected
    for key in ("initializer", "trainer", "inference", "routing", "artifact_file_sha256", "execution"):
        assert actual[key] == parent[key]
    assert actual["gpu_budget"]["maximum_wall_hours"] == 30
    assert not actual["frozen_data_opened"]


@pytest.mark.parametrize("authorized,seconds,success", [(True, 108000, True), (False, 108000, False), (True, 32400, False)])
def test_extended_reservation_is_explicit_and_still_budgeted(tmp_path, authorized, seconds, success):
    config = tmp_path / "config.toml"
    config.write_text("test = true\n")
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    profile = {
        "version": "staged-reservation-profile-v1",
        "extended_time_authorization": "user: extend time and use A6000" if authorized else "",
        "update_timing_basis": "unmeasured_conservative_reservation",
        "trainer_config_sha256": config_sha, "inference_config_sha256": config_sha,
        "topology": "0/1,2,3", "gpu_model": "NVIDIA RTX A6000", "game_concurrency": 1,
        "games_per_minute": 1.3, "update_seconds": 900, "remaining_setup_seconds": 3600,
        "checkpoint_seconds": 600, "safety_factor": 1.25, "evidence": ["historical A6000 timing"],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    output = tmp_path / "budget.json"
    script = Path(__file__).parents[1] / "scripts/preflight_staged_budget.py"
    result = subprocess.run([sys.executable, str(script), "--profile", str(path), "--expected-updates", "40",
        "--interval", "10", "--available-seconds", str(seconds), "--inference-config", str(config),
        "--trainer-config", str(config), "--topology", "0/1,2,3", "--gpu-model", "NVIDIA RTX A6000",
        "--output", str(output)], capture_output=True, text=True)
    assert (result.returncode == 0) is success, result.stderr
    if success:
        report = json.loads(output.read_text())
        assert report["fresh_games"] == 672
        assert report["measured_training_throughput"] is False
        assert 25 < report["required_seconds"] / 3600 < 27
