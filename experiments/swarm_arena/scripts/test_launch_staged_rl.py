from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).with_name("launch_staged_rl.sh")


def test_launcher_uses_one_frozen_flash_attention_uv_runtime() -> None:
    body = SCRIPT.read_text(encoding="utf-8")
    assert "swarm_uv_args=(run --frozen --extra flash-attn)" in body
    assert "$swarm_uv run " not in body
    assert '"$swarm_uv" run ' not in body
    assert body.count("${swarm_uv_command}") == 7


def test_launcher_requires_exact_public_model_repositories() -> None:
    body = SCRIPT.read_text(encoding="utf-8")
    assert "SWARM_PUBLIC_BASE_REPO:?" in body
    assert "SWARM_PUBLIC_ADAPTER_REPO:?" in body
    assert '--expected-public-base-repo "$SWARM_PUBLIC_BASE_REPO"' in body
    assert '--expected-public-adapter-repo "$SWARM_PUBLIC_ADAPTER_REPO"' in body


def test_launcher_uses_declared_gpu_partition_and_dynamic_rollout_urls() -> None:
    body = SCRIPT.read_text(encoding="utf-8")
    assert "SWARM_TRAINER_GPU_IDS:-0" in body
    assert "SWARM_INFERENCE_GPU_IDS:-1,2,3" in body
    assert "SWARM_ROLLOUT_PORTS:-8001,8002,8003" in body
    assert "CUDA_VISIBLE_DEVICES=$swarm_trainer_gpu_ids" in body
    assert "--nproc-per-node=${#swarm_trainer_gpus[@]}" in body
    assert '"${swarm_base_url_args[@]}"' in body
    assert "${swarm_base_url_command}" in body
