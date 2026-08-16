from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from swarm_ctf_eval.progress_eval_v5 import PROGRESS_EVAL_V5_VERSION
from swarm_ctf_eval.rl_production import load_production_plan


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_ready(path: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            ready = json.loads(path.read_text(encoding="utf-8"))
            claimed = ready.pop("ready_sha256")
            if _digest(ready) != claimed:
                raise ValueError(f"checkpoint ready hash mismatch: {path}")
            ready["ready_sha256"] = claimed
            return ready
        time.sleep(1.0)
    raise TimeoutError(f"checkpoint did not become ready: {path}")


def _candidate_models(step: int, sft_model: str) -> list[str]:
    """Use one registered alias for the exact step-zero harness control.

    The four trainable aliases contain byte-identical SFT adapters at step zero,
    but vLLM may execute a four-alias LoRA batch through a different floating-
    point path than a one-alias batch.  Greedy choices near a tie can therefore
    differ even though no optimizer step occurred.  The step-zero pulse is an
    evaluator-invariance control, not a test of alias-kernel equivalence, so it
    must route both arms through the same registered SFT alias.  Fresh runtime
    parity and the controller's adapter checksum checks cover the trainable
    aliases separately.  Every post-zero pulse evaluates the four real policy
    aliases.
    """
    if step < 0:
        raise ValueError("pulse step cannot be negative")
    if step == 0:
        return [sft_model] * 4
    return [f"blue-{index}" for index in range(4)]


def _validate_summary(path: Path, *, step: int) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("version") != PROGRESS_EVAL_V5_VERSION:
        raise ValueError(f"update {step} pulse did not use RL-specific evaluation")
    if summary.get("tier") != "pulse" or summary.get("rows") != 16:
        raise ValueError(f"update {step} pulse is incomplete or has the wrong scope")
    if step == 0:
        invariants = (
            summary["capability_rl_minus_sft"]["ordinary_legacy"]["mean_difference"],
            summary["capability_rl_minus_sft"]["ordinary_hard"]["mean_difference"],
            summary["rl_specific_communication_lift"]["mean_difference"],
        )
        if any(abs(float(value)) > 1e-12 for value in invariants):
            raise ValueError("step-zero SFT-vs-SFT pulse failed exact behavioral invariance")
    return summary


def _wait_retained_checkpoints(
    run_dir: Path,
    *,
    step: int,
    policy_adapter_sha256: dict[str, str],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    missing: list[str] = []
    while time.monotonic() < deadline:
        missing = []
        for index in range(4):
            checkpoint = run_dir / f"run_blue_{index}" / "checkpoints" / f"step_{step}"
            required = (
                checkpoint / "STABLE",
                checkpoint / "trainer" / "rank_0.pt",
                checkpoint / "weight" / "STABLE",
                checkpoint / "weight" / "adapter_model.safetensors",
                checkpoint / "weight" / "adapter_config.json",
            )
            missing.extend(str(path) for path in required if not path.is_file())
        if not missing:
            break
        time.sleep(1.0)
    if missing:
        raise TimeoutError(
            f"update {step} lacks complete retained checkpoints: {missing}"
        )
    for index in range(4):
        adapter = (
            run_dir
            / f"run_blue_{index}"
            / "checkpoints"
            / f"step_{step}"
            / "weight"
            / "adapter_model.safetensors"
        )
        expected = policy_adapter_sha256[f"blue-{index}"]
        if _sha256_file(adapter) != expected:
            raise ValueError(f"update {step} checkpoint hash mismatch for blue-{index}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fail-closed 16-game pulses at staged-controller barriers."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--production-plan", type=Path, required=True)
    parser.add_argument("--barrier-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--expected-updates", type=int, default=120)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--wait-timeout", type=float, default=7200.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.expected_updates < 1 or args.interval < 1:
        parser.error("updates and interval must be positive")
    if args.expected_updates % args.interval:
        parser.error("pulse interval must divide expected updates")
    if len(args.base_url) != 3:
        parser.error("staged pulses require exactly three rollout server URLs")

    plan, _ = load_production_plan(args.production_plan)
    if plan.expected_updates != args.expected_updates:
        raise ValueError("pulse schedule disagrees with production plan")
    by_family = {snapshot.family: snapshot for snapshot in plan.opponent_pool.snapshots}
    eval_script = args.repo_root / "experiments" / "swarm_arena" / "scripts" / "run_progress_eval_v4.py"
    args.eval_root.mkdir(parents=True, exist_ok=True)

    for step in range(0, args.expected_updates + 1, args.interval):
        ready_path = args.barrier_dir / f"step_{step}.ready.json"
        continue_path = args.barrier_dir / f"step_{step}.continue.json"
        ready = _wait_ready(ready_path, args.wait_timeout)
        if ready["production_plan_sha256"] != plan.sha256 or ready["step"] != step:
            raise ValueError(f"checkpoint barrier identity mismatch at step {step}")
        if step:
            _wait_retained_checkpoints(
                args.run_dir,
                step=step,
                policy_adapter_sha256=ready["policy_adapter_sha256"],
                timeout=args.checkpoint_timeout,
            )
        output_dir = args.eval_root / f"update-{step}"
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            config = {
                "base_urls": [value.rstrip("/") + "/v1" for value in args.base_url],
                "candidate": {
                    "revision": ready["policy_revision"],
                    "models": _candidate_models(step, by_family["sft"].model_name),
                },
                "baseline": {
                    "revision": args.baseline_revision,
                    "models": [by_family["sft"].model_name] * 4,
                },
                "opponents": [
                    {
                        "id": "base",
                        "revision": by_family["base"].revision,
                        "models": [by_family["base"].model_name] * 4,
                    },
                    {
                        "id": "sft",
                        "revision": by_family["sft"].revision,
                        "models": [by_family["sft"].model_name] * 4,
                    },
                    {
                        "id": "historical_league",
                        "revision": by_family["historical"].revision,
                        "models": [by_family["historical"].model_name] * 4,
                    },
                ],
            }
            config_path = args.eval_root / f"update-{step}-config.json"
            _atomic_json(config_path, config)
            command = [
                sys.executable,
                str(eval_script),
                "--tier",
                "pulse",
                "--config",
                str(config_path),
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(output_dir),
                "--monitor-opponent-id",
                "sft",
                "--rl-specific-communication",
            ]
            if output_dir.exists():
                command.append("--resume")
            completed = subprocess.run(
                command,
                cwd=args.repo_root,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"development pulse failed at update {step}")
        _validate_summary(summary_path, step=step)
        continuation = {
            "version": "swarm-checkpoint-barrier-v1",
            "step": step,
            "ready_sha256": ready["ready_sha256"],
        }
        if continue_path.is_file():
            if json.loads(continue_path.read_text(encoding="utf-8")) != continuation:
                raise ValueError(f"existing continuation mismatch at update {step}")
        else:
            _atomic_json(continue_path, continuation)

    (args.eval_root / "COMPLETE").touch()
    print(
        json.dumps(
            {
                "status": "complete",
                "updates": list(range(0, args.expected_updates + 1, args.interval)),
                "production_plan_sha256": plan.sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
