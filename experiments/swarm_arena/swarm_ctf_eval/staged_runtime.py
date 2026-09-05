from __future__ import annotations

import json
from pathlib import Path


def orchestrator_lora(orch: dict, policy_id: str) -> dict:
    """Read LoRA metadata from the current serialized orchestrator schema."""
    student = orch.get("student")
    if not isinstance(student, dict) or not isinstance(student.get("model"), dict):
        raise ValueError(f"{policy_id} orchestrator is missing student model metadata")
    lora = student["model"].get("lora")
    if not isinstance(lora, dict):
        raise ValueError(f"{policy_id} orchestrator is missing LoRA metadata")
    return lora


def parity_quarantined_logical_updates(output_dir: Path) -> set[int]:
    """Read trainer-owned, append-only parity quarantine decisions."""
    path = output_dir / "audit" / "rollout_parity_quarantine.jsonl"
    if not path.is_file():
        return set()
    updates: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("version") != "prime-rl-parity-quarantine-v1":
            raise ValueError(f"unknown parity quarantine record at line {line_number}")
        if row.get("replacement_batch_sampled") is not False:
            raise ValueError("parity quarantine record permits replacement sampling")
        if row.get("optimizer_step_applied") is not False:
            raise ValueError("parity quarantine record falsely claims an optimizer step")
        action = row.get("action")
        if action == "abort_quarantine_limit_exceeded":
            raise RuntimeError("parity quarantine limit rejected this run")
        if action != "quarantine_logical_update":
            raise ValueError(f"unknown parity quarantine action at line {line_number}")
        logical_update = int(row["logical_update"])
        if logical_update in updates:
            raise ValueError(f"duplicate parity quarantine for update {logical_update}")
        updates.add(logical_update)
    return updates


def optimizer_application_summary(
    run_dir: Path, *, step: int, interval: int
) -> dict[str, int]:
    """Report applied versus quarantined work for one evaluation stage."""
    progress_path = run_dir / "live_rl_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if not isinstance(progress, list) or len(progress) < step:
        raise ValueError(f"update {step} lacks complete controller progress")
    start = max(1, step - interval + 1)
    rows = progress[start - 1 : step]
    if [int(row["step"]) + 1 for row in rows] != list(range(start, step + 1)):
        raise ValueError(f"update {step} controller progress is not contiguous")
    applied = sum(row.get("optimizer_step_applied") is True for row in rows)
    quarantined = sum(row.get("parity_quarantined") is True for row in rows)
    if applied + quarantined != len(rows):
        raise ValueError(
            f"update {step} progress does not classify every logical update"
        )
    return {
        "logical_updates": len(rows),
        "optimizer_steps_applied": applied,
        "parity_quarantined_updates": quarantined,
        "replacement_batches_sampled": 0,
    }
