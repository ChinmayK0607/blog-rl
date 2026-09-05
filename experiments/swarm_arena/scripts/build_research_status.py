"""Render a timestamped status snapshot from a compact, evidence-backed run record."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.record.read_bytes()
    row = json.loads(data)
    cost_text = row.get("cost_description") or (
        f'Estimated final cost: ${row["estimated_final_cost_usd"]:.2f}; '
        f'last provider reading ${row["provider_last_spend_usd"]:.2f}. '
        f'Final invoice verified: {row["final_invoice_verified"]}.'
    )
    report = f"""# Latest recorded run status

Generated from `{args.record.name}` (SHA-256 `{hashlib.sha256(data).hexdigest()}`).
Last verified **{row["observed_at_utc"]}**; this is not a live provider query.

- Run: `{row["run_id"]}` at `{row["source_commit"]}`.
- State: **{row["status"]}** — {row["reason"]}.
- Durable optimizer updates: **{row["durable_optimizer_updates"]}**.
- Evaluation: {row["evaluation_rows"]}/{row["evaluation_rows_required"]} rows; {row["measurement_scope"]}.
- Scientific classification: `{row["scientific_verdict"]}`.
- Exact pod absence verified: {row["provider_absence_verified"]} (`{row["pod_id"]}`).
- {cost_text}
- HF revision: `{row["hf_revision"]}`; {row["hf_and_wandb_hash_verified_files"]} files independently hash-verified on each mirror.
- Next decision: {row["next_decision"]}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)


if __name__ == "__main__":
    main()
