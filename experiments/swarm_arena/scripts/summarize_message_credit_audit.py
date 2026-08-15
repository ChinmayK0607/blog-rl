from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swarm_ctf_eval.message_credit_audit import summarize_message_credit_records
from swarm_ctf_eval.safety_supervisor import verify_hash_chain


def load_verified_payloads(evidence_paths: list[Path]) -> list[dict[str, Any]]:
    """Verify each independent chain before combining its payloads."""
    return [
        record["payload"]
        for evidence_path in evidence_paths
        for record in verify_hash_chain(evidence_path)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen Stage B audit gates.")
    parser.add_argument(
        "evidence",
        type=Path,
        nargs="+",
        help="One or more independently hash-chained evidence shards.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_message_credit_records(load_verified_payloads(args.evidence))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], sort_keys=True))
    print(json.dumps({"gates": summary["gates"], "verdict": summary["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
