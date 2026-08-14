from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_ctf_eval.message_credit_audit import summarize_message_credit_records
from swarm_ctf_eval.safety_supervisor import verify_hash_chain


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen Stage B audit gates.")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = verify_hash_chain(args.evidence)
    summary = summarize_message_credit_records(
        [record["payload"] for record in records]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], sort_keys=True))
    print(json.dumps({"gates": summary["gates"], "verdict": summary["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
