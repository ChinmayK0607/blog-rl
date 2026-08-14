from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = [group for step in payload for group in step["groups"]]
    by_kind: dict[str, list[dict]] = {"critical": [], "decoy": []}
    for group in groups:
        by_kind[group["scenario"]["kind"]].append(group)

    summary: dict[str, object] = {
        "input": str(path),
        "cases": len(groups),
        "kinds": {},
    }
    kind_rows = summary["kinds"]
    assert isinstance(kind_rows, dict)
    for kind, rows in by_kind.items():
        if not rows:
            kind_rows[kind] = {"cases": 0}
            continue
        credits = [
            float(value)
            for row in rows
            for value in row["advantages"].values()
        ]
        sender_credits = [
            float(row["advantages"][row["scenario"]["sender"]]) for row in rows
        ]
        receiver_credits = [
            float(row["advantages"][row["scenario"]["receiver"]]) for row in rows
        ]
        off_role_nonzero_cases = 0
        for row in rows:
            intended = {row["scenario"]["sender"], row["scenario"]["receiver"]}
            if any(
                abs(float(value)) > 1e-12
                for agent, value in row["advantages"].items()
                if agent not in intended
            ):
                off_role_nonzero_cases += 1
        kind_rows[kind] = {
            "cases": len(rows),
            "mean_return": fmean(float(row["actual_return"]) for row in rows),
            "nonzero_credit_case_rate": fmean(
                any(abs(float(value)) > 1e-12 for value in row["advantages"].values())
                for row in rows
            ),
            "mean_absolute_agent_credit": fmean(abs(value) for value in credits),
            "positive_agent_credits": sum(value > 1e-12 for value in credits),
            "negative_agent_credits": sum(value < -1e-12 for value in credits),
            "sender_nonzero_rate": fmean(abs(value) > 1e-12 for value in sender_credits),
            "receiver_nonzero_rate": fmean(
                abs(value) > 1e-12 for value in receiver_credits
            ),
            "off_role_nonzero_case_rate": off_role_nonzero_cases / len(rows),
        }

    critical_by_pair = {
        row["scenario"]["pair_index"]: row for row in by_kind["critical"]
    }
    decoy_by_pair = {
        row["scenario"]["pair_index"]: row for row in by_kind["decoy"]
    }
    paired = []
    paired_credits: list[float] = []
    paired_sender: list[float] = []
    paired_receiver: list[float] = []
    paired_off_role_cases = 0
    common_namespaces = True
    for pair_index in sorted(set(critical_by_pair) & set(decoy_by_pair)):
        critical = critical_by_pair[pair_index]
        decoy = decoy_by_pair[pair_index]
        differences = {
            agent: float(critical["advantages"][agent])
            - float(decoy["advantages"][agent])
            for agent in sorted(critical["advantages"])
        }
        sender = critical["scenario"]["sender"]
        receiver = critical["scenario"]["receiver"]
        intended = {sender, receiver}
        paired_credits.extend(differences.values())
        paired_sender.append(differences[sender])
        paired_receiver.append(differences[receiver])
        paired_off_role_cases += any(
            abs(value) > 1e-12
            for agent, value in differences.items()
            if agent not in intended
        )
        common_namespaces &= (
            critical["scenario"].get("sampling_namespace")
            == decoy["scenario"].get("sampling_namespace")
            and critical["scenario"].get("sampling_namespace") is not None
        )
        paired.append(
            {
                "pair_index": pair_index,
                "critical_return": critical["actual_return"],
                "decoy_return": decoy["actual_return"],
                "critical_minus_decoy_advantage": differences,
            }
        )
    summary["paired"] = paired
    if paired:
        summary["paired_summary"] = {
            "pairs": len(paired),
            "common_sampling_namespaces": common_namespaces,
            "nonzero_credit_pair_rate": fmean(
                any(abs(value) > 1e-12 for value in row["critical_minus_decoy_advantage"].values())
                for row in paired
            ),
            "mean_absolute_agent_credit": fmean(
                abs(value) for value in paired_credits
            ),
            "positive_agent_credits": sum(value > 1e-12 for value in paired_credits),
            "negative_agent_credits": sum(value < -1e-12 for value in paired_credits),
            "sender_nonzero_rate": fmean(
                abs(value) > 1e-12 for value in paired_sender
            ),
            "receiver_nonzero_rate": fmean(
                abs(value) > 1e-12 for value in paired_receiver
            ),
            "off_role_nonzero_pair_rate": paired_off_role_cases / len(paired),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a live RL rollout-only diagnostic.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.input)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
