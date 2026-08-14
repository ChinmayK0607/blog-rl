from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERSION = "arena-development-collapse-audit-v1"


def _concentration(values: list[str]) -> float | None:
    if not values:
        return None
    return max(Counter(values).values()) / len(values)


def _message_target(message: dict[str, Any]) -> str | None:
    intent = message.get("intent")
    if isinstance(intent, dict) and isinstance(intent.get("target"), str):
        return str(intent["target"])
    facts = message.get("facts")
    if isinstance(facts, list) and facts and isinstance(facts[0], dict):
        node = facts[0].get("node")
        return str(node) if isinstance(node, str) else None
    return None


def _is_nonempty(message: dict[str, Any]) -> bool:
    return bool(
        message.get("facts")
        or message.get("intent") is not None
        or int(message.get("request_resource", 0)) > 0
    )


def _behavior_summary(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    kl_report: dict[str, Any],
    *,
    speaking_extreme: float = 0.98,
    concentration_limit: float = 0.95,
    kl_mean_limit: float = 0.08,
    kl_p99_limit: float = 0.30,
) -> dict[str, Any]:
    messages: dict[str, list[bool]] = defaultdict(list)
    targets: dict[str, list[str]] = defaultdict(list)
    actions: dict[str, list[str]] = defaultdict(list)
    for row, raw in records:
        side = str(row["side"])
        agent_models = raw[f"{side.lower()}_agent_models"]
        for turn in raw["turns"]:
            for broadcast in turn["broadcasts"]:
                if broadcast["team"] != side:
                    continue
                policy = str(agent_models[broadcast["agent_id"]])
                message = broadcast["parsed_message"]
                nonempty = _is_nonempty(message)
                messages[policy].append(nonempty)
                target = _message_target(message)
                if nonempty and target is not None:
                    targets[policy].append(target)
            for action in turn["actions"]:
                if action["team"] != side:
                    continue
                policy = str(agent_models[action["agent_id"]])
                actions[policy].append(
                    json.dumps(action["selected_action"], sort_keys=True, separators=(",", ":"))
                )

    per_policy = {}
    expected = set(kl_report["per_policy"])
    if set(messages) != expected or set(actions) != expected:
        raise ValueError("evaluation policy identities differ from the KL report")
    for policy in sorted(expected):
        speaking_rate = statistics.mean(messages[policy])
        target_concentration = _concentration(targets[policy])
        action_concentration = _concentration(actions[policy])
        kl = kl_report["per_policy"][policy]["candidate_to_baseline_kl"]
        per_policy[policy] = {
            "decisions": len(actions[policy]),
            "messages": len(messages[policy]),
            "speaking_rate": speaking_rate,
            "always_speaking": speaking_rate >= speaking_extreme,
            "never_speaking": speaking_rate <= 1 - speaking_extreme,
            "message_targets": len(targets[policy]),
            "message_target_concentration": target_concentration,
            "repeated_target_collapse": len(targets[policy]) >= 20
            and target_concentration is not None
            and target_concentration >= concentration_limit,
            "action_concentration": action_concentration,
            "action_collapse": len(actions[policy]) >= 20
            and action_concentration is not None
            and action_concentration >= concentration_limit,
            "reference_state_kl_mean": float(kl["mean"]),
            "reference_state_kl_p99": float(kl["p99"]),
            "excessive_kl": float(kl["mean"]) > kl_mean_limit
            or float(kl["p99"]) > kl_p99_limit,
        }
    return per_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a completed development evaluation for policy collapse."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--policy-kl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite collapse audit: {args.output}")

    rows = {
        str(row["evaluation_id"]): row
        for line in (args.eval_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line and (row := json.loads(line))
    }
    raw = {
        str(record["evaluation_id"]): record["raw"]
        for line in (args.eval_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line and (record := json.loads(line))
    }
    if set(rows) != set(raw):
        raise ValueError("evaluation rows and raw trajectories are incomplete or misaligned")
    selected = [
        (row, raw[evaluation_id])
        for evaluation_id, row in sorted(rows.items())
        if row["policy_variant"] == "candidate_rl" and row["condition"] == "normal"
    ]
    if not selected:
        raise ValueError("collapse audit has no candidate normal trajectories")
    summary = json.loads((args.eval_dir / "summary.json").read_text(encoding="utf-8"))
    kl_report = json.loads(args.policy_kl.read_text(encoding="utf-8"))
    per_policy = _behavior_summary(selected, kl_report)
    opponent_returns = {
        str(key): float(value)
        for key, value in summary["candidate_normal_return_by_opponent"].items()
    }
    return_gain = float(summary["ordinary_candidate_minus_sft"]["mean_difference"])
    message_gain = float(
        summary["critical_normal_minus_intervention"]["dropped"]["mean_difference"]
    )
    flags = {
        "always_or_never_speaking": any(
            item["always_speaking"] or item["never_speaking"]
            for item in per_policy.values()
        ),
        "repeated_target_collapse": any(
            item["repeated_target_collapse"] for item in per_policy.values()
        ),
        "action_collapse": any(item["action_collapse"] for item in per_policy.values()),
        "excessive_kl": any(item["excessive_kl"] for item in per_policy.values()),
        "performance_against_only_one_opponent": len(opponent_returns) >= 3
        and sum(value > 0 for value in opponent_returns.values()) <= 1,
        "return_gain_without_message_gain": return_gain > 0.01 and message_gain <= 0.005,
    }
    report = {
        "version": VERSION,
        "per_policy": per_policy,
        "mean_return_gain": return_gain,
        "mean_critical_normal_minus_dropped": message_gain,
        "mean_return_by_opponent": opponent_returns,
        "flags": flags,
        "passed": not any(flags.values()),
        "scope": "development diagnostic stop/inspect gates; never reward terms",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
