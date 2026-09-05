from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


CaseKey = tuple[int, int, str]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def case_key(row: dict[str, Any]) -> CaseKey:
    return int(row["seed"]), int(row["size"]), str(row["opponent_style"])


def condition(row: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        item for item in row["conditions"] if item["condition"] == name and item["permutation"] == 0
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} condition for case {case_key(row)}, found {len(matches)}")
    return matches[0]


def mean_ci(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - radius, mean + radius]


def randomization_p_value(values: list[float], trials: int = 100_000, seed: int = 0) -> float:
    observed = abs(statistics.mean(values))
    if observed == 0:
        return 1.0
    generator = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(trials):
        randomized = statistics.mean(value if generator.getrandbits(1) else -value for value in values)
        at_least_as_extreme += int(abs(randomized) >= observed - 1e-12)
    return (at_least_as_extreme + 1) / (trials + 1)


def paired_effect(values: list[float], *, trials: int) -> dict[str, Any]:
    interval = mean_ci(values)
    return {
        "cases": len(values),
        "mean_difference": statistics.mean(values),
        "mean_difference_95": interval,
        "randomization_p_two_sided": randomization_p_value(values, trials=trials),
        "positive_case_rate": statistics.mean(value > 0 for value in values),
        "zero_case_rate": statistics.mean(value == 0 for value in values),
    }


def compare(
    base_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    base_summary: dict[str, Any],
    sft_summary: dict[str, Any],
    *,
    trials: int = 100_000,
) -> dict[str, Any]:
    base = {case_key(row): row for row in base_rows}
    sft = {case_key(row): row for row in sft_rows}
    if len(base) != len(base_rows) or len(sft) != len(sft_rows):
        raise ValueError("duplicate arena case identity")
    if set(base) != set(sft):
        raise ValueError("base and SFT runs do not contain identical frozen cases")
    for field in ("eval_version", "arena_version", "manifest_sha256"):
        if base_summary[field] != sft_summary[field]:
            raise ValueError(f"base and SFT summaries disagree on {field}")

    ordered_keys = sorted(base)
    regret_delta = [
        float(condition(sft[key], "generated")["regret"])
        - float(condition(base[key], "generated")["regret"])
        for key in ordered_keys
    ]
    reward_delta = [
        float(condition(sft[key], "generated")["environment_reward"])
        - float(condition(base[key], "generated")["environment_reward"])
        for key in ordered_keys
    ]
    generated_minus_dropped = [
        float(condition(sft[key], "generated")["environment_reward"])
        - float(condition(sft[key], "dropped")["environment_reward"])
        for key in ordered_keys
    ]
    generated_minus_shuffled = [
        float(condition(sft[key], "generated")["environment_reward"])
        - float(condition(sft[key], "shuffled")["environment_reward"])
        for key in ordered_keys
    ]

    primary = paired_effect(regret_delta, trials=trials)
    primary["direction"] = "negative favors SFT"
    reward = paired_effect(reward_delta, trials=trials)
    reward["direction"] = "positive favors SFT"
    message_vs_none = paired_effect(generated_minus_dropped, trials=trials)
    message_vs_none["direction"] = "positive supports useful generated communication"
    message_vs_wrong = paired_effect(generated_minus_shuffled, trials=trials)
    message_vs_wrong["direction"] = "positive supports message-semantic sensitivity"

    primary_supported = primary["mean_difference_95"][1] < 0
    communication_supported = message_vs_none["mean_difference_95"][0] > 0
    semantic_supported = message_vs_wrong["mean_difference_95"][0] > 0
    return {
        "comparison_protocol": "frozen-paired-arena-v1",
        "eval_version": sft_summary["eval_version"],
        "arena_version": sft_summary["arena_version"],
        "manifest_sha256": sft_summary["manifest_sha256"],
        "num_cases": len(ordered_keys),
        "randomization_trials": trials,
        "randomization_seed": 0,
        "primary_endpoint": {
            "name": "generated-condition oracle regret: SFT minus base",
            **primary,
        },
        "supporting_endpoint": {
            "name": "generated-condition environment reward: SFT minus base",
            **reward,
        },
        "mechanism_checks": {
            "generated_minus_dropped_reward": message_vs_none,
            "generated_minus_shuffled_reward": message_vs_wrong,
        },
        "protocol": {
            "base_message_strict_rate": base_summary["message_strict_rate"],
            "sft_message_strict_rate": sft_summary["message_strict_rate"],
            "base_action_order_consistency_rate": base_summary["action_order_consistency_rate"],
            "sft_action_order_consistency_rate": sft_summary["action_order_consistency_rate"],
        },
        "claim_gates": {
            "lower_oracle_regret_95_excludes_zero": primary_supported,
            "generated_beats_dropped_95_excludes_zero": communication_supported,
            "generated_beats_shuffled_95_excludes_zero": semantic_supported,
            "coordination_improvement_supported": primary_supported
            and communication_supported
            and semantic_supported,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of base and SFT frozen-arena runs.")
    parser.add_argument("--base-rows", type=Path, required=True)
    parser.add_argument("--sft-rows", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        load_rows(args.base_rows),
        load_rows(args.sft_rows),
        json.loads(args.base_summary.read_text(encoding="utf-8")),
        json.loads(args.sft_summary.read_text(encoding="utf-8")),
        trials=args.trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
