#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from experiments.rebuttal.statistics_utils import mean_std_ci95
except ModuleNotFoundError:
    from statistics_utils import mean_std_ci95


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            row["seed"] = int(row["seed"])
            row["step"] = int(row["step"])
            yield row


def load_scores(path):
    with open(path, encoding="utf-8") as fp:
        payload = json.load(fp)
    scores = {}
    for dimension, value in payload.items():
        if isinstance(value, list):
            value = value[0]
        if not isinstance(value, (int, float)):
            raise ValueError(f"{path}: {dimension} has unsupported value {type(value)}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{path}: {dimension} has non-finite score {value}")
        scores[dimension] = float(value)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-dimension VBench means, sample std, and 95% CIs over seeds."
    )
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--expected_seeds",
        nargs="*",
        type=int,
        default=[],
        help="Fail unless every result group contains exactly these unique training seeds.",
    )
    args = parser.parse_args()

    long_rows = []
    for manifest in args.manifests:
        for row in read_manifest(manifest):
            result_path = Path(row["result_json"])
            if not result_path.is_absolute():
                result_path = Path(manifest).resolve().parent / result_path
            for dimension, score in load_scores(result_path).items():
                long_rows.append(
                    {
                        **row,
                        "dimension": dimension,
                        "score": score,
                    }
                )
    if not long_rows:
        raise ValueError("No VBench records found")

    groups = defaultdict(dict)
    for row in long_rows:
        key = (
            row["model"],
            row.get("schedule", ""),
            row["step"],
            row.get("window", "full"),
            row["dimension"],
        )
        seed = row["seed"]
        if seed in groups[key]:
            raise ValueError(f"Duplicate seed={seed} for VBench group {key}")
        groups[key][seed] = row["score"]

    summary_rows = []
    expected_seeds = set(args.expected_seeds)
    if len(expected_seeds) != len(args.expected_seeds):
        raise ValueError("--expected_seeds contains duplicates")
    for key, seed_scores in sorted(groups.items()):
        model, schedule, step, window, dimension = key
        actual_seeds = set(seed_scores)
        if expected_seeds and actual_seeds != expected_seeds:
            raise ValueError(
                f"VBench group {key} has seeds {sorted(actual_seeds)}, "
                f"expected {sorted(expected_seeds)}"
            )
        values = [seed_scores[seed] for seed in sorted(seed_scores)]
        mean, std, ci95 = mean_std_ci95(values)
        summary_rows.append(
            {
                "model": model,
                "schedule": schedule,
                "step": step,
                "window": window,
                "dimension": dimension,
                "num_seeds": len(seed_scores),
                "seeds": ",".join(str(seed) for seed in sorted(seed_scores)),
                "mean": mean,
                "sample_std": std,
                "ci95": ci95,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "vbench_long.csv"
    summary_path = output_dir / "vbench_summary.csv"
    with open(long_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    with open(summary_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
