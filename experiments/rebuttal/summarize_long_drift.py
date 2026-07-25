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


POSITION_X = {"early": 0.0, "middle": 0.5, "late": 1.0}


def result_scores(path):
    with open(path, encoding="utf-8") as fp:
        payload = json.load(fp)
    scores = {
        key: float(value[0] if isinstance(value, list) else value)
        for key, value in payload.items()
    }
    for key, value in scores.items():
        if not math.isfinite(value):
            raise ValueError(f"{path}: {key} has non-finite score {value}")
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute late-minus-early VBench drift and a three-window slope."
    )
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--expected_seeds",
        nargs="*",
        type=int,
        default=[],
        help="Fail unless every drift group contains exactly these unique training seeds.",
    )
    args = parser.parse_args()

    grouped = defaultdict(dict)
    for manifest in args.manifests:
        manifest_path = Path(manifest).resolve()
        with open(manifest_path, newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                window = row["window"]
                if window not in POSITION_X:
                    continue
                result_path = Path(row["result_json"])
                if not result_path.is_absolute():
                    result_path = manifest_path.parent / result_path
                key = (
                    row["model"],
                    row.get("schedule", ""),
                    int(row["seed"]),
                    int(row["step"]),
                )
                if window in grouped[key]:
                    raise ValueError(f"Duplicate {key} window={window}")
                grouped[key][window] = result_scores(result_path)

    drift_rows = []
    for key, windows in sorted(grouped.items()):
        missing = set(POSITION_X) - set(windows)
        if missing:
            raise ValueError(f"{key} is missing windows {sorted(missing)}")
        dimensions = set(windows["early"])
        if any(set(scores) != dimensions for scores in windows.values()):
            raise ValueError(f"{key} has inconsistent VBench dimensions across windows")
        model, schedule, seed, step = key
        for dimension in sorted(dimensions):
            values = [windows[position][dimension] for position in ("early", "middle", "late")]
            slope = values[2] - values[0]
            drift_rows.append(
                {
                    "model": model,
                    "schedule": schedule,
                    "seed": seed,
                    "step": step,
                    "dimension": dimension,
                    "early": values[0],
                    "middle": values[1],
                    "late": values[2],
                    "late_minus_early": values[2] - values[0],
                    "three_window_linear_slope": slope,
                }
            )
    if not drift_rows:
        raise ValueError("No complete early/middle/late records")

    summary_groups = defaultdict(dict)
    for row in drift_rows:
        key = (row["model"], row["schedule"], row["step"], row["dimension"])
        seed = row["seed"]
        if seed in summary_groups[key]:
            raise ValueError(f"Duplicate seed={seed} for drift group {key}")
        summary_groups[key][seed] = row["late_minus_early"]
    summary_rows = []
    expected_seeds = set(args.expected_seeds)
    if len(expected_seeds) != len(args.expected_seeds):
        raise ValueError("--expected_seeds contains duplicates")
    for key, seed_values in sorted(summary_groups.items()):
        actual_seeds = set(seed_values)
        if expected_seeds and actual_seeds != expected_seeds:
            raise ValueError(
                f"Drift group {key} has seeds {sorted(actual_seeds)}, "
                f"expected {sorted(expected_seeds)}"
            )
        values = [seed_values[seed] for seed in sorted(seed_values)]
        mean, std, ci95 = mean_std_ci95(values)
        summary_rows.append(
            {
                "model": key[0],
                "schedule": key[1],
                "step": key[2],
                "dimension": key[3],
                "num_seeds": len(seed_values),
                "seeds": ",".join(str(seed) for seed in sorted(seed_values)),
                "mean_late_minus_early": mean,
                "sample_std": std,
                "ci95": ci95,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "long_drift_per_seed.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(drift_rows[0]))
        writer.writeheader()
        writer.writerows(drift_rows)
    with open(output_dir / "long_drift_summary.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {output_dir / 'long_drift_per_seed.csv'}")
    print(f"Wrote {output_dir / 'long_drift_summary.csv'}")


if __name__ == "__main__":
    main()
