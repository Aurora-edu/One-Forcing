#!/usr/bin/env python3
"""Estimate a formal run after step 10 while respecting its update cadence."""

import argparse
import json
import math
import statistics
from pathlib import Path

from omegaconf import OmegaConf


def estimate(run_dir: Path, minimum_step: int):
    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} must contain resolved_config.yaml and metrics.jsonl"
        )
    config = OmegaConf.load(config_path)
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records or int(records[-1]["step"]) < minimum_step:
        last_step = int(records[-1]["step"]) if records else 0
        raise ValueError(
            f"Run has reached step {last_step}; wait until at least {minimum_step}"
        )
    timed = [
        record
        for record in records
        if "per_iteration_time" in record
        and math.isfinite(float(record["per_iteration_time"]))
    ]
    generator_times = [
        float(record["per_iteration_time"])
        for record in timed
        if "generator_loss" in record
    ]
    critic_times = [
        float(record["per_iteration_time"])
        for record in timed
        if "generator_loss" not in record
    ]
    if not generator_times or not critic_times:
        raise ValueError(
            "Need at least one timed generator update and one timed critic update"
        )
    update_ratio = int(config.dfake_gen_update_ratio)
    if update_ratio < 1:
        raise ValueError("dfake_gen_update_ratio must be positive")
    generator_mean = statistics.fmean(generator_times)
    critic_mean = statistics.fmean(critic_times)
    cadence_mean = (
        generator_mean + (update_ratio - 1) * critic_mean
    ) / update_ratio
    target_step = int(config.max_steps)
    current_step = int(records[-1]["step"])
    return {
        "run_dir": str(run_dir.resolve()),
        "current_step": current_step,
        "target_step": target_step,
        "update_ratio": update_ratio,
        "timed_generator_updates": len(generator_times),
        "timed_critic_updates": len(critic_times),
        "mean_generator_iteration_seconds": generator_mean,
        "mean_critic_iteration_seconds": critic_mean,
        "cadence_weighted_seconds_per_step": cadence_mean,
        "projected_compute_hours_total": cadence_mean * target_step / 3600.0,
        "projected_compute_hours_remaining": (
            cadence_mean * max(target_step - current_step, 0) / 3600.0
        ),
        "scope": (
            "Model-compute estimate from observed update cadence; checkpoint I/O "
            "and evaluation are not included."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--minimum_step", type=int, default=10)
    args = parser.parse_args()
    if args.minimum_step < 2:
        raise ValueError("--minimum_step must be at least 2")
    payload = estimate(Path(args.run_dir), args.minimum_step)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
