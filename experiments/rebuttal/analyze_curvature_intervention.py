#!/usr/bin/env python3
"""Verify the paired curvature intervention before expensive training."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.curvature_intervention import (
    CurvatureTrajectoryDataset,
    curvature_profile,
    rectify_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    dataset = CurvatureTrajectoryDataset(args.data_path, intervention="curved")
    count = min(args.limit, len(dataset))
    curved_profiles = []
    rectified_profiles = []
    for index in range(count):
        source = dataset[index]["trajectory"]
        rectified = rectify_trajectory(source, dataset.selected_timesteps)
        if not torch.equal(source[0], rectified[0]):
            raise AssertionError(f"row {index}: initial noise endpoint changed")
        if not torch.equal(source[-2], rectified[-2]):
            raise AssertionError(f"row {index}: ODE target endpoint changed")
        if not torch.equal(source[-1], rectified[-1]):
            raise AssertionError(f"row {index}: clean conditioning latent changed")
        curved_profiles.append(curvature_profile(source, dataset.selected_timesteps))
        rectified_profiles.append(
            curvature_profile(rectified, dataset.selected_timesteps)
        )

    curved = np.asarray(curved_profiles, dtype=np.float64)
    rectified = np.asarray(rectified_profiles, dtype=np.float64)
    if not np.isfinite(curved).all() or not np.isfinite(rectified).all():
        raise ValueError("Curvature intervention produced non-finite metrics")
    if float(np.max(curved)) <= 0.0:
        raise AssertionError(
            "Source trajectories have zero measured curvature; there is no causal "
            "curvature intervention to test"
        )
    tolerance = max(1e-12, float(np.max(curved)) * 1e-10)
    if float(np.max(rectified)) > tolerance:
        raise AssertionError(
            "Rectified trajectories are not straight within numerical tolerance: "
            f"max={float(np.max(rectified))}, tolerance={tolerance}"
        )

    output = {
        "schema_version": 1,
        "num_paired_trajectories": count,
        "selected_timesteps": dataset.selected_timesteps,
        "use_ema": False,
        "controlled_invariants": [
            "prompt",
            "noise_seed",
            "initial_noise_state",
            "ode_target_endpoint",
            "clean_conditioning_latent",
            "selected_timesteps",
        ],
        "curved_mean_per_segment": curved.mean(axis=0).tolist(),
        "rectified_mean_per_segment": rectified.mean(axis=0).tolist(),
        "curved_mean": float(curved.mean()),
        "rectified_mean": float(rectified.mean()),
        "rectified_max": float(rectified.max()),
        "straightness_tolerance": tolerance,
    }
    if not all(
        math.isfinite(value)
        for value in (
            output["curved_mean"],
            output["rectified_mean"],
            output["rectified_max"],
        )
    ):
        raise ValueError("Non-finite curvature summary")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"PASS: paired intervention verified; wrote {output_path}")


if __name__ == "__main__":
    main()
