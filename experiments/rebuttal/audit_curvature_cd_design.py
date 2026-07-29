#!/usr/bin/env python3
"""Fail-closed audit for the small adjacent-state curvature CD experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.curvature_intervention import (  # noqa: E402
    CurvatureTrajectoryDataset,
    rectify_trajectory,
)
from utils.config import load_config  # noqa: E402
from utils.scheduler import FlowMatchScheduler  # noqa: E402


def resolve_eval_timesteps(config_path: Path) -> tuple[list[int], list[float]]:
    config = load_config(str(config_path))
    if not bool(config.warp_denoising_step):
        raise ValueError(f"{config_path}: curvature evaluation requires timestep warping")
    scheduler = FlowMatchScheduler(
        shift=float(config.model_kwargs.timestep_shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0.0])))
    pseudo = [int(value) for value in config.denoising_step_list]
    if any(value < 0 or value > 1000 for value in pseudo):
        raise ValueError(f"{config_path}: pseudo timesteps must be in [0, 1000]")
    resolved = [float(timesteps[1000 - value].item()) for value in pseudo]
    return pseudo, resolved


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--eval_all1_config", required=True)
    parser.add_argument("--eval_all4_config", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    train_config = load_config(str(Path(args.train_config).resolve()))
    if int(train_config.num_frame_per_block) != 1:
        raise ValueError("Curvature CD training must be framewise")
    if int(train_config.max_steps) % 4 != 0:
        raise ValueError("The fixed budget must balance all four adjacent pairs")
    if int(train_config.training_num_frames) < 1:
        raise ValueError("training_num_frames must be positive")

    dataset = CurvatureTrajectoryDataset(args.data_path, intervention="curved")
    if dataset.manifest.get("use_ema") is not False:
        raise ValueError("The source trajectories are not audited raw/no-EMA")
    ode_timesteps = [float(value) for value in dataset.selected_timesteps[:-1]]
    if len(ode_timesteps) != 5 or ode_timesteps[-1] != 0.0:
        raise ValueError(f"Expected five ODE states ending at zero: {ode_timesteps}")
    if int(train_config.training_num_frames) > dataset.shape[2]:
        raise ValueError("training_num_frames exceeds stored trajectory length")

    point_mse = []
    count = min(args.limit, len(dataset))
    for index in range(count):
        source = dataset[index]["trajectory"]
        rectified = rectify_trajectory(source, dataset.selected_timesteps)
        point_mse.append(
            (source.float() - rectified.float()).square().flatten(1).mean(1).tolist()
        )
    point_mse = np.asarray(point_mse, dtype=np.float64)
    mean_point_mse = point_mse.mean(axis=0)
    if not np.isfinite(point_mse).all():
        raise ValueError("Non-finite intervention magnitude")
    if mean_point_mse[0] != 0.0 or mean_point_mse[-2] != 0.0 or mean_point_mse[-1] != 0.0:
        raise AssertionError("Controlled endpoints or clean conditioning changed")
    if not np.all(mean_point_mse[1:-2] > 0.0):
        raise AssertionError("Every interior ODE state must be changed by rectification")

    all1_pseudo, all1_resolved = resolve_eval_timesteps(
        Path(args.eval_all1_config).resolve()
    )
    all4_pseudo, all4_resolved = resolve_eval_timesteps(
        Path(args.eval_all4_config).resolve()
    )
    tolerance = 1e-4
    if len(all1_resolved) != 1 or abs(all1_resolved[0] - ode_timesteps[0]) > tolerance:
        raise AssertionError(
            f"all1 does not start at the trained high-noise state: {all1_resolved}"
        )
    if len(all4_resolved) != 4 or any(
        abs(actual - expected) > tolerance
        for actual, expected in zip(all4_resolved, ode_timesteps[:-1])
    ):
        raise AssertionError(
            "all4 inference timesteps do not match the four trained inputs: "
            f"resolved={all4_resolved}, trained={ode_timesteps[:-1]}"
        )

    target_exposure = []
    for pair_index in range(len(ode_timesteps) - 1):
        boundary = ode_timesteps[pair_index + 1] == 0.0
        low_state_changed = bool(mean_point_mse[pair_index + 1] > 0.0)
        high_state_changed = bool(mean_point_mse[pair_index] > 0.0)
        exposed = low_state_changed if not boundary else high_state_changed
        if not exposed:
            raise AssertionError(f"Pair {pair_index} is not exposed to the intervention")
        target_exposure.append(
            {
                "pair_index": pair_index,
                "high_timestep": ode_timesteps[pair_index],
                "low_timestep": ode_timesteps[pair_index + 1],
                "target_kind": "boundary_identity" if boundary else "ema_bootstrap",
                "high_state_changed": high_state_changed,
                "low_state_changed": low_state_changed,
                "intervention_exposed": exposed,
            }
        )

    num_pairs = len(target_exposure)
    max_steps = int(train_config.max_steps)
    output = {
        "schema_version": 1,
        "status": "pass",
        "num_rows_audited": count,
        "source_use_ema": False,
        "evaluation_use_ema": False,
        "framewise": True,
        "training_num_frames": int(train_config.training_num_frames),
        "max_steps_per_arm": max_steps,
        "num_adjacent_pairs": num_pairs,
        "updates_per_pair": max_steps // num_pairs,
        "pair_schedule": "deterministic_low_to_high_sweep",
        "ode_timesteps": ode_timesteps,
        "mean_intervention_mse_per_stored_point": mean_point_mse.tolist(),
        "pair_intervention_exposure": target_exposure,
        "all1_pseudo_timesteps": all1_pseudo,
        "all1_resolved_timesteps": all1_resolved,
        "all4_pseudo_timesteps": all4_pseudo,
        "all4_resolved_timesteps": all4_resolved,
        "primary_comparison": "rectified_all1 - curved_all1",
        "negative_control": "rectified_all4 - curved_all4",
        "difference_in_differences": (
            "(rectified_all1-curved_all1)-(rectified_all4-curved_all4)"
        ),
    }
    output_path = Path(args.output_path).resolve()
    atomic_write_json(output_path, output)
    print(f"PASS: adjacent-state curvature CD design audited: {output_path}")


if __name__ == "__main__":
    main()
