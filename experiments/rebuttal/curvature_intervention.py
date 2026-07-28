#!/usr/bin/env python3
"""Paired trajectory intervention used by the curvature causal experiment.

The source row contains five teacher ODE states followed by the clean dataset
latent.  The fifth ODE state is the regression endpoint; the clean latent is
kept only for the causal model's conditioning path.  Rectification changes
only the three interior ODE states while preserving the initial noise state,
the ODE endpoint, and the clean latent exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


MANIFEST_NAME = "curvature_dataset_manifest.json"


def normalize_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    """Return one trajectory as [points, frames, channels, height, width]."""
    if trajectory.ndim == 6 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if trajectory.ndim != 5:
        raise ValueError(
            "Expected trajectory shape [points,frames,channels,height,width], "
            f"got {tuple(trajectory.shape)}"
        )
    if trajectory.shape[0] < 4:
        raise ValueError("A curvature trajectory requires at least four points")
    return trajectory


def validate_selected_timesteps(selected_timesteps, num_points: int) -> list[float | None]:
    values = list(selected_timesteps)
    if len(values) != num_points:
        raise ValueError(
            f"selected_timesteps has {len(values)} entries for {num_points} points"
        )
    if values[-1] is not None:
        raise ValueError(
            "The final trajectory point is the clean dataset latent and must have timestep null"
        )
    ode_timesteps = values[:-1]
    if any(value is None for value in ode_timesteps):
        raise ValueError("Only the final clean-latent timestep may be null")
    numeric = [float(value) for value in ode_timesteps]
    if not all(left > right for left, right in zip(numeric, numeric[1:])):
        raise ValueError(
            f"ODE timesteps must be strictly descending, got {numeric}"
        )
    return [*numeric, None]


def rectify_trajectory(
    trajectory: torch.Tensor,
    selected_timesteps,
) -> torch.Tensor:
    """Linearize ODE states in time while preserving all controlled endpoints.

    The final point is a clean dataset latent and is not part of the teacher ODE
    path.  It remains bitwise unchanged.  The penultimate point is the ODE
    regression target and also remains unchanged.
    """
    trajectory = normalize_trajectory(trajectory)
    timesteps = validate_selected_timesteps(selected_timesteps, trajectory.shape[0])
    ode_times = torch.tensor(
        timesteps[:-1],
        device=trajectory.device,
        dtype=torch.float64,
    )
    high = ode_times[0]
    low = ode_times[-1]
    if not bool(high > low):
        raise ValueError(f"Invalid endpoint timesteps: high={high}, low={low}")

    weights = ((ode_times - low) / (high - low)).to(dtype=trajectory.dtype)
    weights = weights.reshape(-1, 1, 1, 1, 1)
    start = trajectory[0:1]
    endpoint = trajectory[-2:-1]

    rectified = trajectory.clone()
    rectified[:-1] = weights * start + (1.0 - weights) * endpoint
    # Explicit assignments make endpoint preservation robust to roundoff.
    rectified[0].copy_(trajectory[0])
    rectified[-2].copy_(trajectory[-2])
    rectified[-1].copy_(trajectory[-1])
    return rectified


def curvature_profile(trajectory: torch.Tensor, selected_timesteps) -> list[float]:
    """Compute Eq. (2)-style local-velocity deviation per ODE segment."""
    trajectory = normalize_trajectory(trajectory).double()
    timesteps = validate_selected_timesteps(selected_timesteps, trajectory.shape[0])
    points = trajectory[:-1].flatten(1)
    times = torch.tensor(timesteps[:-1], dtype=torch.float64)
    global_velocity = (points[-1] - points[0]) / (times[-1] - times[0])
    local_velocity = (points[1:] - points[:-1]) / (
        times[1:] - times[:-1]
    ).unsqueeze(1)
    deviation = (local_velocity - global_velocity.unsqueeze(0)).square().mean(dim=1)
    return [float(value.item()) for value in deviation]


class CurvatureTrajectoryDataset(Dataset):
    """Read the audited raw-trajectory LMDB and apply one paired intervention."""

    def __init__(self, data_path: str, intervention: str):
        if intervention not in {"curved", "rectified"}:
            raise ValueError("intervention must be 'curved' or 'rectified'")
        self.data_path = Path(data_path).resolve()
        manifest_path = self.data_path / MANIFEST_NAME
        data_file = self.data_path / "data.mdb"
        if not manifest_path.is_file() or not data_file.is_file():
            raise FileNotFoundError(
                f"Expected {manifest_path} and {data_file} for curvature training"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported curvature dataset schema: {self.manifest}")
        if self.manifest.get("use_ema") is not False:
            raise ValueError("Curvature causal training requires raw/no-EMA trajectories")
        self.shape = tuple(int(value) for value in self.manifest["latents_shape"])
        if len(self.shape) != 6 or self.shape[0] < 1:
            raise ValueError(f"Invalid stored trajectory shape: {self.shape}")
        self.selected_timesteps = validate_selected_timesteps(
            self.manifest["selected_timesteps"], self.shape[1]
        )
        self.intervention = intervention
        self.env = lmdb.open(
            str(self.data_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.env.begin() as transaction:
            latent_bytes = transaction.get(f"latents_{index}_data".encode())
            prompt_bytes = transaction.get(f"prompts_{index}_data".encode())
            noise_seed_bytes = transaction.get(f"noise_seed_{index}_data".encode())
        if latent_bytes is None or prompt_bytes is None or noise_seed_bytes is None:
            raise KeyError(f"Curvature LMDB is missing row {index}")
        trajectory = torch.from_numpy(
            np.frombuffer(latent_bytes, dtype=np.float16).copy().reshape(self.shape[1:])
        ).float()
        if self.intervention == "rectified":
            trajectory = rectify_trajectory(trajectory, self.selected_timesteps)
        return {
            "prompt": prompt_bytes.decode("utf-8"),
            "trajectory": trajectory,
            "noise_seed": int(noise_seed_bytes.decode("ascii")),
            "row_index": index,
        }
