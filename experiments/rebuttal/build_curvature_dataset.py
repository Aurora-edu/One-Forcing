#!/usr/bin/env python3
"""Build one audited LMDB shared by both curvature-intervention arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.curvature_intervention import (
    MANIFEST_NAME,
    normalize_trajectory,
    validate_selected_timesteps,
)


def atomic_write_json(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_source(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError(f"{path}: expected schema_version=2 trajectory payload")
    if payload.get("use_ema") is not False:
        raise ValueError(f"{path}: curvature dataset must use raw/no-EMA weights")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path}: missing non-empty prompt")
    trajectory = normalize_trajectory(payload["trajectory"])
    timesteps = validate_selected_timesteps(
        payload.get("selected_timesteps"), trajectory.shape[0]
    )
    noise_seed = payload.get("noise_seed")
    if not isinstance(noise_seed, int) or noise_seed < 0:
        raise ValueError(f"{path}: invalid noise_seed={noise_seed!r}")
    guidance_scale = float(payload.get("guidance_scale"))
    checkpoint = payload.get("generator_ckpt")
    if not checkpoint:
        raise ValueError(f"{path}: missing generator_ckpt provenance")
    return prompt, trajectory, timesteps, noise_seed, guidance_scale, str(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument("--output_lmdb", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--map_size_gib",
        type=float,
        default=0.0,
        help="LMDB map size. Zero estimates it from source file sizes.",
    )
    args = parser.parse_args()
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be -1 or positive")
    if args.map_size_gib < 0:
        raise ValueError("--map_size_gib must be non-negative")

    trajectory_dir = Path(args.trajectory_dir).resolve()
    files = sorted(trajectory_dir.glob("*.pt"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No trajectory .pt files in {trajectory_dir}")

    output_lmdb = Path(args.output_lmdb).resolve()
    if output_lmdb.exists():
        if not output_lmdb.is_dir():
            raise FileExistsError(f"Curvature output is not a directory: {output_lmdb}")
        if any(output_lmdb.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty curvature dataset: {output_lmdb}"
            )
    output_lmdb.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(path.stat().st_size for path in files)
    estimated = max(1 << 30, int(source_bytes * 1.25))
    map_size = (
        int(args.map_size_gib * (1 << 30)) if args.map_size_gib else estimated
    )
    environment = lmdb.open(str(output_lmdb), map_size=map_size, subdir=True)

    common_shape = None
    common_timesteps = None
    common_guidance = None
    common_checkpoint = None
    seed_digest = hashlib.sha256()
    prompt_digest = hashlib.sha256()
    try:
        for index, path in enumerate(files):
            (
                prompt,
                trajectory,
                timesteps,
                noise_seed,
                guidance_scale,
                checkpoint,
            ) = load_source(path)
            shape = tuple(trajectory.shape)
            if common_shape is None:
                common_shape = shape
                common_timesteps = timesteps
                common_guidance = guidance_scale
                common_checkpoint = checkpoint
            checks = {
                "shape": (shape, common_shape),
                "selected_timesteps": (timesteps, common_timesteps),
                "guidance_scale": (guidance_scale, common_guidance),
                "generator_ckpt": (checkpoint, common_checkpoint),
            }
            mismatches = {
                key: values for key, values in checks.items() if values[0] != values[1]
            }
            if mismatches:
                raise ValueError(f"{path}: uncontrolled trajectory mismatch: {mismatches}")

            latent = trajectory.to(dtype=torch.float16).contiguous().numpy()
            with environment.begin(write=True) as transaction:
                transaction.put(f"latents_{index}_data".encode(), latent.tobytes())
                transaction.put(f"prompts_{index}_data".encode(), prompt.encode("utf-8"))
                transaction.put(
                    f"noise_seed_{index}_data".encode(), str(noise_seed).encode("ascii")
                )
            seed_digest.update(f"{index}:{noise_seed}\n".encode("ascii"))
            prompt_digest.update(prompt.encode("utf-8"))
            prompt_digest.update(b"\n")
            if (index + 1) % 100 == 0 or index + 1 == len(files):
                print(f"Stored curvature trajectories {index + 1}/{len(files)}", flush=True)

        stored_shape = (len(files), *common_shape)
        with environment.begin(write=True) as transaction:
            transaction.put(
                b"latents_shape",
                " ".join(str(value) for value in stored_shape).encode("ascii"),
            )
            transaction.put(b"prompts_shape", f"{len(files)}".encode("ascii"))
        environment.sync()
    finally:
        environment.close()

    manifest = {
        "schema_version": 1,
        "trajectory_schema_version": 2,
        "num_trajectories": len(files),
        "latents_shape": list(stored_shape),
        "selected_timesteps": common_timesteps,
        "guidance_scale": common_guidance,
        "generator_ckpt": common_checkpoint,
        "use_ema": False,
        "weight_source": "generator",
        "noise_seed_sequence_sha256": seed_digest.hexdigest(),
        "prompt_sequence_sha256": prompt_digest.hexdigest(),
        "source_directory": str(trajectory_dir),
        "source_file_first": files[0].name,
        "source_file_last": files[-1].name,
    }
    atomic_write_json(output_lmdb / MANIFEST_NAME, manifest)
    print(f"Wrote {output_lmdb / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
