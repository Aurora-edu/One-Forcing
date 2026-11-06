"""Reparameterization checks on fixed physical trajectories, one file at a time.

Equation 3 is a finite-difference velocity-deviation proxy, not geometric
curvature. Integrals use each coordinate's nonuniform interval widths. A segment
crossing the physical high-noise boundary is clipped in the chosen coordinate.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.iclr.protocol import atomic_json


def _dot(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    total = 0.0
    for start in range(0, a.numel(), 65536):
        left = a[start:start + 65536].double()
        right = b[start:start + 65536].double()
        total += float(torch.dot(left, right))
    return total


def _norm(a):
    return math.sqrt(max(0.0, _dot(a, a)))


def validate_points(points, physical_t):
    times = np.asarray(physical_t, dtype=np.float64)
    if points.ndim < 2 or len(points) != len(times) or len(times) < 2:
        raise ValueError("expected matching [points,...] and physical timestep vector")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("physical noise time must increase strictly")
    if abs(times[0]) > 1e-7 or abs(times[-1] - 1) > 1e-7:
        raise ValueError("full trajectory endpoints must be physical t=0 and t=1")
    # Chunked check; no full float64 copy of video trajectories.
    flat = points.reshape(-1)
    for start in range(0, flat.numel(), 65536):
        if not bool(torch.isfinite(flat[start:start + 65536]).all()):
            raise ValueError("nonfinite trajectory")
    return times


def geometry(points, physical_t, high_noise=0.9):
    if points.dtype not in (torch.float32, torch.float64):
        points = points.float()
    times = validate_points(points, physical_t)
    chord = points[-1] - points[0]
    chord_squared = _dot(chord, chord)
    chord_length = math.sqrt(chord_squared)
    lengths = [_norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]
    path_length = sum(lengths)
    turns = []
    previous = None
    previous_length = None
    previous_index = None
    for index, length in enumerate(lengths):
        if length == 0:
            continue
        segment = points[index + 1] - points[index]
        if previous is not None:
            cosine = max(-1.0, min(1.0, _dot(previous, segment) / (previous_length * length)))
            # Roundoff on a perfectly straight line should not manufacture a turn.
            angle = 0.0 if cosine > 1 - 1e-12 else math.acos(cosine)
            turns.append((index, angle, previous_index != index - 1))
        previous, previous_length, previous_index = segment, length, index
    orthogonal_distances = []
    for point in points:
        offset = point - points[0]
        projection = _dot(offset, chord) / chord_squared if chord_squared else 0.0
        # Subtract explicitly in chunks to avoid cancellation in ||x||²-<x,c>²/||c||².
        squared = 0.0
        flat, chord_flat = offset.reshape(-1), chord.reshape(-1)
        for start in range(0, flat.numel(), 65536):
            residual = flat[start:start + 65536].double() - projection * chord_flat[start:start + 65536].double()
            squared += float(torch.dot(residual, residual))
        orthogonal_distances.append(math.sqrt(max(squared, 0.0)))
    total_turn = sum(angle for _, angle, _ in turns)
    high_turn = sum(angle for index, angle, _ in turns if times[index] >= high_noise)
    return {
        "path_length": path_length, "chord_length": chord_length,
        "path_to_chord_ratio": path_length / chord_length if chord_length else None,
        "max_distance_to_chord": max(orthogonal_distances),
        "max_normalized_distance_to_chord": max(orthogonal_distances) / chord_length if chord_length else None,
        "total_turn_radians": total_turn,
        "mean_turn_radians": total_turn / len(turns) if turns else 0.0,
        "high_noise_turn_fraction": high_turn / total_turn if total_turn else None,
        "turns_bridging_repeated_points": sum(int(gap) for _, _, gap in turns),
        "zero_length_segments": sum(length == 0 for length in lengths),
        "stationary": path_length == 0,
    }, np.asarray(lengths)


def velocity_proxy(points, physical_t, *, parameterization="flow", power=2.0, high_noise=0.9):
    if points.dtype not in (torch.float32, torch.float64):
        points = points.float()
    times = validate_points(points, physical_t)
    if not 0 <= high_noise <= 1 or power <= 0 or not math.isfinite(power):
        raise ValueError("invalid threshold or power")
    if parameterization == "flow":
        coordinate = times.copy()
        mapped_threshold = high_noise
    elif parameterization == "power":
        coordinate = times ** power
        mapped_threshold = high_noise ** power
    elif parameterization == "arclength":
        lengths = np.asarray([_norm(points[i + 1] - points[i]) for i in range(len(points) - 1)])
        if lengths.sum() == 0:
            return {"parameterization": parameterization, "defined": False,
                    "reason": "stationary path has no arc-length clock", "segments": []}
        coordinate = np.concatenate(([0.0], lengths.cumsum() / lengths.sum()))
        mapped_threshold = float(np.interp(high_noise, times, coordinate))
    else:
        raise ValueError(f"unknown parameterization: {parameterization}")
    chord = points[-1] - points[0]
    d = chord.numel()
    integral_u = high_u = integral_t = high_t = 0.0
    segments = []
    for index in range(len(points) - 1):
        dt = float(times[index + 1] - times[index])
        du = float(coordinate[index + 1] - coordinate[index])
        if du == 0:
            # Only stationary plateaus under arc length. No eps-generated velocity.
            segments.append({"index": index, "physical_low": float(times[index]),
                "physical_high": float(times[index + 1]), "du": 0.0, "proxy": None,
                "high_noise_du": 0.0})
            continue
        delta = points[index + 1] - points[index]
        value = 0.0
        delta_flat, chord_flat = delta.reshape(-1), chord.reshape(-1)
        for start in range(0, d, 65536):
            difference = (delta_flat[start:start + 65536].double() / du
                          - chord_flat[start:start + 65536].double())
            value += float(torch.dot(difference, difference)) / d
        high_dt = max(0.0, float(times[index + 1] - max(times[index], high_noise)))
        high_du = max(0.0, float(coordinate[index + 1] - max(coordinate[index], mapped_threshold)))
        integral_u += value * du
        high_u += value * high_du
        integral_t += value * dt
        high_t += value * high_dt
        segments.append({"index": index, "physical_low": float(times[index]),
            "physical_high": float(times[index + 1]), "du": du, "proxy": value,
            "high_noise_du": high_du})
    return {"parameterization": parameterization, "defined": True,
        "velocity_deviation_integral_du": integral_u,
        "high_noise_integral_du": high_u,
        "high_noise_fraction_du": high_u / integral_u if integral_u else None,
        "velocity_deviation_integral_dt": integral_t,
        "high_noise_fraction_dt": high_t / integral_t if integral_t else None,
        "physical_high_noise_threshold": high_noise,
        "coordinate_at_threshold": mapped_threshold,
        "zero_width_segments": sum(s["du"] == 0 for s in segments),
        "segments": segments}


def load_physical_trajectory(path, time_scale=1000.0):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != 2:
        raise ValueError("schema_version=2 and actual physical selected_timesteps are required")
    points = payload["trajectory"]
    if points.ndim == 6 and points.shape[0] == 1:
        points = points[0]
    times = list(payload["selected_timesteps"])
    if len(times) != len(points):
        raise ValueError("timesteps and trajectory point counts disagree")
    # A trailing null is the DATASET clean latent, never an ODE endpoint.
    if times and times[-1] is None:
        times, points = times[:-1], points[:-1]
    if any(t is None for t in times) or time_scale <= 0:
        raise ValueError("invalid physical times")
    times = np.asarray(times, dtype=np.float64) / time_scale
    if np.all(np.diff(times) < 0):
        points = points.flip(0)
        times = times[::-1].copy()
    validate_points(points, times)
    return points.float(), times, {key: payload.get(key) for key in
        ("prompt", "noise_seed", "guidance_scale", "generator_ckpt", "use_ema",
         "trajectory_kind", "clean_conditioning", "teacher_tensor_sha256",
         "solver", "shift", "negative_prompt", "num_solver_steps")}


def analyze(points, times, threshold=0.9, power=2.0):
    geometric, _ = geometry(points, times, threshold)
    return {"geometry": geometric, "parameterizations": [velocity_proxy(
        points, times, parameterization=kind, power=power, high_noise=threshold)
        for kind in ("flow", "power", "arclength")]}


def bootstrap(values, samples=2000):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return {"n": 0, "mean": None, "std": None, "bootstrap_95ci": None}
    rng = np.random.default_rng(0)
    means = [float(rng.choice(values, size=len(values)).mean()) for _ in range(samples)]
    return {"n": len(values), "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist() if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/trajectory_analysis")
    parser.add_argument("--physical-time-scale", type=float, default=1000.0)
    parser.add_argument("--high-noise-threshold", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-check", action="store_true",
                        help="test fixtures only; requires trajectory_kind=synthetic_test and labels all results")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"input": args.trajectories, "output": args.output,
            "physical_region": f"t >= {args.high_noise_threshold}",
            "parameterizations": ["flow", f"t^{args.power}", "arclength"],
            "trailing_null": "exclude dataset clean latent", "gpu_execution": False}, indent=2))
        return
    paths = sorted(Path(args.trajectories).glob("*.pt"))
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError("no trajectory files")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    identity = None
    analysis_kind = "synthetic_validation" if args.synthetic_check else "teacher_analysis"
    with (root / "per_trajectory.jsonl").open("w") as stream:
        for path in paths:
            points, times, metadata = load_physical_trajectory(path, args.physical_time_scale)
            if args.synthetic_check:
                if metadata.get("trajectory_kind") != "synthetic_test":
                    raise ValueError("--synthetic-check requires explicitly labeled synthetic_test fixtures")
            else:
                if (metadata.get("trajectory_kind") != "bidirectional_teacher_full"
                        or metadata.get("clean_conditioning") is not False
                        or not metadata.get("teacher_tensor_sha256")):
                    raise ValueError("geometry conclusions require full bidirectional teacher paths from collect_teacher; sparse causal CD data and synthetic fixtures are unsuitable")
                if metadata.get("num_solver_steps") != len(points) - 1:
                    raise ValueError("trajectory does not contain every declared solver state")
                current_identity = tuple(metadata.get(k) for k in ("teacher_tensor_sha256",
                    "guidance_scale", "solver", "shift", "negative_prompt", "num_solver_steps"))
                if identity is None:
                    identity = current_identity
                if current_identity != identity:
                    raise ValueError("mixed teacher weights or solver/guidance protocols; analyze separately")
            if points.ndim != 5 or points.shape[1] < 2:
                raise ValueError("expected [points,frames,channels,height,width], at least 2 frames")
            for representation in ("latent", "temporal_difference"):
                current = points if representation == "latent" else points[:, 1:] - points[:, :-1]
                result = analyze(current, times, args.high_noise_threshold, args.power)
                stream.write(json.dumps({"file": str(path), "analysis_kind": analysis_kind, "representation": representation,
                    "metadata": metadata, **result}, allow_nan=False) + "\n")
                for parameter in result["parameterizations"]:
                    rows.append({"file": path.name, "analysis_kind": analysis_kind, "representation": representation,
                        "parameterization": parameter["parameterization"],
                        "high_noise_fraction_du": parameter.get("high_noise_fraction_du"),
                        "velocity_deviation_integral_du": parameter.get("velocity_deviation_integral_du"),
                        **result["geometry"]})
                del current
            del points
    with (root / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for representation in ("latent", "temporal_difference"):
        for parameter in ("flow", "power", "arclength"):
            subset = [row for row in rows if row["representation"] == representation
                      and row["parameterization"] == parameter]
            summaries.append({"representation": representation, "parameterization": parameter,
                "high_noise_fraction_du": bootstrap([row["high_noise_fraction_du"] for row in subset]),
                "path_to_chord_ratio": bootstrap([row["path_to_chord_ratio"] for row in subset]),
                "total_turn_radians": bootstrap([row["total_turn_radians"] for row in subset])})
    atomic_json(root / "summary.json", {"physical_threshold": args.high_noise_threshold,
        "analysis_kind": analysis_kind, "synthetic_is_paper_result": False,
        "coordinate_power": args.power, "summaries": summaries,
        "estimator": "fixed-vertex finite differences; piecewise constant C in u, clipped at g(physical threshold)",
        "uncertainty_unit": "trajectory, not training seed",
        "warning": "sparse trajectories cannot establish high-noise curvature or causality; repeat on denser teacher trajectories"})
    print(f"Wrote {root}; {len(paths)} trajectories, no GPU execution")


if __name__ == "__main__":
    main()
