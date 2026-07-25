#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def parse_set(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=TRAJECTORY_DIR")
    label, directory = value.split("=", 1)
    if not label or not directory:
        raise argparse.ArgumentTypeError("Use non-empty LABEL=TRAJECTORY_DIR")
    return label, Path(directory)


def load_trajectory(path, include_last_point):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = {}
    if isinstance(payload, dict) and payload.get("schema_version") == 2:
        prompt = payload["prompt"]
        trajectory = payload["trajectory"]
        metadata = {
            "noise_seed": payload.get("noise_seed"),
            "selected_timesteps": payload.get("selected_timesteps"),
            "guidance_scale": payload.get("guidance_scale"),
            "use_ema": payload.get("use_ema"),
        }
    elif isinstance(payload, dict) and len(payload) == 1:
        prompt, trajectory = next(iter(payload.items()))
    else:
        raise ValueError(f"{path}: unsupported trajectory payload")
    if trajectory.ndim == 6 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if trajectory.ndim != 5:
        raise ValueError(
            f"{path}: expected [steps,frames,channels,height,width], got {trajectory.shape}"
        )
    if not include_last_point:
        if trajectory.shape[0] < 4:
            raise ValueError(f"{path}: too few points to exclude the final ground-truth point")
        trajectory = trajectory[:-1]
    return prompt, trajectory.float(), metadata


def curvature_metrics(trajectory):
    points = trajectory.flatten(1)
    velocity = points[1:] - points[:-1]
    segment_norm = torch.linalg.vector_norm(velocity, dim=1)
    path_length = segment_norm.sum()
    chord_length = torch.linalg.vector_norm(points[-1] - points[0])
    eps = torch.finfo(points.dtype).eps
    path_excess = path_length / chord_length.clamp_min(eps) - 1.0

    if len(velocity) > 1:
        cosine = torch.nn.functional.cosine_similarity(velocity[:-1], velocity[1:], dim=1)
        turning_angle = torch.acos(cosine.clamp(-1.0, 1.0)).mean()
        normalized_second_difference = (
            torch.linalg.vector_norm(velocity[1:] - velocity[:-1], dim=1).sum()
            / path_length.clamp_min(eps)
        )
    else:
        turning_angle = torch.zeros(())
        normalized_second_difference = torch.zeros(())
    return {
        "path_excess_ratio": float(path_excess.item()),
        "mean_turning_angle_radians": float(turning_angle.item()),
        "normalized_second_difference": float(normalized_second_difference.item()),
        "path_length": float(path_length.item()),
        "chord_length": float(chord_length.item()),
    }


def bootstrap_ci(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    parser = argparse.ArgumentParser(
        description="Compare discrete flow-trajectory curvature across controlled teacher variants."
    )
    parser.add_argument("--trajectory_sets", nargs="+", type=parse_set, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--include_last_point",
        action="store_true",
        help="Include the final ground-truth clean latent appended by get_causal_ode_data_framewise.py.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument(
        "--reference_label",
        default="",
        help="Reference label for paired metric deltas. Defaults to the first trajectory set.",
    )
    parser.add_argument(
        "--allow_missing_pair_metadata",
        action="store_true",
        help=(
            "Allow legacy trajectories without seed/timestep/guidance/use_ema metadata. "
            "Do not use this for the controlled rebuttal comparison."
        ),
    )
    args = parser.parse_args()

    labels = [label for label, _ in args.trajectory_sets]
    if len(labels) != len(set(labels)):
        raise ValueError("Trajectory-set labels must be unique")
    reference_label = args.reference_label or labels[0]
    if reference_label not in labels:
        raise ValueError(f"Unknown --reference_label {reference_label!r}")

    paths_by_label = {}
    for label, directory in args.trajectory_sets:
        paths = sorted(directory.glob("*.pt"))
        if args.limit > 0:
            paths = paths[:args.limit]
        if not paths:
            raise FileNotFoundError(f"No .pt trajectories in {directory}")
        paths_by_label[label] = {path.name: path for path in paths}
    reference_names = set(paths_by_label[reference_label])
    for label, paths in paths_by_label.items():
        if set(paths) != reference_names:
            missing = sorted(reference_names - set(paths))[:8]
            extra = sorted(set(paths) - reference_names)[:8]
            raise ValueError(
                f"Unpaired trajectory files for {label}: missing={missing}, extra={extra}"
            )

    rows = []
    for filename in sorted(reference_names):
        paired = {}
        for label in labels:
            path = paths_by_label[label][filename]
            prompt, trajectory, metadata = load_trajectory(path, args.include_last_point)
            paired[label] = (prompt, trajectory, metadata, path)
        reference_prompt, reference_trajectory, reference_metadata, _ = paired[reference_label]
        for label, (prompt, trajectory, metadata, path) in paired.items():
            if prompt != reference_prompt:
                raise ValueError(
                    f"{filename}: prompt mismatch for {label} versus {reference_label}"
                )
            if trajectory.shape != reference_trajectory.shape:
                raise ValueError(
                    f"{filename}: trajectory shape mismatch for {label}: "
                    f"{trajectory.shape} versus {reference_trajectory.shape}"
                )
            for field in (
                "noise_seed",
                "selected_timesteps",
                "guidance_scale",
                "use_ema",
            ):
                reference_value = reference_metadata.get(field)
                value = metadata.get(field)
                if (
                    not args.allow_missing_pair_metadata
                    and (reference_value is None or value is None)
                ):
                    raise ValueError(
                        f"{filename}: missing controlled-pair metadata {field!r} "
                        f"for {label} or {reference_label}"
                    )
                if reference_value != value:
                    raise ValueError(
                        f"{filename}: {field} mismatch for {label}: "
                        f"{value!r} versus {reference_value!r}"
                    )
            metrics = curvature_metrics(trajectory)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"{path}: non-finite curvature metric")
            rows.append(
                {
                    "label": label,
                    "file": str(path.resolve()),
                    "pair_id": filename,
                    "prompt": prompt,
                    "num_points": int(trajectory.shape[0]),
                    **metrics,
                }
            )

    metric_names = [
        "path_excess_ratio",
        "mean_turning_angle_radians",
        "normalized_second_difference",
    ]
    summaries = []
    for label, _ in args.trajectory_sets:
        label_rows = [row for row in rows if row["label"] == label]
        for metric in metric_names:
            values = [row[metric] for row in label_rows]
            summaries.append(
                {
                    "label": label,
                    "metric": metric,
                    "num_trajectories": len(values),
                    "mean": float(np.mean(values)),
                    "sample_std": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    ),
                    "bootstrap_95ci": bootstrap_ci(
                        values,
                        samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed,
                    ),
                }
            )

    row_lookup = {
        (row["label"], row["pair_id"]): row
        for row in rows
    }
    paired_delta_summaries = []
    for label in labels:
        if label == reference_label:
            continue
        for metric in metric_names:
            deltas = [
                row_lookup[(label, pair_id)][metric]
                - row_lookup[(reference_label, pair_id)][metric]
                for pair_id in sorted(reference_names)
            ]
            paired_delta_summaries.append(
                {
                    "label": label,
                    "reference_label": reference_label,
                    "metric": metric,
                    "num_pairs": len(deltas),
                    "mean_paired_delta": float(np.mean(deltas)),
                    "sample_std": (
                        float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None
                    ),
                    "bootstrap_95ci": bootstrap_ci(
                        deltas,
                        samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed,
                    ),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "curvature_per_trajectory.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "curvature_summary.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "include_last_point": args.include_last_point,
                "reference_label": reference_label,
                "summaries": summaries,
                "paired_delta_summaries": paired_delta_summaries,
            },
            fp,
            indent=2,
            sort_keys=True,
        )
        fp.write("\n")
    print(f"Wrote {output_dir / 'curvature_per_trajectory.csv'}")
    print(f"Wrote {output_dir / 'curvature_summary.json'}")


if __name__ == "__main__":
    main()
