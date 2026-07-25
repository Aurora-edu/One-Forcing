#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import linalg


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_videos(path, limit):
    videos = sorted(item for item in Path(path).iterdir() if item.suffix.lower() in VIDEO_SUFFIXES)
    if limit > 0:
        videos = videos[:limit]
    return videos


def manifest_video_names(path):
    names = []
    with open(path, encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "output_name" not in record:
                raise ValueError(f"{path}:{line_number}: missing output_name")
            name = str(record["output_name"])
            if Path(name).name != name:
                raise ValueError(f"{path}:{line_number}: output_name must be a plain filename")
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate output_name values")
    return names


def validate_manifest_paths(paths, manifest_path, label):
    if not manifest_path:
        return
    expected = set(manifest_video_names(manifest_path))
    actual = {path.name for path in paths}
    if actual != expected:
        missing = sorted(expected - actual)[:8]
        extra = sorted(actual - expected)[:8]
        raise ValueError(
            f"{label} videos do not exactly match {manifest_path}: "
            f"missing={missing}, extra={extra}"
        )


def load_video(path, num_frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        capture.release()
        raise RuntimeError(f"No frames in {path}")
    indices = np.linspace(0, frame_count - 1, num=num_frames)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(index)))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed reading frame {index} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    array = np.stack(frames)
    return torch.from_numpy(array).permute(3, 0, 1, 2).float()


def extract_features(paths, model, device, num_frames, batch_size):
    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        videos = torch.stack([load_video(path, num_frames) for path in batch_paths]).to(device)
        with torch.no_grad():
            batch_features = model(
                videos,
                rescale=True,
                resize=True,
                return_features=True,
            )
        features.append(batch_features.float().cpu().numpy())
        print(f"Features {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(features, axis=0).astype(np.float64)


def frechet_distance(real_features, fake_features):
    real_mean = real_features.mean(axis=0)
    fake_mean = fake_features.mean(axis=0)
    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)
    covariance_mean, _ = linalg.sqrtm(real_cov @ fake_cov, disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(real_cov.shape[0]) * 1e-6
        covariance_mean = linalg.sqrtm((real_cov + offset) @ (fake_cov + offset))
    if np.iscomplexobj(covariance_mean):
        max_imaginary = np.max(np.abs(covariance_mean.imag))
        if max_imaginary > 1e-3:
            raise ValueError(f"FVD covariance sqrt has imaginary component {max_imaginary}")
        covariance_mean = covariance_mean.real
    mean_term = np.sum((real_mean - fake_mean) ** 2)
    trace_term = np.trace(real_cov + fake_cov - 2.0 * covariance_mean)
    return float(np.real(mean_term + trace_term))


def pairwise_squared_distances(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError(
            f"Expected two [samples, features] arrays, got {left.shape} and {right.shape}"
        )
    distances = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def kth_neighbor_radii(features, nearest_k):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected [samples, features], got {features.shape}")
    if nearest_k < 1 or nearest_k >= len(features):
        raise ValueError(
            f"nearest_k must be in [1, num_samples-1], got k={nearest_k}, "
            f"num_samples={len(features)}"
        )
    distances = pairwise_squared_distances(features, features)
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, nearest_k - 1, axis=1)[:, nearest_k - 1]


def manifold_metrics(real_features, fake_features, nearest_k=5):
    """Compute I3D feature precision/recall/density/coverage.

    Precision and recall use k-NN manifolds from the real and generated
    distributions. Density and coverage follow the improved precision/recall
    formulation and are reported because the meta-review explicitly requests
    recall and coverage in addition to FVD.
    """

    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    real_radii = kth_neighbor_radii(real_features, nearest_k)
    fake_radii = kth_neighbor_radii(fake_features, nearest_k)
    fake_to_real = pairwise_squared_distances(fake_features, real_features)

    fake_in_real = fake_to_real <= real_radii[None, :]
    real_in_fake = fake_to_real <= fake_radii[:, None]
    precision = np.mean(np.any(fake_in_real, axis=1))
    recall = np.mean(np.any(real_in_fake, axis=0))
    density = np.mean(np.sum(fake_in_real, axis=1) / float(nearest_k))
    coverage = np.mean(np.min(fake_to_real, axis=0) <= real_radii)
    return {
        "nearest_k": int(nearest_k),
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }


def bootstrap_fvd(real_features, fake_features, samples, seed):
    if samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        real_index = rng.integers(0, len(real_features), size=len(real_features))
        fake_index = rng.integers(0, len(fake_features), size=len(fake_features))
        values.append(
            frechet_distance(real_features[real_index], fake_features[fake_index])
        )
    return {
        "samples": samples,
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute I3D Fréchet Video Distance on decoded real and generated videos."
    )
    parser.add_argument("--real_videos_dir", required=True)
    parser.add_argument("--fake_videos_dir", required=True)
    parser.add_argument("--real_manifest_path", default="")
    parser.add_argument("--fake_manifest_path", default="")
    parser.add_argument("--i3d_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_videos", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_samples", type=int, default=0)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--allow_unequal_counts", action="store_true")
    parser.add_argument(
        "--nearest_k",
        type=int,
        default=5,
        help="k for I3D feature precision/recall/density/coverage.",
    )
    args = parser.parse_args()

    real_paths = list_videos(args.real_videos_dir, args.limit)
    fake_paths = list_videos(args.fake_videos_dir, args.limit)
    validate_manifest_paths(real_paths, args.real_manifest_path, "real")
    validate_manifest_paths(fake_paths, args.fake_manifest_path, "fake")
    if len(real_paths) < args.min_videos or len(fake_paths) < args.min_videos:
        raise ValueError(
            f"Need at least {args.min_videos} videos per set; "
            f"found real={len(real_paths)}, fake={len(fake_paths)}"
        )
    if not args.allow_unequal_counts and len(real_paths) != len(fake_paths):
        raise ValueError(
            f"Matched FVD requires equal real/fake counts; got "
            f"real={len(real_paths)}, fake={len(fake_paths)}"
        )
    if args.num_frames < 16 or args.batch_size < 1:
        raise ValueError(
            "The bundled I3D model requires num_frames >=16; "
            "batch_size must be positive"
        )
    if args.nearest_k < 1:
        raise ValueError("--nearest_k must be positive")
    if min(len(real_paths), len(fake_paths)) <= args.nearest_k:
        raise ValueError(
            f"--nearest_k={args.nearest_k} requires at least "
            f"{args.nearest_k + 1} videos in both sets"
        )

    device = torch.device(args.device)
    model = torch.jit.load(args.i3d_path, map_location=device).eval()
    real_features = extract_features(
        real_paths, model, device, args.num_frames, args.batch_size
    )
    fake_features = extract_features(
        fake_paths, model, device, args.num_frames, args.batch_size
    )
    score = frechet_distance(real_features, fake_features)
    distribution_metrics = manifold_metrics(
        real_features,
        fake_features,
        nearest_k=args.nearest_k,
    )
    result = {
        "schema_version": 1,
        "metric": "FVD-I3D",
        "fvd": score,
        "num_real_videos": len(real_paths),
        "num_fake_videos": len(fake_paths),
        "num_sampled_frames": args.num_frames,
        "feature_dimension": int(real_features.shape[1]),
        "i3d_path": str(Path(args.i3d_path).resolve()),
        "real_manifest_path": (
            str(Path(args.real_manifest_path).resolve())
            if args.real_manifest_path
            else None
        ),
        "fake_manifest_path": (
            str(Path(args.fake_manifest_path).resolve())
            if args.fake_manifest_path
            else None
        ),
        "feature_distribution_metrics": distribution_metrics,
        "bootstrap": bootstrap_fvd(
            real_features,
            fake_features,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"FVD-I3D: {score:.6f}")
    print(
        "I3D manifold metrics: "
        + ", ".join(
            f"{key}={value:.6f}"
            for key, value in distribution_metrics.items()
            if key != "nearest_k"
        )
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
