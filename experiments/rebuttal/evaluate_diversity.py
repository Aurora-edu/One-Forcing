#!/usr/bin/env python3
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def read_manifest(path):
    groups = defaultdict(list)
    output_names = set()
    with open(path, encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("prompt_index", "sample_index", "seed", "output_name"):
                if key not in record:
                    raise ValueError(f"{path}:{line_number}: missing {key}")
            prompt_index = int(record["prompt_index"])
            sample_index = int(record["sample_index"])
            if prompt_index < 0 or sample_index < 0 or int(record["seed"]) < 0:
                raise ValueError(
                    f"{path}:{line_number}: indices and seed must be non-negative"
                )
            output_name = str(record["output_name"])
            if Path(output_name).name != output_name or not output_name.endswith(".mp4"):
                raise ValueError(
                    f"{path}:{line_number}: output_name must be a plain .mp4 filename"
                )
            if output_name in output_names:
                raise ValueError(f"{path}:{line_number}: duplicate output_name")
            output_names.add(output_name)
            if any(
                int(existing["sample_index"]) == sample_index
                for existing in groups[prompt_index]
            ):
                raise ValueError(
                    f"{path}:{line_number}: duplicate prompt/sample pair "
                    f"({prompt_index}, {sample_index})"
                )
            groups[prompt_index].append(record)
    if not groups:
        raise ValueError(f"No records in {path}")
    for records in groups.values():
        records.sort(key=lambda item: int(item["sample_index"]))
    return groups


def sample_video_frames(path, num_frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        capture.release()
        raise RuntimeError(f"No frames in {path}")
    indices = np.linspace(0, frame_count - 1, num=min(num_frames, frame_count))
    indices = sorted(set(int(round(value)) for value in indices))
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed reading frame {index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))
    capture.release()
    return torch.stack(frames).float().div_(255.0)


def bootstrap_ci(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if samples <= 0 or len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    parser = argparse.ArgumentParser(
        description="Measure within-prompt perceptual diversity across paired samples."
    )
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--metric", default="lpips-vgg")
    parser.add_argument("--frames_per_video", type=int, default=8)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--min_samples_per_prompt", type=int, default=4)
    args = parser.parse_args()

    if (
        args.frames_per_video < 1
        or args.resize < 32
        or args.min_samples_per_prompt < 2
    ):
        raise ValueError(
            "frames_per_video must be positive, resize must be >=32, and "
            "min_samples_per_prompt must be >=2"
        )
    try:
        import pyiqa
    except ImportError as exc:
        raise SystemExit("Install pyiqa (pinned in requirements.txt) for LPIPS diversity") from exc

    device = torch.device(args.device)
    metric = pyiqa.create_metric(args.metric, as_loss=False, device=device)
    groups = read_manifest(args.manifest_path)
    videos_dir = Path(args.videos_dir)
    expected_names = {
        str(record["output_name"])
        for records in groups.values()
        for record in records
    }
    actual_names = {
        path.name
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    }
    if actual_names != expected_names:
        raise ValueError(
            "Diversity videos do not exactly match the manifest: "
            f"missing={sorted(expected_names - actual_names)[:8]}, "
            f"extra={sorted(actual_names - expected_names)[:8]}"
        )
    prompt_results = []

    with torch.no_grad():
        for prompt_index, records in sorted(groups.items()):
            if len(records) < args.min_samples_per_prompt:
                raise ValueError(
                    f"prompt_index={prompt_index} has {len(records)} samples; "
                    f"at least {args.min_samples_per_prompt} required"
                )
            sampled = {}
            for record in records:
                video_path = videos_dir / record["output_name"]
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                frames = sample_video_frames(video_path, args.frames_per_video)
                sampled[int(record["sample_index"])] = F.interpolate(
                    frames,
                    size=(args.resize, args.resize),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

            pair_scores = []
            for left_index, right_index in itertools.combinations(sorted(sampled), 2):
                left = sampled[left_index].to(device)
                right = sampled[right_index].to(device)
                if left.shape[0] != right.shape[0]:
                    raise RuntimeError("Paired videos yielded different sampled frame counts")
                value = metric(left, right)
                score = float(value.float().mean().item())
                if not math.isfinite(score):
                    raise ValueError(
                        f"Non-finite diversity score for prompt_index={prompt_index}"
                    )
                pair_scores.append(score)
            prompt_results.append(
                {
                    "prompt_index": prompt_index,
                    "num_samples": len(records),
                    "num_pairs": len(pair_scores),
                    "mean_pairwise_lpips": float(np.mean(pair_scores)),
                    "pair_scores": pair_scores,
                }
            )

    prompt_means = [item["mean_pairwise_lpips"] for item in prompt_results]
    result = {
        "schema_version": 1,
        "metric": args.metric,
        "videos_dir": str(videos_dir.resolve()),
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "interpretation": "Higher is more diverse; report jointly with quality metrics.",
        "num_prompts": len(prompt_results),
        "frames_per_video": args.frames_per_video,
        "min_samples_per_prompt": args.min_samples_per_prompt,
        "mean_pairwise_lpips": float(np.mean(prompt_means)),
        "sample_std_over_prompts": (
            float(np.std(prompt_means, ddof=1)) if len(prompt_means) > 1 else None
        ),
        "standard_error": (
            float(np.std(prompt_means, ddof=1) / math.sqrt(len(prompt_means)))
            if len(prompt_means) > 1
            else None
        ),
        "bootstrap_95ci": bootstrap_ci(
            prompt_means,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        "per_prompt": prompt_results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
