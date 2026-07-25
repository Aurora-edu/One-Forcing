#!/usr/bin/env python3
"""Estimate the fixed 48-hour evaluation matrix from measured schedule latency."""

import argparse
import json


VIDEO_COUNTS = {
    "all1": 5120,
    "ffe": 11622,
    "all4": 9440,
}


def estimate(all1_seconds, ffe_seconds, all4_seconds, num_gpus):
    latencies = {
        "all1": float(all1_seconds),
        "ffe": float(ffe_seconds),
        "all4": float(all4_seconds),
    }
    if num_gpus < 1 or any(value <= 0 for value in latencies.values()):
        raise ValueError("Latencies and num_gpus must be positive")
    schedule_hours = {
        schedule: VIDEO_COUNTS[schedule] * latencies[schedule] / num_gpus / 3600.0
        for schedule in VIDEO_COUNTS
    }
    return {
        "num_gpus": num_gpus,
        "video_counts": VIDEO_COUNTS,
        "seconds_per_video": latencies,
        "schedule_generation_hours": schedule_hours,
        "total_generation_hours": sum(schedule_hours.values()),
        "scope": (
            "Generation and VAE decode only; model loading, MP4 validation, "
            "VBench/FVD/LPIPS, and training are excluded."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all1_seconds", type=float, required=True)
    parser.add_argument("--ffe_seconds", type=float, required=True)
    parser.add_argument("--all4_seconds", type=float, required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            estimate(
                args.all1_seconds,
                args.ffe_seconds,
                args.all4_seconds,
                args.num_gpus,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
