#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid frame count/FPS for {path}: {frame_count}, {fps}")
    return frame_count, fps


def starts_for(frame_count, window_frames, positions):
    max_start = frame_count - window_frames
    if max_start < 0:
        raise ValueError(
            f"Video has {frame_count} frames, shorter than window_frames={window_frames}"
        )
    candidates = {
        "early": 0,
        "middle": max_start // 2,
        "late": max_start,
    }
    return {position: candidates[position] for position in positions}


def split_video(path, output_root, window_frames, positions):
    frame_count, fps = video_metadata(path)
    starts = starts_for(frame_count, window_frames, positions)
    writers = {}
    frames_written = {position: 0 for position in positions}
    try:
        for position in positions:
            output_dir = output_root / position
            output_dir.mkdir(parents=True, exist_ok=True)
            writers[position] = imageio.get_writer(
                output_dir / path.name,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )

        reader = imageio.get_reader(path)
        try:
            for frame_index, frame in enumerate(reader):
                for position, start in starts.items():
                    if start <= frame_index < start + window_frames:
                        writers[position].append_data(frame)
                        frames_written[position] += 1
                if frame_index >= max(starts.values()) + window_frames - 1:
                    break
        finally:
            reader.close()
    finally:
        for writer in writers.values():
            writer.close()

    for position, count in frames_written.items():
        if count != window_frames:
            raise RuntimeError(
                f"{path}: wrote {count} frames for {position}, expected {window_frames}"
            )
        output_path = output_root / position / path.name
        encoded_count, encoded_fps = video_metadata(output_path)
        if encoded_count != window_frames:
            raise RuntimeError(
                f"{output_path}: encoded {encoded_count} frames, expected {window_frames}"
            )
        if abs(encoded_fps - fps) > 1e-3:
            raise RuntimeError(
                f"{output_path}: encoded fps={encoded_fps}, expected {fps}"
            )

    return {
        "video": path.name,
        "frame_count": frame_count,
        "fps": fps,
        "window_frames": window_frames,
        "starts": starts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract exact early/middle/late windows for long-rollout VBench."
    )
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--window_frames", type=int, default=81)
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["early", "middle", "late"],
        choices=["early", "middle", "late"],
    )
    args = parser.parse_args()

    if args.window_frames < 2:
        raise ValueError("--window_frames must be at least 2")
    positions = list(dict.fromkeys(args.positions))
    videos_dir = Path(args.videos_dir)
    videos = sorted(
        path for path in videos_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in {videos_dir}")

    output_root = Path(args.output_root)
    records = []
    for path in videos:
        record = split_video(
            path,
            output_root,
            window_frames=args.window_frames,
            positions=positions,
        )
        records.append(record)
        print(f"Split {path.name}: {record['starts']}")

    with open(output_root / "windows.json", "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"Wrote {output_root / 'windows.json'}")


if __name__ == "__main__":
    main()
