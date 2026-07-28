#!/usr/bin/env python3
"""Validate and finalize a multi-GPU manifest export."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cv2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, limit: int):
    records = []
    with open(path, encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "output_name" not in record:
                raise ValueError(f"{path}:{line_number}: missing output_name")
            name = str(record["output_name"])
            if Path(name).name != name or not name.endswith(".mp4"):
                raise ValueError(
                    f"{path}:{line_number}: output_name must be a plain .mp4 filename"
                )
            records.append(record)
    if limit > 0:
        records = records[:limit]
    if not records:
        raise ValueError(f"Manifest contains no selected records: {path}")
    names = [str(record["output_name"]) for record in records]
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest contains duplicate output names: {path}")
    return records


def validate_video(path: Path, expected_frames: int, expected_fps: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open generated video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count != expected_frames:
        raise RuntimeError(
            f"{path}: found {frame_count} frames, expected {expected_frames}"
        )
    if not math.isfinite(fps) or abs(fps - expected_fps) > 1e-3:
        raise RuntimeError(f"{path}: encoded fps={fps}, expected {expected_fps}")


def validate_export(
    output_folder: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    num_shards: int,
    latent_frames: int,
    fps: int,
    limit: int = -1,
    expected_weight_source: str | None = None,
):
    if expected_weight_source not in {None, "generator", "generator_ema"}:
        raise ValueError(f"Invalid expected_weight_source={expected_weight_source!r}")
    records = read_manifest(manifest_path, limit=limit)
    expected_names = {str(record["output_name"]) for record in records}
    actual_names = {
        path.name
        for path in output_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    }
    if actual_names != expected_names:
        raise ValueError(
            "Sharded export does not exactly match the selected manifest: "
            f"missing={sorted(expected_names - actual_names)[:8]}, "
            f"extra={sorted(actual_names - expected_names)[:8]}"
        )

    checkpoint_resolved = str(checkpoint_path.resolve())
    manifest_resolved = str(manifest_path.resolve())
    total_from_shards = 0
    for shard_index in range(num_shards):
        done_path = output_folder / (
            f"export.shard_{shard_index:02d}_of_{num_shards:02d}.done"
        )
        if not done_path.is_file():
            raise FileNotFoundError(f"Missing shard completion record: {done_path}")
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        expected_shard_count = len(records[shard_index::num_shards])
        checks = {
            "checkpoint_path": checkpoint_resolved,
            "manifest_path": manifest_resolved,
            "num_videos": expected_shard_count,
            "num_total_videos": len(records),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "latent_frames_per_video": latent_frames,
            "rgb_frames_per_video": 1 + 4 * (latent_frames - 1),
            "fps": fps,
        }
        if expected_weight_source is not None:
            checks["weight_source"] = expected_weight_source
            checks["use_ema"] = expected_weight_source == "generator_ema"
        mismatches = {
            key: (payload.get(key), expected)
            for key, expected in checks.items()
            if payload.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Stale or inconsistent shard record {done_path}: {mismatches}"
            )
        total_from_shards += int(payload["num_videos"])
    if total_from_shards != len(records):
        raise AssertionError(
            f"Shard counts sum to {total_from_shards}, expected {len(records)}"
        )

    expected_frames = 1 + 4 * (latent_frames - 1)
    for index, name in enumerate(sorted(expected_names), start=1):
        validate_video(output_folder / name, expected_frames, fps)
        if index % 250 == 0 or index == len(expected_names):
            print(
                f"Validated videos {index}/{len(expected_names)}",
                flush=True,
            )

    result = {
        "schema_version": 1,
        "checkpoint_path": checkpoint_resolved,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "manifest_path": manifest_resolved,
        "manifest_sha256": sha256_file(manifest_path),
        "num_videos": len(records),
        "num_shards": num_shards,
        "latent_frames_per_video": latent_frames,
        "rgb_frames_per_video": expected_frames,
        "fps": fps,
        "status": "complete",
    }
    if expected_weight_source is not None:
        result["weight_source"] = expected_weight_source
        result["use_ema"] = expected_weight_source == "generator_ema"
    return result


def atomic_write_json(path: Path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--expected_weight_source",
        choices=["generator", "generator_ema"],
        required=True,
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")
    if args.num_output_frames < 1 or args.fps < 1:
        raise ValueError("--num_output_frames and --fps must be positive")
    output_folder = Path(args.output_folder).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not output_folder.is_dir():
        raise FileNotFoundError(output_folder)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    payload = validate_export(
        output_folder=output_folder,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        num_shards=args.num_shards,
        latent_frames=args.num_output_frames,
        fps=args.fps,
        limit=args.limit,
        expected_weight_source=args.expected_weight_source,
    )
    done_path = output_folder / "export.done"
    atomic_write_json(done_path, payload)
    print(f"Wrote {done_path}", flush=True)


if __name__ == "__main__":
    main()
