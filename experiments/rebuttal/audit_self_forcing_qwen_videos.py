#!/usr/bin/env python3
"""Audit that existing Self-Forcing videos exactly cover the matched manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    videos_path = Path(args.videos_path).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    if not videos_path.is_dir():
        raise FileNotFoundError(videos_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {str(record["output_name"]) for record in records}
    actual = {
        path.name
        for path in videos_path.iterdir()
        if path.suffix.lower() == ".mp4" and path.is_file()
    }
    if len(records) != 944 or len(expected) != 944:
        raise ValueError("Matched manifest must contain 944 unique video names")
    if actual != expected:
        raise ValueError(
            "Self-Forcing video set does not match the paired manifest: "
            f"missing={sorted(expected - actual)[:8]}, "
            f"extra={sorted(actual - expected)[:8]}"
        )

    payload = {
        "schema_version": 1,
        "status": "pass",
        "reuse_existing_videos": True,
        "self_forcing_inference_executed": False,
        "videos_path": str(videos_path),
        "num_videos": len(actual),
        "samples_per_prompt": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sorted_filename_sha256": hashlib.sha256(
            "\n".join(sorted(actual)).encode("utf-8")
        ).hexdigest(),
    }
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"PASS: audited {len(actual)} existing Self-Forcing videos: {output_path}")


if __name__ == "__main__":
    main()
