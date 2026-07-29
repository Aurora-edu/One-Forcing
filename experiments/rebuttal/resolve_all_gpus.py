#!/usr/bin/env python3
"""Resolve and audit every physical GPU on the current experiment host."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def run_nvidia_smi(arguments: list[str]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def detect_gpus() -> list[dict]:
    lines = run_nvidia_smi(
        [
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not lines:
        raise RuntimeError("nvidia-smi found no physical GPUs")
    gpus = []
    for line in lines:
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4 or not parts[0].isdigit():
            raise ValueError(f"Unexpected nvidia-smi GPU record: {line!r}")
        index = int(parts[0])
        pids = run_nvidia_smi(
            [
                "-i",
                str(index),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": int(parts[3]),
                "compute_pids_before_launch": [int(pid) for pid in pids],
            }
        )
    indices = [gpu["index"] for gpu in gpus]
    if len(indices) != len(set(indices)):
        raise ValueError(f"nvidia-smi returned duplicate GPU indices: {indices}")
    return sorted(gpus, key=lambda gpu: gpu["index"])


def resolve_requested(requested: str, detected: list[int]) -> list[int]:
    if requested == "all":
        return detected
    parts = requested.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("--requested must be 'all' or comma-separated GPU indices")
    selected = [int(part) for part in parts]
    if len(selected) != len(set(selected)):
        raise ValueError(f"--requested contains duplicate GPU indices: {requested}")
    if set(selected) != set(detected):
        raise ValueError(
            "Reviewer experiments must use every physical GPU on this host: "
            f"requested={selected}, detected={detected}"
        )
    return detected


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requested", default="all")
    parser.add_argument("--require_idle", action="store_true")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    gpus = detect_gpus()
    detected = [gpu["index"] for gpu in gpus]
    selected = resolve_requested(args.requested, detected)
    busy = {
        gpu["index"]: gpu["compute_pids_before_launch"]
        for gpu in gpus
        if gpu["compute_pids_before_launch"]
    }
    if args.require_idle and busy:
        raise RuntimeError(
            "Some GPUs belong to existing sessions; refusing to interrupt or overlap: "
            f"{busy}"
        )
    payload = {
        "schema_version": 1,
        "status": "pass",
        "hostname": os.uname().nodename,
        "detected_gpu_count": len(gpus),
        "selected_gpu_indices": selected,
        "uses_all_detected_gpus": True,
        "required_idle": bool(args.require_idle),
        "gpus": gpus,
    }
    atomic_write_json(Path(args.output_path).resolve(), payload)
    print(",".join(str(index) for index in selected))


if __name__ == "__main__":
    main()
