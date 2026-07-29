#!/usr/bin/env python3
"""Fail-closed audit for a checkpoint evaluated from generator_ema weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    state = payload.get("generator_ema")
    if not isinstance(state, dict) or not state:
        raise KeyError(f"Checkpoint has no non-empty generator_ema state: {path}")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"generator_ema contains non-tensor values: {path}")
    return {
        "schema_version": 1,
        "checkpoint_path": str(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256_file(path),
        "num_generator_ema_tensors": len(state),
        "selected_weight_source": "generator_ema",
        "use_ema": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    audit = audit_checkpoint(checkpoint)
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
