#!/usr/bin/env python3
"""Fail-closed audit for a raw generator checkpoint used in follow-up runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def audit_checkpoint(path: Path, expected_step: int | None = None) -> dict:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    if "generator" not in payload:
        raise KeyError(f"Checkpoint has no raw generator state: {path}")
    generator = payload["generator"]
    if not isinstance(generator, dict) or not generator:
        raise ValueError(f"Raw generator state is empty or invalid: {path}")
    if not all(torch.is_tensor(value) for value in generator.values()):
        raise ValueError(f"Raw generator state contains non-tensor values: {path}")
    step = payload.get("step")
    if expected_step is not None:
        if step is None:
            raise KeyError(
                f"Cannot verify expected step {expected_step}; checkpoint has no step: {path}"
            )
        if int(step) != expected_step:
            raise ValueError(
                f"Checkpoint step mismatch: found {step}, expected {expected_step}: {path}"
            )
    return {
        "schema_version": 1,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_size_bytes": path.stat().st_size,
        "step": int(step) if step is not None else None,
        "num_generator_tensors": len(generator),
        "generator_ema_present_but_not_selected": "generator_ema" in payload,
        "selected_weight_source": "generator",
        "use_ema": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_step", type=int)
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()
    if args.expected_step is not None and args.expected_step < 0:
        raise ValueError("--expected_step must be non-negative")
    path = Path(args.checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = audit_checkpoint(path, args.expected_step)
    if args.output_path:
        output_path = Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(audit, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
