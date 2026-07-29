#!/usr/bin/env python3
"""Verify that completed curved/rectified CD arms differ only by treatment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


CONTROLLED_FIELDS = (
    "initial_generator_ckpt",
    "data_path",
    "dataset_manifest_sha256",
    "seed",
    "framewise",
    "training_num_frames",
    "training_timesteps",
    "adjacent_pairs",
    "pair_schedule",
    "training_objective",
    "target_network",
    "target_ema_decay",
    "evaluation_use_ema",
    "final_step",
    "global_sample_order_sha256",
)


def load_done(path: Path, expected_intervention: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete training marker: {path}")
    if payload.get("intervention") != expected_intervention:
        raise ValueError(f"Unexpected intervention in {path}: {payload.get('intervention')}")
    if payload.get("use_ema") is not False or payload.get("weight_source") != "generator":
        raise ValueError(f"Final evaluation source is not raw/no-EMA: {path}")
    if payload.get("contains_generator_ema") is not False:
        raise ValueError(f"The saved checkpoint unexpectedly contains generator_ema: {path}")
    checkpoint = Path(payload["final_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curved_done", required=True)
    parser.add_argument("--rectified_done", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    curved_path = Path(args.curved_done).resolve()
    rectified_path = Path(args.rectified_done).resolve()
    curved = load_done(curved_path, "curved")
    rectified = load_done(rectified_path, "rectified")
    mismatches = {
        field: {"curved": curved.get(field), "rectified": rectified.get(field)}
        for field in CONTROLLED_FIELDS
        if curved.get(field) != rectified.get(field)
    }
    if mismatches:
        raise ValueError(f"Paired-arm controls do not match: {mismatches}")

    output = {
        "schema_version": 1,
        "status": "pass",
        "curved_done": str(curved_path),
        "rectified_done": str(rectified_path),
        "controlled_fields": list(CONTROLLED_FIELDS),
        "controlled_values": {field: curved.get(field) for field in CONTROLLED_FIELDS},
        "treatment_field": "intervention",
        "curved_intervention": "curved",
        "rectified_intervention": "rectified",
        "use_ema": False,
        "weight_source": "generator",
    }
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"PASS: completed curvature CD arms are paired: {output_path}")


if __name__ == "__main__":
    main()
