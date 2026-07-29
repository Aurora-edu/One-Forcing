#!/usr/bin/env python3
"""Audit and summarize two manifest-matched raw/no-EMA VBench conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.summarize_single_seed_vbench import (  # noqa: E402
    audit_result,
    score_delta,
)
from utils.config import load_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_config_sha256(path: Path) -> str:
    config = load_config(str(path))
    resolved = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def export_inputs(run: dict) -> dict:
    export = json.loads(Path(run["export_path"]).read_text(encoding="utf-8"))
    intent_path = Path(run["export_path"]).with_name("export.intent.json")
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    manifest_path = Path(export["manifest_path"]).resolve()
    config_path = Path(intent["config_path"]).resolve()
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Missing exported protocol input: manifest={manifest_path}, config={config_path}"
        )
    manifest_sha256 = sha256_file(manifest_path)
    if export.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Completed export manifest hash is inconsistent: {manifest_path}")
    if intent.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Export intent manifest hash is inconsistent: {manifest_path}")
    return {
        "intent_path": str(intent_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "config_path": str(config_path),
        "resolved_config_sha256": resolved_config_sha256(config_path),
        "method": intent.get("method"),
        "schedule": intent.get("schedule"),
        "num_output_frames": intent.get("num_output_frames"),
        "selected_manifest_records": intent.get("selected_manifest_records"),
        "num_shards": intent.get("num_shards"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_result", required=True)
    parser.add_argument("--candidate_result", required=True)
    parser.add_argument("--reference_label", default="full_step200")
    parser.add_argument("--candidate_label", default="dmd_only_step200")
    parser.add_argument("--comparison_name", default="candidate_minus_reference")
    parser.add_argument(
        "--comparison_direction",
        choices=["candidate_minus_reference", "reference_minus_candidate"],
        default="candidate_minus_reference",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gpu_audit", required=True)
    args = parser.parse_args()
    if args.reference_label == args.candidate_label:
        raise ValueError("The two labels must be different")

    reference = audit_result(args.reference_label, Path(args.reference_result).resolve())
    candidate = audit_result(args.candidate_label, Path(args.candidate_result).resolve())
    reference_inputs = export_inputs(reference)
    candidate_inputs = export_inputs(candidate)
    gpu_audit_path = Path(args.gpu_audit).resolve()
    gpu_audit = json.loads(gpu_audit_path.read_text(encoding="utf-8"))
    detected_gpu_count = int(gpu_audit.get("detected_gpu_count", 0))
    if (
        gpu_audit.get("status") != "pass"
        or gpu_audit.get("uses_all_detected_gpus") is not True
        or detected_gpu_count < 1
        or candidate_inputs["num_shards"] != detected_gpu_count
    ):
        raise ValueError(
            f"DMD candidate did not use every detected GPU: {gpu_audit_path}"
        )
    for field in (
        "manifest_sha256",
        "resolved_config_sha256",
        "method",
        "schedule",
        "num_output_frames",
        "selected_manifest_records",
    ):
        if reference_inputs[field] != candidate_inputs[field]:
            raise ValueError(
                f"Paired VBench {field} mismatch: reference={reference_inputs[field]}, "
                f"candidate={candidate_inputs[field]}"
            )

    if args.comparison_direction == "candidate_minus_reference":
        right_label, right = args.candidate_label, candidate
        left_label, left = args.reference_label, reference
    else:
        right_label, right = args.reference_label, reference
        left_label, left = args.candidate_label, candidate

    output = {
        "schema_version": 1,
        "protocol": (
            "Complete 16-dimension, one-sample-per-prompt VBench; exact shared "
            "manifest, generation seeds, resolved inference config, and raw/no-EMA weights."
        ),
        "status": "pass",
        "use_ema": False,
        "weight_source": "generator",
        "pairing": {
            "manifest_sha256": reference_inputs["manifest_sha256"],
            "resolved_config_sha256": reference_inputs["resolved_config_sha256"],
            "method": reference_inputs["method"],
            "schedule": reference_inputs["schedule"],
            "num_output_frames": reference_inputs["num_output_frames"],
            "selected_manifest_records": reference_inputs[
                "selected_manifest_records"
            ],
            "reference_inputs": reference_inputs,
            "candidate_inputs": candidate_inputs,
            "gpu_audit_path": str(gpu_audit_path),
            "detected_gpu_count": detected_gpu_count,
            "uses_all_detected_gpus": True,
        },
        "runs": {
            args.reference_label: reference,
            args.candidate_label: candidate,
        },
        "comparisons": {
            args.comparison_name: {
                "formula": f"{right_label} - {left_label}",
                **score_delta(right, left),
            }
        },
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
    print(f"PASS: paired raw/no-EMA VBench audited: {output_path}")


if __name__ == "__main__":
    main()
