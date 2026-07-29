#!/usr/bin/env python3
"""Audit an all-GPU, seed-paired Qwen all4 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.consolidate_results import (  # noqa: E402
    load_vbench_results,
    official_totals,
)
from utils.config import load_config  # noqa: E402


EXPECTED_DIMENSIONS = {
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_nonempty_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"Expected non-empty one-line records in {path}")
    return lines


def audit_manifest(
    prompt_path: Path,
    rewrite_path: Path,
    manifest_path: Path,
) -> dict:
    prompts = read_nonempty_lines(prompt_path)
    rewrites = read_nonempty_lines(rewrite_path)
    if len(prompts) != 944 or len(rewrites) != 944:
        raise ValueError(
            f"Expected 944 prompts and rewrites, found {len(prompts)} and {len(rewrites)}"
        )
    if len(set(prompts)) != 944:
        raise ValueError("Official VBench prompts must be unique")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 944:
        raise ValueError("Matched Qwen manifest must contain one sample per prompt")
    prompt_digest = sha256_file(prompt_path)
    for index, (record, prompt, rewrite) in enumerate(zip(records, prompts, rewrites)):
        expected = {
            "prompt_index": index,
            "sample_index": 0,
            "seed": index,
            "output_name": f"{prompt}-0.mp4",
            "prompt": prompt,
            "extended_prompt": rewrite,
            "prompt_file_sha256": prompt_digest,
        }
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Manifest record {index} is not protocol matched: {mismatches}")
    return {
        "num_prompts": 944,
        "samples_per_prompt": 1,
        "base_seed": 0,
        "seed_formula": "base_seed + prompt_index",
        "rng_scope": "per_manifest_record",
        "shard_order_independent": True,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_digest,
        "qwen_rewrite_path": str(rewrite_path),
        "qwen_rewrite_sha256": sha256_file(rewrite_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def audit_vbench_result(label: str, path: Path, *, use_ema: bool) -> dict:
    if not path.is_file() or not path.name.endswith("_eval_results.json"):
        raise FileNotFoundError(path)
    name = path.name[: -len("_eval_results.json")]
    protocol_path = path.with_name(f"{name}_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("mode") != "vbench_standard":
        raise ValueError(f"{label}: expected vbench_standard")
    if protocol.get("samples_per_prompt") != 1:
        raise ValueError(f"{label}: expected one sample per prompt")
    if protocol.get("official_five_sample_protocol") is not False:
        raise ValueError(f"{label}: inconsistent one-sample protocol metadata")
    if set(protocol.get("dimensions", [])) != EXPECTED_DIMENSIONS:
        raise ValueError(f"{label}: expected exactly all 16 VBench dimensions")

    export_path = path.parent.parent / "videos" / "export.done"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    expected_source = "generator_ema" if use_ema else "generator"
    if export.get("use_ema") is not use_ema:
        raise ValueError(f"{label}: use_ema provenance mismatch: {export_path}")
    if export.get("weight_source") != expected_source:
        raise ValueError(f"{label}: weight source mismatch: {export_path}")

    scores = load_vbench_results(path)
    if set(scores) != EXPECTED_DIMENSIONS:
        raise ValueError(f"{label}: result does not contain exactly all 16 dimensions")
    totals = official_totals(scores)
    if totals is None or not all(math.isfinite(value) for value in totals.values()):
        raise ValueError(f"{label}: incomplete or non-finite aggregate")
    return {
        "result_path": str(path),
        "protocol_path": str(protocol_path),
        "export_path": str(export_path),
        "checkpoint_path": export["checkpoint_path"],
        "samples_per_prompt": 1,
        "use_ema": use_ema,
        "weight_source": expected_source,
        "scores": scores,
        "normalized_aggregates": totals,
    }


def resolved_config_sha256(path: Path) -> str:
    resolved = OmegaConf.to_yaml(load_config(str(path)), resolve=True, sort_keys=True)
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def audit_export_inputs(run: dict, pairing: dict) -> dict:
    export_path = Path(run["export_path"])
    export = json.loads(export_path.read_text(encoding="utf-8"))
    intent_path = export_path.with_name("export.intent.json")
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    manifest_path = Path(export["manifest_path"]).resolve()
    config_path = Path(intent["config_path"]).resolve()
    extended_path = Path(intent["extended_prompt_path"]).resolve()
    manifest_digest = sha256_file(manifest_path)
    checks = {
        "manifest_sha256": (manifest_digest, pairing["manifest_sha256"]),
        "export_manifest_sha256": (
            export.get("manifest_sha256"),
            pairing["manifest_sha256"],
        ),
        "intent_manifest_sha256": (
            intent.get("manifest_sha256"),
            pairing["manifest_sha256"],
        ),
        "prompt_sha256": (intent.get("prompt_sha256"), pairing["prompt_sha256"]),
        "extended_prompt_sha256": (
            intent.get("extended_prompt_sha256"),
            pairing["qwen_rewrite_sha256"],
        ),
        "extended_prompt_file_sha256": (
            sha256_file(extended_path),
            pairing["qwen_rewrite_sha256"],
        ),
        "method": (intent.get("method"), "framewise"),
        "schedule": (intent.get("schedule"), "all4"),
        "num_output_frames": (intent.get("num_output_frames"), 21),
        "selected_manifest_records": (intent.get("selected_manifest_records"), 944),
        "use_ema": (intent.get("use_ema"), run["use_ema"]),
    }
    mismatches = {
        key: values for key, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"{run['weight_source']} export mismatch: {mismatches}")
    config = load_config(str(config_path))
    if list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError(f"Export config is not all4: {config_path}")
    if int(config.num_frame_per_block) != 1:
        raise ValueError(f"Export config is not framewise: {config_path}")
    num_shards = int(intent.get("num_shards", 0))
    if num_shards < 1:
        raise ValueError("Export intent has no positive num_shards")
    return {
        "intent_path": str(intent_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "config_path": str(config_path),
        "resolved_config_sha256": resolved_config_sha256(config_path),
        "extended_prompt_path": str(extended_path),
        "extended_prompt_sha256": sha256_file(extended_path),
        "method": intent["method"],
        "schedule": intent["schedule"],
        "num_output_frames": intent["num_output_frames"],
        "selected_manifest_records": intent["selected_manifest_records"],
        "num_shards": num_shards,
        "use_ema": intent["use_ema"],
    }


def delta(right: dict, left: dict) -> dict:
    return {
        "scores": {
            key: right["scores"][key] - left["scores"][key]
            for key in sorted(EXPECTED_DIMENSIONS)
        },
        "normalized_aggregates": {
            key: right["normalized_aggregates"][key]
            - left["normalized_aggregates"][key]
            for key in right["normalized_aggregates"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one_forcing_result", required=True)
    parser.add_argument("--self_forcing_result", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--qwen_rewrite_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--gpu_audit", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    pairing = audit_manifest(
        Path(args.prompt_path).resolve(),
        Path(args.qwen_rewrite_path).resolve(),
        Path(args.manifest_path).resolve(),
    )
    one_forcing = audit_vbench_result(
        "one_forcing_raw_noema_all4",
        Path(args.one_forcing_result).resolve(),
        use_ema=False,
    )
    self_forcing = audit_vbench_result(
        "self_forcing_ema_all4",
        Path(args.self_forcing_result).resolve(),
        use_ema=True,
    )
    one_forcing_inputs = audit_export_inputs(one_forcing, pairing)
    self_forcing_inputs = audit_export_inputs(self_forcing, pairing)
    controlled_fields = (
        "manifest_sha256",
        "resolved_config_sha256",
        "extended_prompt_sha256",
        "method",
        "schedule",
        "num_output_frames",
        "selected_manifest_records",
        "num_shards",
    )
    controlled_mismatches = {
        field: (one_forcing_inputs[field], self_forcing_inputs[field])
        for field in controlled_fields
        if one_forcing_inputs[field] != self_forcing_inputs[field]
    }
    if controlled_mismatches:
        raise ValueError(f"OF/SF paired protocol mismatch: {controlled_mismatches}")

    gpu_audit_path = Path(args.gpu_audit).resolve()
    gpu_audit = json.loads(gpu_audit_path.read_text(encoding="utf-8"))
    expected_gpu_count = int(gpu_audit.get("detected_gpu_count", 0))
    if (
        gpu_audit.get("status") != "pass"
        or gpu_audit.get("uses_all_detected_gpus") is not True
        or expected_gpu_count < 1
        or one_forcing_inputs["num_shards"] != expected_gpu_count
        or self_forcing_inputs["num_shards"] != expected_gpu_count
    ):
        raise ValueError(f"All-GPU audit does not match exports: {gpu_audit_path}")

    pairing.update(
        {
            "gpu_audit_path": str(gpu_audit_path),
            "detected_gpu_count": expected_gpu_count,
            "selected_gpu_indices": gpu_audit["selected_gpu_indices"],
            "uses_all_detected_gpus": True,
            "controlled_fields": list(controlled_fields),
            "one_forcing_inputs": one_forcing_inputs,
            "self_forcing_inputs": self_forcing_inputs,
        }
    )
    output = {
        "schema_version": 2,
        "status": "pass",
        "protocol": (
            "Both all4 methods are regenerated from the same 944 original prompts, "
            "exact Qwen rewrites, and one-sample manifest. Every record resets all "
            "generation RNGs to base_seed+prompt_index, so sharding across every "
            "detected GPU preserves exact per-prompt random pairing. Both video sets "
            "are scored by the same repository-pinned VBench 0.1.5 command."
        ),
        "pairing": pairing,
        "runs": {
            "one_forcing_raw_noema_all4": one_forcing,
            "self_forcing_ema_all4": self_forcing,
        },
        "comparisons": {
            "one_forcing_minus_self_forcing": {
                "formula": "one_forcing_raw_noema_all4 - self_forcing_ema_all4",
                **delta(one_forcing, self_forcing),
            }
        },
    }
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"PASS: all-GPU matched Qwen all4 comparison audited: {output_path}")


if __name__ == "__main__":
    main()
