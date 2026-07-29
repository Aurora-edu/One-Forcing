#!/usr/bin/env python3
"""Audit all-GPU One-Forcing against existing historical Self-Forcing outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.consolidate_results import (  # noqa: E402
    load_vbench_results,
    official_totals,
)
from experiments.rebuttal.summarize_single_seed_vbench import (  # noqa: E402
    audit_result,
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
RNG_PROTOCOL = "self_forcing_two_shard_seed0"


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
    prompt_digest = sha256_file(prompt_path)
    rewrite_digest = sha256_file(rewrite_path)
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 944:
        raise ValueError("Matched Qwen manifest must contain one sample per prompt")
    for index, (record, prompt, rewrite) in enumerate(zip(records, prompts, rewrites)):
        expected = {
            "prompt_index": index,
            "sample_index": 0,
            "seed": index // 2,
            "initial_noise_seed": index // 2,
            "output_name": f"{prompt}-0.mp4",
            "prompt": prompt,
            "extended_prompt": rewrite,
            "prompt_file_sha256": prompt_digest,
            "rewrite_file_sha256": rewrite_digest,
            "rng_protocol": RNG_PROTOCOL,
            "rng_shard_index": index % 2,
            "rng_position_in_shard": index // 2,
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
        "process_seed": 0,
        "initial_noise_seed_scope": "index_within_historical_even_odd_shard",
        "historical_num_rng_streams": 2,
        "historical_prompt_sharding": "even_odd",
        "rng_protocol": RNG_PROTOCOL,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_digest,
        "qwen_rewrite_path": str(rewrite_path),
        "qwen_rewrite_sha256": rewrite_digest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_video_names": [record["output_name"] for record in records],
    }


def collect_result_video_names(value) -> set[str]:
    names = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"video_path", "video"} and isinstance(item, str):
                names.add(Path(item).name)
            else:
                names.update(collect_result_video_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(collect_result_video_names(item))
    return names


def audit_existing_self_forcing_result(path: Path, expected_names: set[str]) -> dict:
    if not path.is_file() or not path.name.endswith("_eval_results.json"):
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    referenced_names = collect_result_video_names(raw)
    outside = sorted(referenced_names - expected_names)
    if outside:
        raise ValueError(
            f"Existing Self-Forcing result references videos outside the manifest: {outside[:8]}"
        )
    if len(referenced_names) < 900:
        raise ValueError(
            "Existing Self-Forcing result does not reference a complete VBench-scale set"
        )
    scores = load_vbench_results(path)
    if set(scores) != EXPECTED_DIMENSIONS:
        raise ValueError("Self-Forcing result does not contain exactly all 16 dimensions")
    totals = official_totals(scores)
    if totals is None or not all(math.isfinite(value) for value in totals.values()):
        raise ValueError("Self-Forcing aggregate is incomplete or non-finite")
    return {
        "result_path": str(path),
        "result_sha256": sha256_file(path),
        "reuse_existing_result": True,
        "self_forcing_inference_executed": False,
        "samples_per_prompt": 1,
        "use_ema": True,
        "weight_source": "generator_ema",
        "num_result_referenced_videos": len(referenced_names),
        "scores": scores,
        "normalized_aggregates": totals,
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
    parser.add_argument("--self_forcing_video_audit", required=True)
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
    expected_names = set(pairing.pop("expected_video_names"))
    one_forcing = audit_result(
        "one_forcing_raw_noema_all4", Path(args.one_forcing_result).resolve()
    )
    if set(one_forcing["scores"]) != EXPECTED_DIMENSIONS:
        raise ValueError("One-Forcing result does not contain exactly all 16 dimensions")
    self_forcing = audit_existing_self_forcing_result(
        Path(args.self_forcing_result).resolve(), expected_names
    )

    export_path = Path(one_forcing["export_path"])
    export = json.loads(export_path.read_text(encoding="utf-8"))
    intent_path = export_path.with_name("export.intent.json")
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    gpu_audit_path = Path(args.gpu_audit).resolve()
    gpu_audit = json.loads(gpu_audit_path.read_text(encoding="utf-8"))
    detected_gpu_count = int(gpu_audit.get("detected_gpu_count", 0))
    expected_intent = {
        "manifest_sha256": pairing["manifest_sha256"],
        "prompt_sha256": pairing["prompt_sha256"],
        "extended_prompt_sha256": pairing["qwen_rewrite_sha256"],
        "method": "framewise",
        "schedule": "all4",
        "num_output_frames": 21,
        "selected_manifest_records": 944,
        "use_ema": False,
        "num_shards": detected_gpu_count,
        "gpu_indices": gpu_audit.get("selected_gpu_indices"),
        "historical_num_rng_streams": 2,
        "rng_protocol": RNG_PROTOCOL,
        "rng_state_reset_per_record": True,
        "process_seed": 0,
        "initial_noise_seed_scope": "index_within_historical_even_odd_shard",
    }
    intent_mismatches = {
        key: (intent.get(key), value)
        for key, value in expected_intent.items()
        if intent.get(key) != value
    }
    if intent_mismatches:
        raise ValueError(f"One-Forcing export intent mismatch: {intent_mismatches}")
    if (
        gpu_audit.get("status") != "pass"
        or gpu_audit.get("uses_all_detected_gpus") is not True
        or detected_gpu_count < 2
        or detected_gpu_count % 2
    ):
        raise ValueError(f"Invalid all-GPU audit: {gpu_audit_path}")
    if export.get("manifest_sha256") != pairing["manifest_sha256"]:
        raise ValueError("One-Forcing completion record has a stale manifest hash")
    if (
        export.get("use_ema") is not False
        or export.get("weight_source") != "generator"
        or export.get("rng_protocol") != RNG_PROTOCOL
        or export.get("rng_state_reset_per_record") is not True
        or int(export.get("num_shards", 0)) != detected_gpu_count
    ):
        raise ValueError(f"One-Forcing completion provenance mismatch: {export_path}")
    exported_config = Path(intent["config_path"]).resolve()
    config = load_config(str(exported_config))
    if list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError(f"One-Forcing source config is not all4: {exported_config}")
    if int(config.num_frame_per_block) != 1:
        raise ValueError("One-Forcing comparison must use framewise generation")

    sf_video_audit_path = Path(args.self_forcing_video_audit).resolve()
    sf_video_audit = json.loads(sf_video_audit_path.read_text(encoding="utf-8"))
    sf_expected = {
        "status": "pass",
        "reuse_existing_videos": True,
        "self_forcing_inference_executed": False,
        "num_videos": 944,
        "samples_per_prompt": 1,
        "manifest_sha256": pairing["manifest_sha256"],
    }
    sf_mismatches = {
        key: (sf_video_audit.get(key), value)
        for key, value in sf_expected.items()
        if sf_video_audit.get(key) != value
    }
    if sf_mismatches:
        raise ValueError(f"Self-Forcing existing-video audit mismatch: {sf_mismatches}")

    pairing.update(
        {
            "gpu_audit_path": str(gpu_audit_path),
            "detected_gpu_count": detected_gpu_count,
            "selected_gpu_indices": gpu_audit["selected_gpu_indices"],
            "uses_all_detected_gpus": True,
            "one_forcing_export_path": str(export_path),
            "one_forcing_export_intent_path": str(intent_path),
            "one_forcing_config_path": str(exported_config),
            "self_forcing_video_audit_path": str(sf_video_audit_path),
            "self_forcing_videos_reused": True,
            "self_forcing_result_reused": True,
        }
    )
    output = {
        "schema_version": 3,
        "status": "pass",
        "protocol": (
            "Only One-Forcing inference is executed. Its raw/no-EMA all4 samples "
            "use the same 944 original prompts, exact Qwen rewrites, one sample, "
            "and historical initial-noise seeds plus the two process-global "
            "seed-0 CUDA RNG streams used by the existing Self-Forcing videos. "
            "Those two global RNG streams are restored per record "
            "while work is distributed across every detected GPU. Existing "
            "Self-Forcing EMA videos and result JSON are audited and reused."
        ),
        "pairing": pairing,
        "runs": {
            "one_forcing_raw_noema_all4": one_forcing,
            "self_forcing_ema_all4_existing": self_forcing,
        },
        "comparisons": {
            "one_forcing_minus_self_forcing": {
                "formula": "one_forcing_raw_noema_all4 - self_forcing_ema_all4_existing",
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
    print(f"PASS: existing-SF matched Qwen comparison audited: {output_path}")


if __name__ == "__main__":
    main()
