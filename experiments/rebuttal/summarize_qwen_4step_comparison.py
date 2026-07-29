#!/usr/bin/env python3
"""Audit a matched Qwen-rewrite all4 comparison against Self-Forcing videos."""

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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if len(prompts) != len(rewrites):
        raise ValueError(
            f"Original/Qwen prompt counts differ: {len(prompts)} != {len(rewrites)}"
        )
    if len(prompts) != 944:
        raise ValueError(f"Expected 944 unique VBench prompts, found {len(prompts)}")
    prompt_digest = sha256_file(prompt_path)
    rewrite_digest = sha256_file(rewrite_path)
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(prompts):
        raise ValueError("Matched Qwen manifest must contain one sample per prompt")
    for index, (record, prompt, rewrite) in enumerate(zip(records, prompts, rewrites)):
        expected = {
            "prompt_index": index,
            "sample_index": 0,
            "seed": 0,
            "output_name": f"{prompt}-0.mp4",
            "prompt": prompt,
            "extended_prompt": rewrite,
            "prompt_file_sha256": prompt_digest,
            "rewrite_file_sha256": rewrite_digest,
            "rng_protocol": "self_forcing_two_shard_seed0",
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
        "num_prompts": len(prompts),
        "samples_per_prompt": 1,
        "process_seed": 0,
        "num_generation_shards": 2,
        "prompt_sharding": "even_odd",
        "rng_protocol": "self_forcing_two_shard_seed0",
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "qwen_rewrite_path": str(rewrite_path),
        "qwen_rewrite_sha256": sha256_file(rewrite_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def audit_self_forcing_result(path: Path) -> dict:
    suffix = "_eval_results.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected Self-Forcing result name: {path}")
    protocol_path = path.with_name(path.name[: -len(suffix)] + "_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("mode") != "vbench_standard" or protocol.get("samples_per_prompt") != 1:
        raise ValueError("Self-Forcing rescoring must use one-sample vbench_standard")
    scores = load_vbench_results(path)
    if set(scores) != EXPECTED_DIMENSIONS:
        raise ValueError("Self-Forcing result does not contain exactly all 16 dimensions")
    totals = official_totals(scores)
    if totals is None or not all(math.isfinite(value) for value in totals.values()):
        raise ValueError("Self-Forcing aggregate is incomplete or non-finite")
    return {
        "result_path": str(path),
        "protocol_path": str(protocol_path),
        "weight_source": "generator_ema",
        "use_ema": True,
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
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    one_forcing = audit_result(
        "one_forcing_raw_noema_all4", Path(args.one_forcing_result).resolve()
    )
    if set(one_forcing["scores"]) != EXPECTED_DIMENSIONS:
        raise ValueError("One-Forcing result does not contain exactly all 16 dimensions")
    self_forcing = audit_self_forcing_result(Path(args.self_forcing_result).resolve())
    pairing = audit_manifest(
        Path(args.prompt_path).resolve(),
        Path(args.qwen_rewrite_path).resolve(),
        Path(args.manifest_path).resolve(),
    )
    one_forcing_export = json.loads(
        Path(one_forcing["export_path"]).read_text(encoding="utf-8")
    )
    exported_manifest = Path(one_forcing_export["manifest_path"]).resolve()
    exported_manifest_sha256 = sha256_file(exported_manifest)
    if exported_manifest_sha256 != pairing["manifest_sha256"]:
        raise ValueError("One-Forcing export did not use the audited Qwen manifest")
    if one_forcing_export.get("manifest_sha256") != exported_manifest_sha256:
        raise ValueError("One-Forcing completion record has a stale manifest hash")

    intent_path = Path(one_forcing["export_path"]).with_name("export.intent.json")
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    expected_intent = {
        "manifest_sha256": pairing["manifest_sha256"],
        "method": "framewise",
        "schedule": "all4",
        "num_output_frames": 21,
        "selected_manifest_records": 944,
        "use_ema": False,
        "num_shards": 2,
        "rng_protocol": "self_forcing_two_shard_seed0",
        "process_seed": 0,
    }
    mismatches = {
        key: (intent.get(key), expected)
        for key, expected in expected_intent.items()
        if intent.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"One-Forcing all4 export intent mismatch: {mismatches}")
    exported_config = Path(intent["config_path"]).resolve()
    config = load_config(str(exported_config))
    if list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError(f"One-Forcing source config is not all4: {exported_config}")
    if int(config.num_frame_per_block) != 1:
        raise ValueError("One-Forcing comparison must use framewise generation")
    pairing["one_forcing_export_manifest_path"] = str(exported_manifest)
    pairing["one_forcing_export_config_path"] = str(exported_config)
    pairing["one_forcing_export_intent_path"] = str(intent_path)
    self_forcing_video_audit_path = Path(args.self_forcing_video_audit).resolve()
    self_forcing_video_audit = json.loads(
        self_forcing_video_audit_path.read_text(encoding="utf-8")
    )
    expected_video_audit = {
        "status": "pass",
        "num_videos": 944,
        "samples_per_prompt": 1,
        "manifest_sha256": pairing["manifest_sha256"],
    }
    video_audit_mismatches = {
        key: (self_forcing_video_audit.get(key), expected)
        for key, expected in expected_video_audit.items()
        if self_forcing_video_audit.get(key) != expected
    }
    if video_audit_mismatches:
        raise ValueError(
            f"Self-Forcing video-set audit mismatch: {video_audit_mismatches}"
        )
    pairing["self_forcing_video_audit_path"] = str(self_forcing_video_audit_path)

    output = {
        "schema_version": 1,
        "status": "pass",
        "protocol": (
            "Both methods use all4, the same 944 original VBench prompts and exact "
            "Qwen rewrites, one sample per prompt, and the historical two-process "
            "even/odd seed-0 generation protocol. Both video sets are scored by the "
            "same repository-pinned VBench 0.1.5 command."
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
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"PASS: matched Qwen 4-step comparison audited: {output_path}")


if __name__ == "__main__":
    main()
