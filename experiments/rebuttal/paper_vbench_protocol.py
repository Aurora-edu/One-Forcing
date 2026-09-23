#!/usr/bin/env python3
"""Prepare and audit a shared, Qwen-conditioned VBench evaluation protocol.

This protocol is for paired rebuttal evaluations.  It fixes the VBench prompt
metadata and the exact historical Qwen rewrites, and records generation seeds.
It does not assert that a newly generated score reproduces a paper checkpoint's
historical random stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from experiments.rebuttal.consolidate_results import (
        load_vbench_results,
        official_totals,
    )
    from experiments.rebuttal.merge_qwen_rewrite_shards import (
        load_pair_mapping,
        merge_in_prompt_order,
    )
    from experiments.rebuttal.prepare_vbench_prompts import unique_vbench_prompts
except ModuleNotFoundError:
    from consolidate_results import load_vbench_results, official_totals
    from merge_qwen_rewrite_shards import load_pair_mapping, merge_in_prompt_order
    from prepare_vbench_prompts import unique_vbench_prompts


ASSETS = Path(__file__).resolve().parent / "assets" / "qwen_vbench"
CONFIGS = Path(__file__).resolve().parent / "configs"
PINNED_HASHES = {
    "VBench_full_info.json": "12d720a3f5ec60d7640edadd2272876056da098632171fc30356be25674c4deb",
    "shard00_pairs.jsonl": "53e85750f9fec2ff0a1af9b1d8ac9adf3c9e6b69dbf69cf529d3b56be4017d7e",
    "shard01_pairs.jsonl": "a9126faa105e2aeb976b352877576f75a97b57e6784c78cb20d3b8c1d5dbdbb6",
}
EXPECTED_PROMPTS = 944


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_inputs(asset_dir: Path = ASSETS) -> tuple[list[str], list[str]]:
    for name, expected in PINNED_HASHES.items():
        path = asset_dir / name
        if sha256_file(path) != expected:
            raise ValueError(f"Pinned Qwen/VBench asset changed: {path}")
    prompts = unique_vbench_prompts(asset_dir / "VBench_full_info.json")
    if len(prompts) != EXPECTED_PROMPTS:
        raise ValueError(f"Expected {EXPECTED_PROMPTS} official prompts, got {len(prompts)}")
    mapping = load_pair_mapping(
        [asset_dir / "shard00_pairs.jsonl", asset_dir / "shard01_pairs.jsonl"]
    )
    rewrites = merge_in_prompt_order(prompts, mapping)
    if any("\n" in rewrite or "\r" in rewrite for rewrite in rewrites):
        raise ValueError("Qwen rewrites must contain exactly one line per prompt")
    return prompts, rewrites


def protocol_paths(output_dir: Path, samples_per_prompt: int) -> dict[str, Path]:
    return {
        "prompts": output_dir / "vbench_prompts.txt",
        "rewrites": output_dir / "vbench_qwen_rewrites.txt",
        "manifest": output_dir / f"vbench_qwen_{samples_per_prompt}sample_seed0.jsonl",
        "protocol": output_dir / f"vbench_qwen_{samples_per_prompt}sample_protocol.json",
    }


def expected_payloads(
    prompts: list[str], rewrites: list[str], samples_per_prompt: int
) -> dict[str, str]:
    if samples_per_prompt not in (1, 5):
        raise ValueError("Supported protocols are 1 or 5 samples per prompt")
    prompt_text = "".join(f"{prompt}\n" for prompt in prompts)
    rewrite_text = "".join(f"{rewrite}\n" for rewrite in rewrites)
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    rewrite_hash = hashlib.sha256(rewrite_text.encode("utf-8")).hexdigest()
    records = []
    for prompt_index, (prompt, rewrite) in enumerate(zip(prompts, rewrites)):
        for sample_index in range(samples_per_prompt):
            records.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "seed": prompt_index * samples_per_prompt + sample_index,
                    "output_name": f"{prompt}-{sample_index}.mp4",
                    "prompt": prompt,
                    "extended_prompt": rewrite,
                    "prompt_file_sha256": prompt_hash,
                    "rewrite_file_sha256": rewrite_hash,
                }
            )
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    protocol = {
        "schema_version": 1,
        "prompt_count": EXPECTED_PROMPTS,
        "samples_per_prompt": samples_per_prompt,
        "generation_seed_rule": "prompt_index * samples_per_prompt + sample_index",
        "conditioning": "pinned_qwen_rewrite",
        "scoring_prompt": "original_vbench_prompt",
        "full_info_sha256": PINNED_HASHES["VBench_full_info.json"],
        "prompt_sha256": prompt_hash,
        "rewrite_sha256": rewrite_hash,
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "official_five_sample_protocol": samples_per_prompt == 5,
    }
    return {
        "prompts": prompt_text,
        "rewrites": rewrite_text,
        "manifest": manifest_text,
        "protocol": json.dumps(protocol, indent=2, sort_keys=True) + "\n",
    }


def prepare(output_dir: Path, samples_per_prompt: int) -> dict[str, Path]:
    prompts, rewrites = pinned_inputs()
    paths = protocol_paths(output_dir, samples_per_prompt)
    payloads = expected_payloads(prompts, rewrites, samples_per_prompt)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        if path.exists():
            if path.read_text(encoding="utf-8") != payloads[key]:
                raise FileExistsError(f"Refusing to overwrite a different protocol file: {path}")
        else:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payloads[key])
    audit_inputs(output_dir, samples_per_prompt)
    return paths


def audit_inputs(output_dir: Path, samples_per_prompt: int) -> dict:
    paths = protocol_paths(output_dir, samples_per_prompt)
    protocol = audit_files(
        paths["prompts"], paths["rewrites"], paths["manifest"],
        samples_per_prompt,
    )
    if (
        not paths["protocol"].is_file()
        or json.loads(paths["protocol"].read_text(encoding="utf-8")) != protocol
    ):
        raise ValueError(f"VBench protocol metadata is missing or inconsistent: {paths['protocol']}")
    return protocol


def verify_full_info(path: Path) -> None:
    if sha256_file(path) != PINNED_HASHES["VBench_full_info.json"]:
        raise ValueError(f"VBench full-info does not match pinned paper metadata: {path}")


def audit_files(
    prompt_path: Path, rewrite_path: Path, manifest_path: Path,
    samples_per_prompt: int,
) -> dict:
    prompts, rewrites = pinned_inputs()
    payloads = expected_payloads(prompts, rewrites, samples_per_prompt)
    paths = {
        "prompts": prompt_path,
        "rewrites": rewrite_path,
        "manifest": manifest_path,
    }
    for key, path in paths.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != payloads[key]:
            raise ValueError(f"VBench {key} does not match pinned Qwen protocol: {path}")
    protocol_path = protocol_paths(prompt_path.parent, samples_per_prompt)["protocol"]
    if protocol_path.is_file() and protocol_path.read_text(encoding="utf-8") != payloads["protocol"]:
        raise ValueError(f"VBench protocol metadata does not match pinned inputs: {protocol_path}")
    return json.loads(payloads["protocol"])


def audit_run(
    output_root: Path, name: str, samples_per_prompt: int, schedule: str,
    weight_source: str = "generator",
) -> dict:
    if schedule not in ("ffe", "all1", "all4"):
        raise ValueError(f"Unsupported schedule: {schedule}")
    if weight_source not in ("generator", "generator_ema"):
        raise ValueError(f"Unsupported weight source: {weight_source}")
    if weight_source == "generator_ema" and schedule != "all4":
        raise ValueError("Self-Forcing EMA baseline requires all4")
    input_dir = output_root / "manifests"
    protocol = audit_inputs(input_dir, samples_per_prompt)
    paths = protocol_paths(input_dir, samples_per_prompt)
    videos = output_root / "videos"
    intent = json.loads((videos / "export.intent.json").read_text(encoding="utf-8"))
    done = json.loads((videos / "export.done").read_text(encoding="utf-8"))
    config_path = (
        Path(__file__).resolve().parents[2] / "self_forcing_config.yaml"
        if weight_source == "generator_ema"
        else CONFIGS / f"eval_{schedule}.yaml"
    )
    if (
        Path(intent.get("config_path", "")).resolve() != config_path.resolve()
        or intent.get("config_sha256") != sha256_file(config_path)
    ):
        raise ValueError(f"Export did not use the pinned {schedule} evaluation config")
    expected = {
        "prompt_sha256": protocol["prompt_sha256"],
        "extended_prompt_sha256": protocol["rewrite_sha256"],
        "manifest_sha256": protocol["manifest_sha256"],
        "selected_manifest_records": EXPECTED_PROMPTS * samples_per_prompt,
        "method": "chunkwise" if weight_source == "generator_ema" else "framewise",
        "schedule": schedule,
        "num_output_frames": 21,
        "fps": 16,
        "use_ema": weight_source == "generator_ema",
    }
    mismatches = {
        key: (intent.get(key), value)
        for key, value in expected.items()
        if intent.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Export intent violates Qwen VBench protocol: {mismatches}")
    # Older committed short-video exporters did not record sink_size; they did
    # not support attention sinks and therefore always used the zero-sink path.
    if intent.get("sink_size", 0) != 0:
        raise ValueError("Short-video Qwen VBench protocol requires sink_size=0")
    for key, path in (
        ("prompt_path", paths["prompts"]),
        ("extended_prompt_path", paths["rewrites"]),
        ("manifest_path", paths["manifest"]),
    ):
        if Path(intent[key]).resolve() != path.resolve():
            raise ValueError(f"Export intent uses unexpected {key}: {intent[key]}")
    if (
        done.get("status") != "complete"
        or done.get("manifest_sha256") != protocol["manifest_sha256"]
        or done.get("num_videos") != EXPECTED_PROMPTS * samples_per_prompt
        or done.get("weight_source") != weight_source
        or done.get("use_ema") is not (weight_source == "generator_ema")
        or done.get("latent_frames_per_video") != 21
        or done.get("rgb_frames_per_video") != 81
        or done.get("fps") != 16
        or done.get("checkpoint_path") != intent.get("checkpoint_path")
        or done.get("checkpoint_size_bytes") != intent.get("checkpoint_size_bytes")
    ):
        raise ValueError("Completed export violates 81-frame Qwen VBench protocol")
    vbench_dir = output_root / "vbench"
    scoring = json.loads(
        (vbench_dir / f"{name}_protocol.json").read_text(encoding="utf-8")
    )
    if (
        scoring.get("mode") != "vbench_standard"
        or scoring.get("samples_per_prompt") != samples_per_prompt
        or scoring.get("official_five_sample_protocol") is not (samples_per_prompt == 5)
        or len(scoring.get("dimensions", [])) != 16
        or len(set(scoring["dimensions"])) != 16
    ):
        raise ValueError("VBench scoring protocol is not complete 16-dimensional evaluation")
    result_path = vbench_dir / f"{name}_eval_results.json"
    scores = load_vbench_results(result_path)
    if set(scores) != set(scoring["dimensions"]):
        raise ValueError("VBench result dimensions differ from scoring protocol")
    totals = official_totals(scores)
    if totals is None:
        raise ValueError("Complete 16-dimension VBench scores are required")
    return {
        "schema_version": 1,
        "status": "pass",
        "result_path": str(result_path.resolve()),
        "checkpoint_path": intent["checkpoint_path"],
        "checkpoint_size_bytes": intent["checkpoint_size_bytes"],
        "config_path": intent["config_path"],
        "schedule": schedule,
        "method": expected["method"],
        "weight_source": weight_source,
        "protocol": protocol,
        "scores": scores,
        "normalized_aggregates": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output_dir", required=True)
    prepare_parser.add_argument("--samples_per_prompt", type=int, default=5)
    audit_parser = subparsers.add_parser("audit_run")
    audit_parser.add_argument("--output_root", required=True)
    audit_parser.add_argument("--name", required=True)
    audit_parser.add_argument("--schedule", choices=["ffe", "all1", "all4"], required=True)
    audit_parser.add_argument("--samples_per_prompt", type=int, default=5)
    audit_parser.add_argument(
        "--weight_source", choices=["generator", "generator_ema"],
        default="generator",
    )
    verify_parser = subparsers.add_parser("verify_inputs")
    verify_parser.add_argument("--prompt_path", required=True)
    verify_parser.add_argument("--rewrite_path", required=True)
    verify_parser.add_argument("--manifest_path", required=True)
    verify_parser.add_argument("--full_info_path", required=True)
    verify_parser.add_argument("--samples_per_prompt", type=int, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        paths = prepare(Path(args.output_dir).resolve(), args.samples_per_prompt)
        print("PASS: pinned Qwen VBench inputs: " + str(paths["protocol"]))
    elif args.command == "verify_inputs":
        verify_full_info(Path(args.full_info_path).resolve())
        audit_files(
            Path(args.prompt_path).resolve(),
            Path(args.rewrite_path).resolve(),
            Path(args.manifest_path).resolve(),
            args.samples_per_prompt,
        )
        print("PASS: exact pinned Qwen/VBench inputs")
    else:
        output_root = Path(args.output_root).resolve()
        summary = audit_run(
            output_root, args.name, args.samples_per_prompt, args.schedule,
            args.weight_source,
        )
        summary_path = output_root / f"{args.name}_paper_protocol_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("PASS: Qwen VBench run: " + str(summary_path))


if __name__ == "__main__":
    main()
