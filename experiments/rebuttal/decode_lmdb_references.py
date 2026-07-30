#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_videos import (
    stream_decode_to_video,
    validate_video,
    write_video_with_fallback,
)
from utils.dataset import CleanLatentLMDBDataset
from utils.wan_wrapper import WanVAEWrapper


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_or_validate_intent(output_dir, intent):
    intent_path = output_dir / "reference.intent.json"
    existing_videos = [
        path for path in output_dir.iterdir() if path.suffix.lower() == ".mp4"
    ]
    if intent_path.is_file():
        existing = json.loads(intent_path.read_text(encoding="utf-8"))
        if existing != intent:
            raise ValueError(
                f"{output_dir} belongs to a different reference export; "
                "checkpoint the existing directory or use a new --output_dir"
            )
        return
    if existing_videos:
        raise ValueError(
            f"{output_dir} contains {len(existing_videos)} videos without "
            "reference.intent.json; refusing to reuse samples with unknown provenance"
        )
    descriptor = os.open(intent_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as fp:
        json.dump(intent, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())


def normalize_prompt(prompt):
    return str(prompt).replace("\n", " ").strip()


def validate_existing_manifests(reference_records, generation_records):
    if len(reference_records) != len(generation_records):
        raise ValueError(
            "Existing reference/generation manifests have different lengths: "
            f"{len(reference_records)} versus {len(generation_records)}"
        )
    dataset_indices = []
    for order, (reference, generation) in enumerate(
        zip(reference_records, generation_records)
    ):
        expected_reference_name = f"real_{order:04d}.mp4"
        expected_generation_name = f"fake_{order:04d}.mp4"
        checks = {
            "reference order": (reference.get("order"), order),
            "reference output_name": (
                reference.get("output_name"),
                expected_reference_name,
            ),
            "generation output_name": (
                generation.get("output_name"),
                expected_generation_name,
            ),
            "generation prompt_index": (generation.get("prompt_index"), order),
            "generation sample_index": (generation.get("sample_index"), 0),
            "prompt": (
                normalize_prompt(generation.get("prompt", "")),
                normalize_prompt(reference.get("prompt", "")),
            ),
        }
        mismatches = {
            key: values for key, values in checks.items() if values[0] != values[1]
        }
        if mismatches:
            raise ValueError(
                f"Existing manifests disagree at order {order}: {mismatches}"
            )
        if "dataset_index" not in reference:
            raise ValueError(
                f"Existing reference manifest record {order} lacks dataset_index"
            )
        if "seed" not in generation:
            raise ValueError(
                f"Existing generation manifest record {order} lacks seed"
            )
        dataset_indices.append(int(reference["dataset_index"]))
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError(
            "Existing reference manifest contains duplicate dataset_index values"
        )
    return dataset_indices


def select_extension_indices(dataset_size, existing_indices, num_videos, seed):
    existing = set(existing_indices)
    if min(existing, default=0) < 0 or max(existing, default=-1) >= dataset_size:
        raise ValueError(
            f"Existing dataset_index is outside dataset range [0, {dataset_size - 1}]"
        )
    remaining = [index for index in range(dataset_size) if index not in existing]
    if num_videos > len(remaining):
        raise ValueError(
            f"Requested {num_videos} additional videos but only {len(remaining)} "
            "unseen dataset samples remain"
        )
    return random.Random(seed).sample(remaining, num_videos)


def main():
    parser = argparse.ArgumentParser(
        description="Decode deterministic clean-latent LMDB samples as the FVD real set."
    )
    parser.add_argument("--lmdb_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_videos", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--streaming_decode", action="store_true")
    parser.add_argument(
        "--existing_reference_manifest_path",
        default="",
        help="Existing reference manifest to extend without resampling its LMDB rows.",
    )
    parser.add_argument(
        "--existing_generation_manifest_path",
        default="",
        help="Existing fake manifest paired with --existing_reference_manifest_path.",
    )
    args = parser.parse_args()

    if args.num_videos < 2 or args.seed < 0 or args.fps < 1:
        raise ValueError("num_videos must be >=2, seed non-negative, and fps positive")
    if bool(args.existing_reference_manifest_path) != bool(
        args.existing_generation_manifest_path
    ):
        raise ValueError(
            "Pass both existing manifest paths when extending an evaluation set"
        )
    dataset = CleanLatentLMDBDataset(args.lmdb_path, readahead=False)
    existing_reference_records = []
    existing_generation_records = []
    existing_indices = []
    if args.existing_reference_manifest_path:
        existing_reference_records = read_jsonl(args.existing_reference_manifest_path)
        existing_generation_records = read_jsonl(args.existing_generation_manifest_path)
        existing_indices = validate_existing_manifests(
            existing_reference_records,
            existing_generation_records,
        )
    indices = select_extension_indices(
        len(dataset),
        existing_indices,
        args.num_videos,
        args.seed,
    )
    start_order = len(existing_reference_records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intent = {
        "schema_version": 1,
        "lmdb_path": str(Path(args.lmdb_path).resolve()),
        "dataset_size": len(dataset),
        "selected_dataset_indices": indices,
        "start_order": start_order,
        "num_videos": args.num_videos,
        "seed": args.seed,
        "fps": args.fps,
        "streaming_decode": bool(args.streaming_decode),
        "existing_reference_manifest_path": (
            str(Path(args.existing_reference_manifest_path).resolve())
            if args.existing_reference_manifest_path
            else None
        ),
        "existing_reference_manifest_sha256": (
            sha256_file(args.existing_reference_manifest_path)
            if args.existing_reference_manifest_path
            else None
        ),
        "existing_generation_manifest_path": (
            str(Path(args.existing_generation_manifest_path).resolve())
            if args.existing_generation_manifest_path
            else None
        ),
        "existing_generation_manifest_sha256": (
            sha256_file(args.existing_generation_manifest_path)
            if args.existing_generation_manifest_path
            else None
        ),
    }
    initialize_or_validate_intent(output_dir, intent)

    device = torch.device("cuda")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    manifest = []
    prompts = []

    with torch.no_grad():
        for local_order, dataset_index in enumerate(indices):
            order = start_order + local_order
            item = dataset[dataset_index]
            output_name = f"real_{order:04d}.mp4"
            output_path = output_dir / output_name
            latent_frames = int(item["clean_latent"].shape[0])
            expected_rgb_frames = 1 + 4 * (latent_frames - 1)
            if not output_path.exists():
                latent = item["clean_latent"].unsqueeze(0).to(
                    device=device,
                    dtype=torch.bfloat16,
                )
                if args.streaming_decode:
                    stream_decode_to_video(vae, latent, str(output_path), fps=args.fps)
                else:
                    video = vae.decode_to_pixel(latent, use_cache=False)
                    frames = (
                        (video[0] * 0.5 + 0.5)
                        .clamp(0, 1)
                        .mul(255)
                        .permute(0, 2, 3, 1)
                        .cpu()
                    )
                    write_video_with_fallback(str(output_path), frames, fps=args.fps)
                    vae.model.clear_cache()
            validate_video(str(output_path), expected_rgb_frames, args.fps)
            manifest.append(
                {
                    "order": order,
                    "dataset_index": dataset_index,
                    "output_name": output_name,
                    "prompt": item["prompts"],
                }
            )
            prompts.append(item["prompts"])
            print(f"{local_order + 1}/{len(indices)} {output_path}", flush=True)

    write_jsonl(output_dir / "reference_manifest.jsonl", manifest)
    with open(output_dir / "reference_prompts.txt", "w", encoding="utf-8") as fp:
        for prompt in prompts:
            fp.write(normalize_prompt(prompt) + "\n")
    prompt_file_sha256 = hashlib.sha256(
        (output_dir / "reference_prompts.txt").read_bytes()
    ).hexdigest()
    generation_manifest = []
    for local_order, prompt in enumerate(prompts):
        order = start_order + local_order
        generation_manifest.append(
            {
                # This manifest is consumed with the new-only prompt file, so the
                # prompt index is local even though filenames/seeds continue at 256.
                "prompt_index": local_order,
                "sample_index": 0,
                "seed": args.seed + order,
                "output_name": f"fake_{order:04d}.mp4",
                "prompt": normalize_prompt(prompt),
                "prompt_file_sha256": prompt_file_sha256,
            }
        )
    write_jsonl(output_dir / "generation_manifest.jsonl", generation_manifest)

    if existing_reference_records:
        combined_reference = existing_reference_records + manifest
        combined_prompts = [
            normalize_prompt(record["prompt"]) for record in combined_reference
        ]
        combined_prompt_path = output_dir / "reference_prompts_combined.txt"
        with open(combined_prompt_path, "w", encoding="utf-8") as fp:
            for prompt in combined_prompts:
                fp.write(prompt + "\n")
        combined_prompt_sha256 = hashlib.sha256(
            combined_prompt_path.read_bytes()
        ).hexdigest()
        combined_generation = []
        for order, record in enumerate(
            existing_generation_records + generation_manifest
        ):
            combined_generation.append(
                {
                    "prompt_index": order,
                    "sample_index": 0,
                    "seed": int(record["seed"]),
                    "output_name": f"fake_{order:04d}.mp4",
                    "prompt": combined_prompts[order],
                    "prompt_file_sha256": combined_prompt_sha256,
                }
            )
        write_jsonl(
            output_dir / "reference_manifest_combined.jsonl",
            combined_reference,
        )
        write_jsonl(
            output_dir / "generation_manifest_combined.jsonl",
            combined_generation,
        )
    print(f"Wrote {output_dir / 'reference_manifest.jsonl'}")
    print(f"Wrote {output_dir / 'reference_prompts.txt'}")
    print(f"Wrote {output_dir / 'generation_manifest.jsonl'}")
    if existing_reference_records:
        print(
            f"Wrote combined {start_order + len(indices)}-sample manifests in "
            f"{output_dir}"
        )


if __name__ == "__main__":
    main()
