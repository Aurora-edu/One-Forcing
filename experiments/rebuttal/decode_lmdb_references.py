#!/usr/bin/env python3
import argparse
import hashlib
import json
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
    args = parser.parse_args()

    if args.num_videos < 2 or args.seed < 0 or args.fps < 1:
        raise ValueError("num_videos must be >=2, seed non-negative, and fps positive")
    dataset = CleanLatentLMDBDataset(args.lmdb_path, readahead=False)
    if args.num_videos > len(dataset):
        raise ValueError(f"Requested {args.num_videos} videos from dataset of size {len(dataset)}")
    indices = random.Random(args.seed).sample(range(len(dataset)), args.num_videos)

    device = torch.device("cuda")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)
    vae.eval().requires_grad_(False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    prompts = []

    with torch.no_grad():
        for order, dataset_index in enumerate(indices):
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
            print(f"{order + 1}/{len(indices)} {output_path}", flush=True)

    with open(output_dir / "reference_manifest.jsonl", "w", encoding="utf-8") as fp:
        for record in manifest:
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with open(output_dir / "reference_prompts.txt", "w", encoding="utf-8") as fp:
        for prompt in prompts:
            fp.write(prompt.replace("\n", " ").strip() + "\n")
    prompt_file_sha256 = hashlib.sha256(
        (output_dir / "reference_prompts.txt").read_bytes()
    ).hexdigest()
    with open(output_dir / "generation_manifest.jsonl", "w", encoding="utf-8") as fp:
        for prompt_index, prompt in enumerate(prompts):
            record = {
                "prompt_index": prompt_index,
                "sample_index": 0,
                "seed": args.seed + prompt_index,
                "output_name": f"fake_{prompt_index:04d}.mp4",
                "prompt": prompt.replace("\n", " ").strip(),
                "prompt_file_sha256": prompt_file_sha256,
            }
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {output_dir / 'reference_manifest.jsonl'}")
    print(f"Wrote {output_dir / 'reference_prompts.txt'}")
    print(f"Wrote {output_dir / 'generation_manifest.jsonl'}")


if __name__ == "__main__":
    main()
