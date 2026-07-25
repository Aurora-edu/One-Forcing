#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import CausalInferencePipeline
from scripts.export_videos import CachedPromptTextEncoder, load_generator_state
from utils.config import load_config
from utils.misc import set_seed


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def first_line(path):
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                return line.rstrip("\n")
    raise ValueError(f"No non-empty prompt in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile first-block, steady-state, diffusion, and optional VAE latency."
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--extended_prompt_path", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--prompt_embedding_cache_path", default="")
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--include_vae", action="store_true")
    args = parser.parse_args()

    if args.num_output_frames < 1 or args.warmup < 0 or args.trials < 1:
        raise ValueError("frames/trials must be positive and warmup must be non-negative")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    device = torch.device("cuda")
    config = load_config(args.config_path)
    prompt = (
        first_line(args.extended_prompt_path)
        if args.extended_prompt_path
        else first_line(args.prompt_path)
    )

    cached_encoder = (
        CachedPromptTextEncoder(args.prompt_embedding_cache_path)
        if args.prompt_embedding_cache_path
        else None
    )
    pipeline = CausalInferencePipeline(config, device=device, text_encoder=cached_encoder)
    state = load_generator_state(args.checkpoint_path, use_ema=args.use_ema)
    pipeline.generator.load_state_dict(state, strict=True, assign=True)
    del state

    if pipeline.text_encoder is not None:
        pipeline.text_encoder.to(device=device, dtype=torch.bfloat16)
    pipeline.generator.to(device=device, dtype=torch.bfloat16)
    pipeline.generator.eval().requires_grad_(False)
    if args.include_vae:
        pipeline.vae.to(device=device, dtype=torch.bfloat16)
        pipeline.vae.eval().requires_grad_(False)

    def make_noise(trial_index):
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + trial_index)
        return torch.randn(
            [1, args.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )

    for trial_index in range(args.warmup):
        set_seed(args.seed + trial_index)
        pipeline.inference(
            noise=make_noise(trial_index),
            text_prompts=[prompt],
            profile=False,
            return_video=args.include_vae,
        )
        if args.include_vae:
            pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()

    trials = []
    for trial_index in range(args.trials):
        set_seed(args.seed + args.warmup + trial_index)
        pipeline.inference(
            noise=make_noise(args.warmup + trial_index),
            text_prompts=[prompt],
            profile=True,
            return_video=args.include_vae,
        )
        if pipeline.last_profile is None:
            raise RuntimeError("Pipeline did not produce profiling data")
        trials.append(dict(pipeline.last_profile))
        if args.include_vae:
            pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()

    metric_keys = (
        "prompt_encoding_ms",
        "initialization_ms",
        "diffusion_ms",
        "vae_ms",
        "total_ms",
        "first_block_ms",
        "steady_block_mean_ms",
    )
    summary = {key: summarize([trial[key] for trial in trials]) for key in metric_keys}
    decoded_frames = trials[0]["decoded_frames"]
    latent_frames = trials[0]["latent_frames"]
    mean_diffusion_seconds = summary["diffusion_ms"]["mean"] / 1000.0
    mean_total_seconds = summary["total_ms"]["mean"] / 1000.0
    throughput = {
        "latent_frames_per_second_diffusion": latent_frames / mean_diffusion_seconds,
        "decoded_frames_per_second_diffusion": decoded_frames / mean_diffusion_seconds,
        "decoded_frames_per_second_total_profiled": decoded_frames / mean_total_seconds,
    }
    if not args.include_vae:
        throughput["decoded_frames_per_second_total_profiled"] = None
    summary["throughput"] = throughput

    result = {
        "schema_version": 1,
        "config_path": os.path.abspath(args.config_path),
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "schedule": {
            "rollout_schedule": str(getattr(config, "rollout_schedule", "fixed")),
            "denoising_step_list": list(config.denoising_step_list),
            "first_frame_denoising_step_list": (
                list(config.first_frame_denoising_step_list)
                if hasattr(config, "first_frame_denoising_step_list")
                else None
            ),
        },
        "warmup": args.warmup,
        "num_trials": args.trials,
        "include_vae": args.include_vae,
        "latency_scope": {
            "included": [
                "prompt encoding",
                "output/cache initialization",
                "denoising and clean-context cache updates",
                "VAE decode when --include_vae is set",
            ],
            "excluded": ["model/checkpoint loading", "MP4 encoding and disk I/O"],
        },
        "seed": args.seed,
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "summary": summary,
        "trials": trials,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
