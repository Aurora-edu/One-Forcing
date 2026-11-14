#!/usr/bin/env python3
"""One GPU, one pinned official implementation, deterministic manifest sharding."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

from official_baseline_common import (
    LOCK, adapted_pipeline_source, check_completed_video, check_source,
    json_write, load_config, select_weights, sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    intent = json.loads(args.intent.read_text())
    intent_hash = sha256(args.intent)
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard index")
    root = args.intent.parent
    model = intent["model"]
    source_root = Path(intent["source_root"])
    check_source(source_root, model)
    for path, expected in intent["evaluation_code_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Evaluation code changed since launch: {path}")
    rows = intent["records"][args.shard::args.shards]
    os.chdir(intent["runtime_root"])
    # Import official utils/wan/pipeline, never this project's model implementation.
    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torchvision.io import write_video

    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference is required; CPU validation does not generate Wan videos")
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    config = load_config(source_root, model, intent["schedule"])
    module = types.ModuleType("official_baseline_pipeline")
    source_path = source_root / "pipeline/causal_inference.py"
    module.__file__ = str(source_path)
    adapted = adapted_pipeline_source(source_path.read_text(), model)
    exec(compile(adapted, str(source_path), "exec"), module.__dict__)
    import utils.wan_wrapper as official_wrapper
    if not Path(official_wrapper.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError("Wrong model implementation imported")
    pipeline = module.CausalInferencePipeline(config, device=torch.device("cuda:0"))
    checkpoint = torch.load(intent["checkpoint"], map_location="cpu", weights_only=True)
    pipeline.generator.load_state_dict(select_weights(checkpoint, intent["weight_key"]), strict=True)
    del checkpoint
    pipeline = pipeline.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    pipeline.text_encoder.to(device="cuda:0")
    pipeline.generator.to(device="cuda:0")
    pipeline.vae.to(device="cuda:0")
    # No TF32/compile/quantization/attention-sink changes to the native implementation.
    environment = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "model_wrapper": official_wrapper.__file__,
        "weight_key": intent["weight_key"], "strict_weight_load": True,
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "warped_main_steps": (
            pipeline.denoising_step_list[:1].tolist()
            if model == "self_forcing" and intent["schedule"] == "ffe"
            else pipeline.denoising_step_list.tolist()
        ),
        "warped_first_steps": (
            pipeline.denoising_step_list.tolist() if model == "self_forcing"
            else pipeline.denoising_step_list_first_chunk.tolist()
        ),
    }
    json_write(root / "audit" / f"environment.shard{args.shard}.json", environment)
    (root / "audit" / f"pipeline.shard{args.shard}.py").write_text(adapted)
    for index, record in enumerate(rows):
        video_path = root / "videos" / record["output_name"]
        meta_path = root / "metadata" / (record["output_name"] + ".json")
        if check_completed_video(video_path, meta_path, record, intent_hash):
            print(f"resume: {record['output_name']}", flush=True)
            continue
        seed = record["seed"]
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        start = time.monotonic()
        noise = torch.randn([1, 21, 16, 60, 104], device="cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            video = pipeline.inference(noise=noise, text_prompts=[record["extended_prompt"]])
        if tuple(video.shape) != (1, 81, 3, 480, 832) or not torch.isfinite(video).all():
            raise RuntimeError(f"Invalid decoded video for seed {seed}: {tuple(video.shape)}")
        frames = (video[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu()
        temporary = video_path.with_name(video_path.stem + ".partial.mp4")
        # Same torchvision writer defaults as both official inference.py files.
        write_video(str(temporary), frames, fps=16)
        temporary.replace(video_path)
        if record["sample_index"] == 0 and record["prompt_index"] < 10:
            from PIL import Image, ImageDraw
            sheet = Image.new("RGB", (416 * 4, 240 * 3), "white")
            for slot, frame_index in enumerate([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 80]):
                tile = Image.fromarray(frames[frame_index].numpy()).resize((416, 240))
                ImageDraw.Draw(tile).text((5, 5), f"frame {frame_index}", fill="red")
                sheet.paste(tile, ((slot % 4) * 416, (slot // 4) * 240))
            sheet.save(root / "previews" / f"prompt{record['prompt_index']:04d}.jpg")
        pipeline.vae.model.clear_cache()
        info = {
            "record": record, "intent_sha256": intent_hash,
            "video_sha256": sha256(video_path), "frames": 81, "fps": 16,
            "resolution": [832, 480], "seconds": time.monotonic() - start,
        }
        json_write(meta_path, info)
        print(f"[{model} shard {args.shard}] {index + 1}/{len(rows)} seed={seed} {info['seconds']:.2f}s {video_path}", flush=True)
        del video, frames, noise
    json_write(root / f"shard{args.shard}.done.json", {"intent_sha256": intent_hash, "records": len(rows)})


if __name__ == "__main__":
    main()
