import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from einops import rearrange
from torchvision.io import write_video

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.config import load_config
from utils.misc import set_seed
from utils.prompt_embedding_cache import PromptEmbeddingLMDBCache


def sanitize_filename(text: str, max_length: int = 96) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    text = text[:max_length].strip("_")
    return text or "sample"


def raw_filename(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Prompt is empty; cannot build raw filename")
    if "/" in text or "\x00" in text:
        raise ValueError(f"Prompt contains unsupported filename characters: {text!r}")
    return text


def load_generator_state(checkpoint_path: str, use_ema: bool):
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if use_ema:
        if "generator_ema" not in state_dict:
            raise KeyError(
                f"--use_ema was requested but checkpoint has no generator_ema: {checkpoint_path}"
            )
        generator_state = state_dict["generator_ema"]
    elif "generator" in state_dict:
        generator_state = state_dict["generator"]
    elif "model" in state_dict:
        generator_state = state_dict["model"]
    elif state_dict and all(torch.is_tensor(value) for value in state_dict.values()):
        generator_state = state_dict
    else:
        raise KeyError(
            f"Checkpoint has no generator/model state dictionary: {checkpoint_path}"
        )
    fixed = {}
    for name, value in generator_state.items():
        if name.startswith("_fsdp_wrapped_module."):
            name = name.removeprefix("_fsdp_wrapped_module.")
        name = name.replace("._fsdp_wrapped_module.", ".")
        if name in fixed:
            raise KeyError(
                f"Checkpoint keys collide after FSDP normalization: {name}"
            )
        fixed[name] = value
    return fixed


class CachedPromptTextEncoder(torch.nn.Module):
    def __init__(self, cache_path: str):
        super().__init__()
        self.cache = PromptEmbeddingLMDBCache(cache_path)
        self._device = torch.device("cpu")
        self._dtype = torch.bfloat16

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = first
            elif isinstance(first, torch.dtype):
                dtype = first
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        return self

    def forward(self, text_prompts):
        return {
            "prompt_embeds": self.cache.get_batch(
                text_prompts,
                device=self._device,
                dtype=self._dtype,
            )
        }


def write_video_with_fallback(output_path: str, frames: torch.Tensor, fps: int):
    frames = frames.clamp(0, 255).to(torch.uint8)
    try:
        write_video(output_path, frames, fps=fps)
    except ImportError as exc:
        if "PyAV" not in str(exc):
            raise
        import imageio.v2 as imageio

        imageio.mimsave(output_path, frames.numpy(), fps=fps, macro_block_size=1)


def validate_video(path: str, expected_frames: int, expected_fps: int) -> None:
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open generated video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count != expected_frames:
        raise RuntimeError(
            f"{path}: found {frame_count} encoded frames, expected {expected_frames}"
        )
    if abs(fps - expected_fps) > 1e-3:
        raise RuntimeError(f"{path}: encoded fps={fps}, expected {expected_fps}")


def load_manifest(path: str, dataset: TextDataset, prompt_path: str):
    records = []
    prompt_file_sha256 = hashlib.sha256(
        Path(prompt_path).read_bytes()
    ).hexdigest()
    prompt_sample_pairs = set()
    with open(path, encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in (
                "prompt_index",
                "sample_index",
                "seed",
                "output_name",
                "prompt",
                "prompt_file_sha256",
            ):
                if key not in record:
                    raise ValueError(f"{path}:{line_number}: missing {key}")
            prompt_index = int(record["prompt_index"])
            if prompt_index < 0 or prompt_index >= len(dataset):
                raise IndexError(f"{path}:{line_number}: prompt_index={prompt_index} is out of range")
            if int(record["sample_index"]) < 0 or int(record["seed"]) < 0:
                raise ValueError(f"{path}:{line_number}: sample_index and seed must be non-negative")
            output_name = str(record["output_name"])
            if os.path.basename(output_name) != output_name or not output_name.endswith(".mp4"):
                raise ValueError(f"{path}:{line_number}: output_name must be a plain .mp4 filename")
            expected_batch = dataset[prompt_index]
            expected_prompt = expected_batch["prompts"]
            if record["prompt"] != expected_prompt:
                raise ValueError(f"{path}:{line_number}: prompt text does not match prompt_index")
            expected_extended_prompt = expected_batch.get("extended_prompts")
            if expected_extended_prompt is not None:
                if record.get("extended_prompt") != expected_extended_prompt:
                    raise ValueError(
                        f"{path}:{line_number}: extended_prompt does not match prompt_index"
                    )
            elif "extended_prompt" in record:
                raise ValueError(
                    f"{path}:{line_number}: manifest has extended_prompt but the "
                    "dataset has no extended-prompt file"
                )
            if record["prompt_file_sha256"] != prompt_file_sha256:
                raise ValueError(
                    f"{path}:{line_number}: prompt_file_sha256 does not match {prompt_path}"
                )
            pair = (prompt_index, int(record["sample_index"]))
            if pair in prompt_sample_pairs:
                raise ValueError(
                    f"{path}:{line_number}: duplicate prompt/sample pair {pair}"
                )
            prompt_sample_pairs.add(pair)
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    output_names = [str(record["output_name"]) for record in records]
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"Manifest contains duplicate output_name values: {path}")
    return records


def select_records_for_shard(records, shard_index: int, num_shards: int):
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards - 1}], got {shard_index}"
        )
    return records[shard_index::num_shards]


@torch.no_grad()
def stream_decode_to_video(vae, latents: torch.Tensor, output_path: str, fps: int):
    import imageio.v2 as imageio

    if latents.shape[0] != 1:
        raise ValueError("Streaming VAE decode currently requires batch size 1")
    vae.model.clear_cache()
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    frames_written = 0
    try:
        for latent_index in range(latents.shape[1]):
            pixels = vae.decode_to_pixel(
                latents[:, latent_index:latent_index + 1],
                use_cache=True,
            )
            frames = (
                (pixels[0] * 0.5 + 0.5)
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for frame in frames:
                writer.append_data(frame)
                frames_written += 1
    finally:
        writer.close()
        vae.model.clear_cache()

    expected_frames = 1 + 4 * (latents.shape[1] - 1)
    if frames_written != expected_frames:
        raise RuntimeError(
            f"Streaming VAE wrote {frames_written} frames; expected {expected_frames}"
        )
    return frames_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--extended_prompt_path", type=str, default="")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--prompt_embedding_cache_path", type=str, default="")
    parser.add_argument("--offload_generator_before_decode", action="store_true")
    parser.add_argument("--streaming_decode", action="store_true")
    parser.add_argument(
        "--naming",
        type=str,
        default="prompt_index",
        choices=["prompt", "index", "prompt_index", "raw_prompt", "raw_prompt_index"],
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    set_seed(args.seed)

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.num_output_frames < 1:
        raise ValueError("--num_output_frames must be positive")
    if args.streaming_decode and not args.offload_generator_before_decode:
        raise ValueError("--streaming_decode requires --offload_generator_before_decode")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard_index must be in [0, {args.num_shards - 1}], "
            f"got {args.shard_index}"
        )

    config = load_config(args.config_path)

    cached_text_encoder = (
        CachedPromptTextEncoder(args.prompt_embedding_cache_path)
        if args.prompt_embedding_cache_path
        else None
    )

    pipeline = CausalInferencePipeline(config, device=device, text_encoder=cached_text_encoder)

    generator_state = load_generator_state(args.checkpoint_path, use_ema=args.use_ema)
    pipeline.generator.load_state_dict(generator_state, strict=True, assign=True)
    del generator_state
    gc.collect()

    if pipeline.text_encoder is not None:
        pipeline.text_encoder.to(device=device, dtype=torch.bfloat16)
    pipeline.generator.to(device=device, dtype=torch.bfloat16)
    pipeline.vae.to(device=device, dtype=torch.bfloat16)

    dataset = TextDataset(
        prompt_path=args.prompt_path,
        extended_prompt_path=args.extended_prompt_path or None,
    )
    if args.num_samples_per_prompt <= 0:
        raise ValueError("--num_samples_per_prompt must be positive")

    os.makedirs(args.output_folder, exist_ok=True)

    if args.manifest_path:
        all_records = load_manifest(args.manifest_path, dataset, args.prompt_path)
        if args.limit > 0:
            all_records = all_records[:args.limit]
        expected_names = {
            str(record["output_name"]) for record in all_records
        }
        existing_names = {
            name
            for name in os.listdir(args.output_folder)
            if name.lower().endswith(".mp4")
        }
        extra_names = sorted(existing_names - expected_names)
        if extra_names:
            raise ValueError(
                f"Output folder contains videos outside the selected manifest: "
                f"{extra_names[:8]}"
            )
    else:
        num_prompts = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
        all_records = [
            {
                "prompt_index": idx,
                "sample_index": sample_idx,
                "seed": args.seed + idx * args.num_samples_per_prompt + sample_idx,
            }
            for idx in range(num_prompts)
            for sample_idx in range(args.num_samples_per_prompt)
        ]
    records = select_records_for_shard(
        all_records,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    for record in records:
        idx = int(record["prompt_index"])
        sample_idx = int(record["sample_index"])
        sample_seed = int(record["seed"])
        batch = dataset[idx]
        prompt = batch["prompts"]
        conditioned_prompt = batch.get("extended_prompts", prompt)
        safe_prompt = sanitize_filename(prompt)
        raw_prompt_name = None
        if args.naming in {"raw_prompt", "raw_prompt_index"}:
            raw_prompt_name = raw_filename(prompt)

        global_idx = idx * args.num_samples_per_prompt + sample_idx
        if "output_name" in record:
            filename = str(record["output_name"])
        else:
            if args.naming == "index":
                filename = f"{global_idx:04d}.mp4"
            elif args.naming == "prompt":
                suffix = f"-{sample_idx}" if args.num_samples_per_prompt > 1 else ""
                filename = f"{safe_prompt}{suffix}.mp4"
            elif args.naming == "raw_prompt":
                suffix = f"-{sample_idx}" if args.num_samples_per_prompt > 1 else ""
                filename = f"{raw_prompt_name}{suffix}.mp4"
            elif args.naming == "raw_prompt_index":
                filename = f"{raw_prompt_name}-{sample_idx}.mp4"
            else:
                filename = f"{safe_prompt}-{global_idx:04d}.mp4"

        output_path = os.path.join(args.output_folder, filename)
        expected_frames = 1 + 4 * (args.num_output_frames - 1)
        if os.path.exists(output_path):
            validate_video(output_path, expected_frames, args.fps)
            print(f"Skipping existing {output_path}", flush=True)
            continue

        # Reset both the private initial-noise generator and the process-global
        # RNG used by intermediate re-noising. This makes every manifest record
        # shard-order independent and exactly paired across model checkpoints.
        set_seed(sample_seed)
        if args.offload_generator_before_decode:
            pipeline.text_encoder.to(device)
            pipeline.generator.to(device)
            torch.cuda.empty_cache()

        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed)
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[conditioned_prompt],
            return_latents=True,
            return_video=not args.offload_generator_before_decode,
        )
        if args.offload_generator_before_decode:
            pipeline.text_encoder.to("cpu")
            pipeline.generator.to("cpu")
            torch.cuda.empty_cache()
            if args.streaming_decode:
                frames_written = stream_decode_to_video(
                    pipeline.vae,
                    latents,
                    output_path,
                    fps=args.fps,
                )
                print(
                    f"Wrote {output_path} ({frames_written} frames, seed={sample_seed})",
                    flush=True,
                )
                validate_video(output_path, expected_frames, args.fps)
                del latents, sampled_noise
                torch.cuda.empty_cache()
                continue
            else:
                video = pipeline.vae.decode_to_pixel(latents, use_cache=False)
                video = (video * 0.5 + 0.5).clamp(0, 1)

        video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
        write_video_with_fallback(output_path, video[0], fps=args.fps)
        validate_video(output_path, expected_frames, args.fps)
        print(f"Wrote {output_path} (seed={sample_seed})", flush=True)
        pipeline.vae.model.clear_cache()

    if args.num_shards == 1:
        done_name = "export.done"
    else:
        done_name = (
            f"export.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.done"
        )
    done_path = os.path.join(args.output_folder, done_name)
    with open(done_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "checkpoint_path": os.path.abspath(args.checkpoint_path),
                "weight_source": "generator_ema" if args.use_ema else "generator",
                "use_ema": bool(args.use_ema),
                "config_path": os.path.abspath(args.config_path),
                "manifest_path": (
                    os.path.abspath(args.manifest_path) if args.manifest_path else None
                ),
                "num_videos": len(records),
                "num_total_videos": len(all_records),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "latent_frames_per_video": args.num_output_frames,
                "rgb_frames_per_video": 1 + 4 * (args.num_output_frames - 1),
                "fps": args.fps,
                "seed_scope": (
                    "Each manifest seed controls both initial latent noise and "
                    "multi-step intermediate re-noising."
                ),
            },
            fp,
            indent=2,
            sort_keys=True,
        )
        fp.write("\n")
    print(f"Wrote {done_path}", flush=True)


if __name__ == "__main__":
    main()
