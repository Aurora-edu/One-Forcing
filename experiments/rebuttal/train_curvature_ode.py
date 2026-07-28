#!/usr/bin/env python3
"""Train one arm of the paired curvature intervention with raw weights only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.curvature_intervention import (  # noqa: E402
    CurvatureTrajectoryDataset,
    MANIFEST_NAME,
)
from scripts.export_videos import load_generator_state  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.distributed import (  # noqa: E402
    fsdp_state_dict,
    fsdp_wrap,
    get_fsdp_wrap_kwargs,
    launch_distributed_job,
)
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_fixed_seed(seed: int, rank: int) -> None:
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)


def atomic_write_json(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def next_batch(iterator, loader, sampler, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def save_checkpoint(generator, output_dir: Path, step: int, metadata: dict) -> None:
    generator_state = fsdp_state_dict(generator)
    if dist.get_rank() != 0:
        return
    checkpoint_dir = output_dir / f"checkpoint_model_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "model.pt"
    temporary = checkpoint_dir / f".model.pt.{os.getpid()}.tmp"
    torch.save(
        {
            "step": step,
            "generator": generator_state,
            "experiment_metadata": {
                **metadata,
                "checkpoint_step": step,
                "use_ema": False,
                "weight_source": "generator",
            },
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    print(f"Saved raw/no-EMA checkpoint: {checkpoint_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--intervention",
        required=True,
        choices=["curved", "rectified"],
    )
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    data_path = Path(args.data_path).resolve()
    generator_ckpt = Path(args.generator_ckpt).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (config_path, data_path / "data.mdb", generator_ckpt):
        if not path.exists():
            raise FileNotFoundError(path)

    config = load_config(str(config_path))
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.seed is not None:
        config.seed = args.seed
    if int(config.max_steps) < 1 or int(config.seed) < 0:
        raise ValueError("max_steps must be positive and seed must be non-negative")
    if int(config.batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(config.num_frame_per_block) != 1:
        raise ValueError("The controlled experiment must remain framewise")

    dataset_manifest_path = data_path / MANIFEST_NAME
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    source_checkpoint = Path(dataset_manifest["generator_ckpt"]).resolve()
    if source_checkpoint != generator_ckpt:
        raise ValueError(
            "The ODE source checkpoint and training initialization must be identical: "
            f"dataset={source_checkpoint}, argument={generator_ckpt}"
        )
    if dataset_manifest.get("use_ema") is not False:
        raise ValueError("Curvature training requires a raw/no-EMA dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a new arm with existing output: {output_dir}"
        )

    launch_distributed_job()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    is_main = rank == 0
    dtype = torch.bfloat16 if bool(config.mixed_precision) else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_fixed_seed(int(config.seed), rank)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved = OmegaConf.to_container(config, resolve=True)
        atomic_write_json(
            output_dir / "run.intent.json",
            {
                "schema_version": 1,
                "intervention": args.intervention,
                "config": resolved,
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "data_path": str(data_path),
                "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
                "generator_ckpt": str(generator_ckpt),
                "generator_checkpoint_size_bytes": generator_ckpt.stat().st_size,
                "seed": int(config.seed),
                "max_steps": int(config.max_steps),
                "world_size": world_size,
                "use_ema": False,
                "weight_source": "generator",
            },
        )
    dist.barrier()

    dataset = CurvatureTrajectoryDataset(str(data_path), args.intervention)
    global_batch = int(config.batch_size) * world_size
    if len(dataset) < global_batch:
        raise ValueError(
            f"Dataset has {len(dataset)} rows but global batch size is {global_batch}"
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config.seed),
        drop_last=True,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        sampler=sampler,
        num_workers=int(config.dataloader_num_workers),
        pin_memory=True,
        drop_last=True,
    )
    iterator = iter(loader)
    epoch = 0

    generator = WanDiffusionWrapper(
        **OmegaConf.to_container(config.model_kwargs, resolve=True),
        is_causal=True,
    )
    generator.model.num_frame_per_block = 1
    generator.model.requires_grad_(True)
    if bool(config.gradient_checkpointing):
        generator.enable_gradient_checkpointing()
    initial_state = load_generator_state(str(generator_ckpt), use_ema=False)
    generator.load_state_dict(initial_state, strict=True)
    del initial_state
    gc.collect()

    text_encoder = WanTextEncoder().eval().requires_grad_(False)
    generator = fsdp_wrap(
        generator,
        **get_fsdp_wrap_kwargs(
            config,
            "generator",
            default_transformer_modules=["causal_wan_block"],
        ),
    )
    text_encoder = fsdp_wrap(
        text_encoder,
        **get_fsdp_wrap_kwargs(config, "text_encoder"),
    )
    if str(config.text_encoder_fsdp_wrap_strategy) == "none":
        text_encoder = text_encoder.to(device=device, dtype=dtype)
    generator.eval()
    text_encoder.eval()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=float(config.lr),
        betas=(float(config.beta1), float(config.beta2)),
        weight_decay=float(config.weight_decay),
    )
    selected_input_timesteps = torch.tensor(
        dataset.selected_timesteps[:4], device=device, dtype=torch.float32
    )
    if selected_input_timesteps.numel() != 4:
        raise ValueError("Expected exactly four causal ODE training inputs")

    metadata = {
        "schema_version": 1,
        "experiment": "curvature_controlled_causal_intervention",
        "intervention": args.intervention,
        "initial_generator_ckpt": str(generator_ckpt),
        "data_path": str(data_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "seed": int(config.seed),
        "framewise": True,
        "training_input_timesteps": selected_input_timesteps.cpu().tolist(),
    }
    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.time()
    for step in range(1, int(config.max_steps) + 1):
        step_start = time.time()
        batch, iterator, epoch = next_batch(iterator, loader, sampler, epoch)
        trajectory = batch["trajectory"].to(device=device, dtype=dtype)
        prompts = list(batch["prompt"])
        batch_size, _, num_frames = trajectory.shape[:3]

        with torch.no_grad():
            conditioning = text_encoder(text_prompts=prompts)
            input_index = torch.randint(
                0,
                4,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            row_index = torch.arange(batch_size, device=device)
            noisy_input = trajectory[row_index, input_index]
            timestep = selected_input_timesteps[input_index].unsqueeze(1).repeat(
                1, num_frames
            )
            target = trajectory[:, -2]
            clean_conditioning = trajectory[:, -1]

        optimizer.zero_grad(set_to_none=True)
        _, prediction = generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditioning,
            timestep=timestep,
            clean_x=clean_conditioning,
        )
        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
        loss.backward()
        grad_norm = generator.clip_grad_norm_(float(config.max_grad_norm))
        optimizer.step()

        reduced = torch.stack(
            (loss.detach().float(), grad_norm.detach().float())
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
        if is_main:
            record = {
                "step": step,
                "epoch": epoch,
                "loss": float(reduced[0].item()),
                "grad_norm": float(reduced[1].item()),
                "seconds": time.time() - step_start,
                "elapsed_seconds": time.time() - start_time,
                "intervention": args.intervention,
                "use_ema": False,
            }
            append_jsonl(metrics_path, record)
            if step <= 10 or step % int(config.print_interval) == 0:
                print(
                    f"[{args.intervention}] step={step}/{int(config.max_steps)} "
                    f"loss={record['loss']:.8f} grad_norm={record['grad_norm']:.6f} "
                    f"seconds={record['seconds']:.2f}",
                    flush=True,
                )

        if step % int(config.checkpoint_interval) == 0 or step == int(config.max_steps):
            save_checkpoint(generator, output_dir, step, metadata)
            dist.barrier()
            torch.cuda.empty_cache()
        if step % int(config.gc_interval) == 0:
            gc.collect()

    if is_main:
        atomic_write_json(
            output_dir / "training.done",
            {
                **metadata,
                "final_step": int(config.max_steps),
                "final_checkpoint": str(
                    output_dir
                    / f"checkpoint_model_{int(config.max_steps):06d}"
                    / "model.pt"
                ),
                "elapsed_seconds": time.time() - start_time,
                "status": "complete",
                "use_ema": False,
                "weight_source": "generator",
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
