#!/usr/bin/env python3
"""Train one arm of the paired curvature experiment with consistency distillation.

The online/raw generator is the only model saved for evaluation.  A second
generator is maintained as an in-training EMA target, as required by the
consistency objective; it is never exported as ``generator_ema`` and is never
used for VBench inference.
"""

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
from scripts.export_videos import (  # noqa: E402
    CachedPromptTextEncoder,
    load_generator_state,
)
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


def normalize_fsdp_name(name: str) -> str:
    if name.startswith("_fsdp_wrapped_module."):
        name = name.removeprefix("_fsdp_wrapped_module.")
    return name.replace("._fsdp_wrapped_module.", ".")


@torch.no_grad()
def update_ema_target(online, target, decay: float) -> None:
    """Update an identically wrapped on-device target without CPU round trips."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"target EMA decay must be in [0, 1), got {decay}")
    online_parameters = list(online.named_parameters())
    target_parameters = list(target.named_parameters())
    if len(online_parameters) != len(target_parameters):
        raise ValueError("Online and target generators have different parameter counts")
    for (online_name, online_value), (target_name, target_value) in zip(
        online_parameters, target_parameters
    ):
        normalized_online = normalize_fsdp_name(online_name)
        normalized_target = normalize_fsdp_name(target_name)
        if normalized_online != normalized_target:
            raise ValueError(
                "Online/target parameter order mismatch: "
                f"{normalized_online!r} != {normalized_target!r}"
            )
        if online_value.shape != target_value.shape:
            raise ValueError(
                f"Online/target shape mismatch for {normalized_online}: "
                f"{tuple(online_value.shape)} != {tuple(target_value.shape)}"
            )
        target_value.mul_(decay).add_(
            online_value.detach().to(dtype=target_value.dtype),
            alpha=1.0 - decay,
        )


def adjacent_pair_index(step: int, num_pairs: int) -> int:
    """Use a deterministic low-to-high sweep for exact arm balance."""
    if step < 1 or num_pairs < 1:
        raise ValueError("step and num_pairs must be positive")
    return num_pairs - 1 - ((step - 1) % num_pairs)


def select_adjacent_training_pair(
    trajectory: torch.Tensor,
    selected_timesteps,
    *,
    pair_index: int,
    training_num_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Select ``x_t, x_s, clean`` from ``[B,points,F,C,H,W]``."""
    if trajectory.ndim != 6:
        raise ValueError(f"Expected a batched trajectory, got {tuple(trajectory.shape)}")
    timesteps = list(selected_timesteps)
    if len(timesteps) != trajectory.shape[1] or timesteps[-1] is not None:
        raise ValueError("Trajectory points and selected_timesteps are inconsistent")
    num_ode_points = len(timesteps) - 1
    num_pairs = num_ode_points - 1
    if pair_index < 0 or pair_index >= num_pairs:
        raise IndexError(f"pair_index={pair_index} is outside [0, {num_pairs})")
    if training_num_frames < 1 or training_num_frames > trajectory.shape[2]:
        raise ValueError(
            f"training_num_frames={training_num_frames} is outside "
            f"[1, {trajectory.shape[2]}]"
        )
    high_timestep = float(timesteps[pair_index])
    low_timestep = float(timesteps[pair_index + 1])
    if not high_timestep > low_timestep:
        raise ValueError("Adjacent trajectory timesteps must be strictly descending")
    frame_slice = slice(0, training_num_frames)
    return (
        trajectory[:, pair_index, frame_slice],
        trajectory[:, pair_index + 1, frame_slice],
        trajectory[:, -1, frame_slice],
        high_timestep,
        low_timestep,
    )


def load_raw_initial_state(module, checkpoint_path: Path) -> None:
    """Load the exact raw state before applying the repository's FSDP wrapper."""
    initial_state = load_generator_state(str(checkpoint_path), use_ema=False)
    module.load_state_dict(initial_state, strict=True)
    del initial_state
    gc.collect()


def build_generator(config, checkpoint_path: Path, *, trainable: bool):
    generator = WanDiffusionWrapper(
        **OmegaConf.to_container(config.model_kwargs, resolve=True),
        is_causal=True,
    )
    generator.model.num_frame_per_block = 1
    generator.model.requires_grad_(trainable)
    if trainable and bool(config.gradient_checkpointing):
        generator.enable_gradient_checkpointing()
    load_raw_initial_state(generator, checkpoint_path)
    fsdp_kwargs = get_fsdp_wrap_kwargs(
        config,
        "generator",
        default_transformer_modules=["causal_wan_block"],
    )
    return fsdp_wrap(generator, **fsdp_kwargs)


def save_raw_checkpoint(generator, output_dir: Path, step: int, metadata: dict) -> None:
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
                "contains_generator_ema": False,
            },
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    print(f"Saved raw/no-EMA checkpoint: {checkpoint_path}", flush=True)


def validate_config(config) -> None:
    positive_integer_fields = (
        "max_steps",
        "batch_size",
        "training_num_frames",
        "checkpoint_interval",
        "print_interval",
        "gc_interval",
    )
    for field in positive_integer_fields:
        if int(getattr(config, field)) < 1:
            raise ValueError(f"{field} must be positive")
    if int(config.seed) < 0:
        raise ValueError("seed must be non-negative")
    if int(config.num_frame_per_block) != 1:
        raise ValueError("The controlled curvature experiment must remain framewise")
    if not 0.0 <= float(config.target_ema_decay) < 1.0:
        raise ValueError("target_ema_decay must be in [0, 1)")
    if str(config.generator_fsdp_wrap_strategy) == "none":
        raise ValueError("Two-generator CD training requires sharded generators")
    if (
        not getattr(config, "prompt_embedding_cache_path", "")
        and str(config.text_encoder_fsdp_wrap_strategy) == "none"
    ):
        raise ValueError("Memory-efficient initialization requires a sharded text encoder")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--intervention", required=True, choices=["curved", "rectified"])
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt_embedding_cache_path", default="")
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
    if args.prompt_embedding_cache_path:
        config.prompt_embedding_cache_path = str(
            Path(args.prompt_embedding_cache_path).resolve()
        )
    validate_config(config)

    dataset_manifest_path = data_path / MANIFEST_NAME
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    source_checkpoint = Path(dataset_manifest["generator_ckpt"]).resolve()
    if source_checkpoint != generator_ckpt:
        raise ValueError(
            "The ODE source checkpoint and training initialization must be identical: "
            f"dataset={source_checkpoint}, argument={generator_ckpt}"
        )
    if dataset_manifest.get("use_ema") is not False:
        raise ValueError("Curvature CD requires a raw/no-EMA trajectory dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {output_dir}")
    prompt_cache_path = str(getattr(config, "prompt_embedding_cache_path", ""))
    if prompt_cache_path and not (Path(prompt_cache_path) / "data.mdb").is_file():
        raise FileNotFoundError(Path(prompt_cache_path) / "data.mdb")

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
        atomic_write_json(
            output_dir / "run.intent.json",
            {
                "schema_version": 2,
                "experiment": "curvature_adjacent_consistency_distillation",
                "intervention": args.intervention,
                "config": OmegaConf.to_container(config, resolve=True),
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "data_path": str(data_path),
                "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
                "generator_ckpt": str(generator_ckpt),
                "generator_checkpoint_size_bytes": generator_ckpt.stat().st_size,
                "seed": int(config.seed),
                "max_steps": int(config.max_steps),
                "world_size": world_size,
                "training_num_frames": int(config.training_num_frames),
                "framewise": True,
                "training_objective": "adjacent_state_consistency_l2",
                "target_network": "in_training_ema_stop_gradient",
                "target_ema_decay": float(config.target_ema_decay),
                "evaluation_use_ema": False,
                "use_ema": False,
                "weight_source": "generator",
            },
        )
    dist.barrier()

    dataset = CurvatureTrajectoryDataset(str(data_path), args.intervention)
    training_num_frames = int(config.training_num_frames)
    if training_num_frames > dataset.shape[2]:
        raise ValueError(
            f"Requested {training_num_frames} frames but dataset has {dataset.shape[2]}"
        )
    ode_timesteps = [float(value) for value in dataset.selected_timesteps[:-1]]
    num_pairs = len(ode_timesteps) - 1
    if num_pairs < 1 or ode_timesteps[-1] != 0.0:
        raise ValueError(f"Expected a trajectory ending at timestep zero: {ode_timesteps}")
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

    generator = build_generator(config, generator_ckpt, trainable=True)
    target_generator = build_generator(config, generator_ckpt, trainable=False)
    generator.eval()
    target_generator.eval()

    if prompt_cache_path:
        text_encoder = CachedPromptTextEncoder(prompt_cache_path).to(
            device=device, dtype=dtype
        )
    else:
        text_encoder = WanTextEncoder().eval().requires_grad_(False)
        text_encoder_kwargs = get_fsdp_wrap_kwargs(config, "text_encoder")
        text_encoder = fsdp_wrap(text_encoder, **text_encoder_kwargs)
    text_encoder.eval()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=float(config.lr),
        betas=(float(config.beta1), float(config.beta2)),
        weight_decay=float(config.weight_decay),
    )
    metadata = {
        "schema_version": 2,
        "experiment": "curvature_adjacent_consistency_distillation",
        "intervention": args.intervention,
        "initial_generator_ckpt": str(generator_ckpt),
        "data_path": str(data_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "seed": int(config.seed),
        "world_size": world_size,
        "framewise": True,
        "training_num_frames": training_num_frames,
        "training_timesteps": ode_timesteps,
        "adjacent_pairs": [
            [ode_timesteps[index], ode_timesteps[index + 1]]
            for index in range(num_pairs)
        ],
        "pair_schedule": "deterministic_low_to_high_sweep",
        "training_objective": "adjacent_state_consistency_l2",
        "target_network": "in_training_ema_stop_gradient",
        "target_ema_decay": float(config.target_ema_decay),
        "evaluation_use_ema": False,
    }
    metrics_path = output_dir / "metrics.jsonl"
    sample_order_digest = hashlib.sha256()
    start_time = time.time()
    for step in range(1, int(config.max_steps) + 1):
        step_start = time.time()
        batch, iterator, epoch = next_batch(iterator, loader, sampler, epoch)
        trajectory = batch["trajectory"].to(device=device, dtype=dtype)
        prompts = list(batch["prompt"])
        local_row_indices = batch["row_index"].to(device=device, dtype=torch.long)
        gathered_row_indices = [torch.empty_like(local_row_indices) for _ in range(world_size)]
        dist.all_gather(gathered_row_indices, local_row_indices)
        global_row_indices = torch.cat(gathered_row_indices).cpu().tolist()
        pair_index = adjacent_pair_index(step, num_pairs)
        high_state, low_state, clean_conditioning, high_t, low_t = (
            select_adjacent_training_pair(
                trajectory,
                dataset.selected_timesteps,
                pair_index=pair_index,
                training_num_frames=training_num_frames,
            )
        )
        batch_size, num_frames = high_state.shape[:2]
        high_timestep = torch.full(
            (batch_size, num_frames), high_t, device=device, dtype=torch.float32
        )
        low_timestep = torch.full(
            (batch_size, num_frames), low_t, device=device, dtype=torch.float32
        )

        with torch.no_grad():
            conditioning = text_encoder(text_prompts=prompts)
            if low_t == 0.0:
                target = low_state
                target_kind = "boundary_identity"
            else:
                _, target = target_generator(
                    noisy_image_or_video=low_state,
                    conditional_dict=conditioning,
                    timestep=low_timestep,
                    clean_x=clean_conditioning,
                )
                target = target.detach()
                target_kind = "ema_bootstrap"

        optimizer.zero_grad(set_to_none=True)
        _, prediction = generator(
            noisy_image_or_video=high_state,
            conditional_dict=conditioning,
            timestep=high_timestep,
            clean_x=clean_conditioning,
        )
        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
        loss.backward()
        grad_norm = generator.clip_grad_norm_(float(config.max_grad_norm))
        optimizer.step()
        update_ema_target(generator, target_generator, float(config.target_ema_decay))

        reduced = torch.stack((loss.detach().float(), grad_norm.detach().float()))
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
        if is_main:
            sample_order_digest.update(
                f"{step}:".encode("ascii")
                + ",".join(str(value) for value in global_row_indices).encode("ascii")
                + b"\n"
            )
            record = {
                "step": step,
                "epoch": epoch,
                "global_row_indices": global_row_indices,
                "pair_index": pair_index,
                "high_timestep": high_t,
                "low_timestep": low_t,
                "target_kind": target_kind,
                "loss": float(reduced[0].item()),
                "grad_norm": float(reduced[1].item()),
                "seconds": time.time() - step_start,
                "elapsed_seconds": time.time() - start_time,
                "intervention": args.intervention,
                "evaluation_use_ema": False,
            }
            append_jsonl(metrics_path, record)
            if step <= 10 or step % int(config.print_interval) == 0:
                print(
                    f"[{args.intervention}] step={step}/{int(config.max_steps)} "
                    f"pair={high_t:g}->{low_t:g} target={target_kind} "
                    f"loss={record['loss']:.8f} grad_norm={record['grad_norm']:.6f} "
                    f"seconds={record['seconds']:.2f}",
                    flush=True,
                )

        if step % int(config.checkpoint_interval) == 0 or step == int(config.max_steps):
            save_raw_checkpoint(generator, output_dir, step, metadata)
            dist.barrier()
            torch.cuda.empty_cache()
        if step % int(config.gc_interval) == 0:
            gc.collect()

    if is_main:
        final_checkpoint = (
            output_dir / f"checkpoint_model_{int(config.max_steps):06d}" / "model.pt"
        )
        atomic_write_json(
            output_dir / "training.done",
            {
                **metadata,
                "final_step": int(config.max_steps),
                "final_checkpoint": str(final_checkpoint),
                "elapsed_seconds": time.time() - start_time,
                "status": "complete",
                "use_ema": False,
                "weight_source": "generator",
                "contains_generator_ema": False,
                "global_sample_order_sha256": sample_order_digest.hexdigest(),
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
