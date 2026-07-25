#!/usr/bin/env python3
"""Fail-fast checks for a rebuttal training run on a new machine."""

import argparse
import json
import os
import sys
from pathlib import Path

import lmdb
import torch
import torchvision
import transformers
import diffusers
import accelerate
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config
from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from utils.prompt_embedding_cache import prompt_cache_key
from experiments.rebuttal.build_prompt_cache import selected_indices


def require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {path}")
    return path


def validate_checkpoint(path):
    path = require_file(path, "generator checkpoint")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a state-dict mapping: {path}")
    for key in ("generator", "generator_ema", "model"):
        if key in payload and isinstance(payload[key], dict) and payload[key]:
            state = payload[key]
            break
    else:
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            state = payload
        else:
            raise ValueError(
                f"Checkpoint has none of generator/generator_ema/model and is not a bare state dict: {path}"
            )
    if not any(name.endswith("patch_embedding.weight") for name in state):
        raise ValueError(f"Checkpoint does not look like a Wan generator state dict: {path}")
    del state, payload


def validate_teacher(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Teacher directory not found: {path}")
    require_file(path / "config.json", "teacher config")
    shards = sorted(path.glob("diffusion_pytorch_model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No teacher safetensors found in {path}")


def validate_local_wan_assets(require_text_encoder):
    root = REPO_ROOT / "wan_models" / "Wan2.1-T2V-1.3B"
    required = [
        root / "config.json",
        root / "diffusion_pytorch_model.safetensors",
        root / "Wan2.1_VAE.pth",
    ]
    if require_text_encoder:
        required.extend(
            [
                root / "models_t5_umt5-xxl-enc-bf16.pth",
                root / "google" / "umt5-xxl" / "spiece.model",
                root / "google" / "umt5-xxl" / "tokenizer_config.json",
            ]
        )
    for path in required:
        require_file(path, "Wan2.1-1.3B asset")


def open_dataset(path):
    path = Path(path)
    require_file(path / "data.mdb", "training LMDB data.mdb")
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
        max_readers=512,
    )
    shape = get_array_shape_from_lmdb(env, "latents")
    if len(shape) not in (5, 6) or shape[0] < 1:
        env.close()
        raise ValueError(f"Unexpected clean-latent LMDB shape: {shape}")
    first_prompt = retrieve_row_from_lmdb(env, "prompts", str, 0)
    if not first_prompt.strip():
        env.close()
        raise ValueError("The first training prompt is empty")
    return env, shape


def validate_prompt_cache(
    cache_path,
    dataset_env,
    dataset_size,
    negative_prompt,
    required_indices=None,
):
    cache_path = Path(cache_path)
    require_file(cache_path / "data.mdb", "prompt-cache data.mdb")
    cache_env = lmdb.open(
        str(cache_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
        max_readers=512,
    )
    missing = []
    indices = range(dataset_size) if required_indices is None else required_indices
    with cache_env.begin(write=False) as txn:
        for index in indices:
            prompt = retrieve_row_from_lmdb(dataset_env, "prompts", str, index)
            if txn.get(prompt_cache_key(prompt)) is None:
                missing.append((index, prompt))
                if len(missing) >= 8:
                    break
        if txn.get(prompt_cache_key(str(negative_prompt))) is None:
            missing.append(("negative_prompt", str(negative_prompt)))
    cache_env.close()
    if missing:
        preview = ", ".join(f"{index}:{prompt[:40]!r}" for index, prompt in missing)
        raise KeyError(f"Prompt cache is incomplete; first missing entries: {preview}")

    metadata_path = cache_path / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_dataset_size = int(metadata.get("dataset_size", -1))
        if cached_dataset_size != dataset_size:
            raise ValueError(
                f"Prompt-cache metadata dataset_size={cached_dataset_size}, "
                f"training LMDB has {dataset_size}"
            )


def validate_gpus(gpu_ids):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment")
    parsed = [int(item) for item in gpu_ids.split(",")]
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"Duplicate GPU IDs in --gpus: {gpu_ids}")
    device_count = torch.cuda.device_count()
    invalid = [index for index in parsed if index < 0 or index >= device_count]
    if invalid:
        raise ValueError(
            f"GPU IDs {invalid} are invalid; this process sees {device_count} CUDA devices"
        )


def validate_versions():
    actual = {
        "torch": torch.__version__.split("+", 1)[0],
        "torchvision": torchvision.__version__.split("+", 1)[0],
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "accelerate": accelerate.__version__,
        "numpy": np.__version__,
    }
    expected = {
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "transformers": "4.49.0",
        "diffusers": "0.31.0",
        "accelerate": "1.13.0",
        "numpy": "1.24.4",
    }
    mismatches = {
        name: (actual[name], version)
        for name, version in expected.items()
        if actual[name] != version
    }
    if mismatches:
        raise RuntimeError(
            "Training environment does not match the audited versions: "
            + ", ".join(
                f"{name}={found} (expected {wanted})"
                for name, (found, wanted) in mismatches.items()
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--teacher_model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt_embedding_cache_path", default="")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Smoke override; validates only the exact DistributedSampler prefix it consumes.",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"One-Forcing is validated with Python 3.10; got {sys.version.split()[0]}"
        )
    config = load_config(args.config_path)
    config.seed = args.seed
    if str(config.dataset_type) != "clean_latent_lmdb":
        raise ValueError("Rebuttal training requires dataset_type=clean_latent_lmdb")

    validate_versions()
    validate_gpus(args.gpus)
    validate_checkpoint(args.generator_ckpt)
    validate_teacher(args.teacher_model_path)
    validate_local_wan_assets(require_text_encoder=not bool(args.prompt_embedding_cache_path))
    dataset_env, latent_shape = open_dataset(args.data_path)
    try:
        if args.prompt_embedding_cache_path:
            required_indices = None
            if args.max_steps:
                if args.max_steps < 1:
                    raise ValueError("--max_steps must be positive when provided")
                update_ratio = int(config.dfake_gen_update_ratio)
                generator_updates = (args.max_steps + update_ratio - 1) // update_ratio
                batches_per_rank = args.max_steps + generator_updates
                required_indices = selected_indices(
                    int(latent_shape[0]),
                    max_prompts=0,
                    first_batches_per_rank=batches_per_rank,
                    world_size=len(args.gpus.split(",")),
                    train_batch_size=int(config.batch_size),
                    seed=int(config.seed),
                )
            validate_prompt_cache(
                args.prompt_embedding_cache_path,
                dataset_env,
                dataset_size=int(latent_shape[0]),
                negative_prompt=config.negative_prompt,
                required_indices=required_indices,
            )
    finally:
        dataset_env.close()

    print(
        "PASS: training assets, checkpoint structure, LMDB, prompt cache, "
        f"Python, and GPUs are ready (latents_shape={latent_shape})."
    )


if __name__ == "__main__":
    main()
