#!/usr/bin/env python3
"""Precompute deterministic T5 embeddings for rebuttal training prompts."""

import argparse
import io
import json
import os
from pathlib import Path
import sys

import lmdb
import torch
from torch.utils.data import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config
from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from utils.prompt_embedding_cache import prompt_cache_key
from utils.wan_wrapper import WanTextEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the prompt-embedding LMDB used by training. By default all "
            "dataset prompts are cached. --first_batches_per_rank selects the "
            "exact prefix consumed by a DistributedSampler smoke run."
        )
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--data_path", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--map_size_gib", type=int, default=64)
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument("--first_batches_per_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def selected_indices(
    dataset_size: int,
    *,
    max_prompts: int,
    first_batches_per_rank: int,
    world_size: int,
    train_batch_size: int,
    seed: int,
) -> list[int]:
    if first_batches_per_rank:
        if world_size < 1 or train_batch_size < 1:
            raise ValueError("world_size and train_batch_size must be positive")
        per_rank = first_batches_per_rank * train_batch_size
        selected = set()
        dataset_stub = range(dataset_size)
        for rank in range(world_size):
            sampler = DistributedSampler(
                dataset_stub,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=True,
            )
            selected.update(list(iter(sampler))[:per_rank])
        indices = sorted(selected)
    else:
        indices = list(range(dataset_size))

    if max_prompts:
        if max_prompts < 1:
            raise ValueError("max_prompts must be zero or positive")
        indices = indices[:max_prompts]
    return indices


def trim_padding(embedding: torch.Tensor) -> torch.Tensor:
    embedding = embedding.detach().to(device="cpu", dtype=torch.bfloat16)
    nonzero_rows = torch.count_nonzero(embedding, dim=-1).nonzero(as_tuple=False)
    length = int(nonzero_rows[-1].item()) + 1 if len(nonzero_rows) else 1
    return embedding[:length].contiguous()


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    payload = io.BytesIO()
    torch.save(tensor, payload)
    return payload.getvalue()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    config = load_config(args.config_path)
    data_path = os.path.realpath(args.data_path or config.data_path)
    if not os.path.isfile(os.path.join(data_path, "data.mdb")):
        raise FileNotFoundError(f"Clean-latent LMDB not found: {data_path}")

    source_env = lmdb.open(
        data_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    dataset_size = int(get_array_shape_from_lmdb(source_env, "latents")[0])
    seed = int(config.seed if args.seed is None else args.seed)
    indices = selected_indices(
        dataset_size,
        max_prompts=args.max_prompts,
        first_batches_per_rank=args.first_batches_per_rank,
        world_size=args.world_size,
        train_batch_size=args.train_batch_size,
        seed=seed,
    )
    prompts = [
        retrieve_row_from_lmdb(source_env, "prompts", str, index)
        for index in indices
    ]
    prompts.append(str(config.negative_prompt))
    prompts = list(dict.fromkeys(prompts))

    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    output_env = lmdb.open(
        str(output_path),
        map_size=int(args.map_size_gib) * 1024**3,
        subdir=True,
        meminit=False,
    )

    with output_env.begin(write=False) as txn:
        missing_prompts = [
            prompt
            for prompt in prompts
            if txn.get(prompt_cache_key(prompt)) is None
        ]
    skipped = len(prompts) - len(missing_prompts)
    written = 0

    if missing_prompts:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        encoder = WanTextEncoder(dtype=torch.bfloat16, assign_load=True)
        encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()

    for start in range(0, len(missing_prompts), args.batch_size):
        missing = missing_prompts[start : start + args.batch_size]

        with torch.inference_mode():
            batch_embeddings = encoder(missing)["prompt_embeds"]
        with output_env.begin(write=True) as txn:
            for prompt, embedding in zip(missing, batch_embeddings):
                txn.put(
                    prompt_cache_key(prompt),
                    serialize_tensor(trim_padding(embedding)),
                    overwrite=False,
                )
                written += 1
        print(
            f"cached {written}/{len(missing_prompts)} missing prompts "
            f"({skipped} already present)",
            flush=True,
        )

    metadata = {
        "config_path": os.path.realpath(args.config_path),
        "data_path": data_path,
        "dataset_size": dataset_size,
        "selected_dataset_indices": len(indices),
        "unique_prompts": len(prompts),
        "seed": seed,
        "world_size": args.world_size,
        "train_batch_size": args.train_batch_size,
        "first_batches_per_rank": args.first_batches_per_rank,
        "dtype": "bfloat16",
    }
    with output_env.begin(write=True) as txn:
        txn.put(b"__metadata__", json.dumps(metadata, sort_keys=True).encode("utf-8"))
    with open(output_path / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)
        fp.write("\n")
    output_env.sync()
    output_env.close()
    source_env.close()
    print(f"Prompt cache ready: {output_path} ({written} written, {skipped} reused)")


if __name__ == "__main__":
    main()
