#!/usr/bin/env python3
"""Build the exact two-stream seed-0 manifest used by historical Self-Forcing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


RNG_PROTOCOL = "self_forcing_two_shard_seed0"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_nonempty_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"Expected non-empty one-line records in {path}")
    return lines


def build_records(prompt_path: Path, rewrite_path: Path) -> list[dict]:
    prompts = read_nonempty_lines(prompt_path)
    rewrites = read_nonempty_lines(rewrite_path)
    if len(prompts) != 944 or len(rewrites) != 944:
        raise ValueError(
            f"Expected 944 prompts and rewrites, found {len(prompts)} and {len(rewrites)}"
        )
    if len(set(prompts)) != len(prompts):
        raise ValueError("Official VBench prompts must be unique")

    prompt_digest = sha256_file(prompt_path)
    rewrite_digest = sha256_file(rewrite_path)
    records = []
    for index, (prompt, rewrite) in enumerate(zip(prompts, rewrites)):
        if "/" in prompt or "\x00" in prompt:
            raise ValueError(f"Prompt {index} is not a valid VBench filename")
        output_name = f"{prompt}-0.mp4"
        if len(output_name.encode("utf-8")) > 240:
            raise ValueError(f"Prompt {index} makes a filename longer than 240 bytes")
        records.append(
            {
                "prompt_index": index,
                "sample_index": 0,
                # The old exporter started both processes with seed 0, then
                # seeded each video's private initial-noise generator with its
                # local index inside the even/odd prompt shard.
                "seed": index // 2,
                "initial_noise_seed": index // 2,
                "output_name": output_name,
                "prompt": prompt,
                "extended_prompt": rewrite,
                "prompt_file_sha256": prompt_digest,
                "rewrite_file_sha256": rewrite_digest,
                "rng_protocol": RNG_PROTOCOL,
                "rng_shard_index": index % 2,
                "rng_position_in_shard": index // 2,
            }
        )
    return records


def atomic_write(path: Path, records: list[dict]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise FileExistsError(f"Refusing to overwrite a different manifest: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--qwen_rewrite_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    prompt_path = Path(args.prompt_path).resolve()
    rewrite_path = Path(args.qwen_rewrite_path).resolve()
    for path in (prompt_path, rewrite_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path = Path(args.output_path).resolve()
    records = build_records(prompt_path, rewrite_path)
    atomic_write(output_path, records)
    print(
        f"PASS: wrote {len(records)} records using {RNG_PROTOCOL}: "
        f"{output_path} sha256={sha256_file(output_path)}"
    )


if __name__ == "__main__":
    main()
