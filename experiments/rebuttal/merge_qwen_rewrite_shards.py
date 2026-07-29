#!/usr/bin/env python3
"""Merge historical Self-Forcing Qwen pair shards in official prompt order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def read_nonempty_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"Expected non-empty one-line rewrites in {path}")
    return lines


def load_pair_mapping(paths: list[Path]) -> dict[str, str]:
    if len(paths) != 2:
        raise ValueError("The historical Self-Forcing export used exactly two pair shards")
    mapping = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                raise ValueError(f"Blank pair record at {path}:{line_number}")
            record = json.loads(line)
            prompt = record.get("prompt")
            rewrite = record.get("rewrite")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Invalid prompt at {path}:{line_number}")
            if not isinstance(rewrite, str) or not rewrite.strip():
                raise ValueError(f"Invalid rewrite at {path}:{line_number}")
            if prompt in mapping:
                raise ValueError(f"Duplicate prompt across Qwen pair shards: {prompt!r}")
            mapping[prompt] = rewrite
    return mapping


def merge_in_prompt_order(prompts: list[str], mapping: dict[str, str]) -> list[str]:
    if len(prompts) != len(set(prompts)):
        raise ValueError("Official VBench prompt file contains duplicates")
    missing = [prompt for prompt in prompts if prompt not in mapping]
    extra = sorted(set(mapping) - set(prompts))
    if missing or extra:
        raise ValueError(
            "Qwen pair shards do not exactly cover official prompts: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return [mapping[prompt] for prompt in prompts]


def atomic_write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{line}\n" for line in lines)
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise FileExistsError(f"Refusing to overwrite different Qwen merge: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--pair_shard", action="append", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_count", type=int, default=944)
    args = parser.parse_args()
    if args.expected_count < 1:
        raise ValueError("--expected_count must be positive")

    prompt_path = Path(args.prompt_path).resolve()
    paths = [Path(value).resolve() for value in args.pair_shard]
    for path in [prompt_path, *paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    prompts = read_nonempty_lines(prompt_path)
    merged = merge_in_prompt_order(prompts, load_pair_mapping(paths))
    if len(merged) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} merged rewrites, found {len(merged)}"
        )
    output_path = Path(args.output_path).resolve()
    atomic_write(output_path, merged)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"PASS: merged {len(merged)} Qwen rewrites: {output_path} sha256={digest}")


if __name__ == "__main__":
    main()
