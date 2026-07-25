#!/usr/bin/env python3
"""Extract the unique official prompts from VBench_full_info.json."""

import argparse
import json
from pathlib import Path


def unique_vbench_prompts(full_info_path, dimensions=None):
    with open(full_info_path, "r", encoding="utf-8") as fp:
        full_info = json.load(fp)
    if not isinstance(full_info, list) or not full_info:
        raise ValueError(f"Invalid or empty VBench full-info file: {full_info_path}")

    selected_dimensions = set(dimensions or [])
    prompts = []
    seen = set()
    for index, record in enumerate(full_info):
        if (
            not isinstance(record, dict)
            or "prompt_en" not in record
            or "dimension" not in record
        ):
            raise ValueError(f"Invalid VBench record at index {index}")
        record_dimensions = record["dimension"]
        if not isinstance(record_dimensions, list):
            raise ValueError(
                f"Invalid VBench dimension list at index {index}: "
                f"{record_dimensions!r}"
            )
        if selected_dimensions and not selected_dimensions.intersection(
            record_dimensions
        ):
            continue
        prompt = str(record["prompt_en"])
        if not prompt.strip() or "\n" in prompt or "\r" in prompt:
            raise ValueError(f"Invalid one-line VBench prompt at index {index}: {prompt!r}")
        if "/" in prompt or "\x00" in prompt:
            raise ValueError(
                f"VBench prompt {index} cannot be represented as a video filename"
            )
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    if selected_dimensions and not prompts:
        raise ValueError(
            "None of the requested dimensions occur in the VBench full-info file: "
            f"{sorted(selected_dimensions)}"
        )
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_info_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=[],
        help=(
            "Optional VBench dimensions. When provided, emit the unique union of "
            "only prompts assigned to those dimensions."
        ),
    )
    args = parser.parse_args()

    prompts = unique_vbench_prompts(args.full_info_path, args.dimensions)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        for prompt in prompts:
            fp.write(prompt + "\n")
    print(f"Wrote {len(prompts)} unique VBench prompts to {output_path}")


if __name__ == "__main__":
    main()
