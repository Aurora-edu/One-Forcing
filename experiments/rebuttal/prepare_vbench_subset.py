#!/usr/bin/env python3
"""Create a VBench dimension subset while preserving full-manifest seeds."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

try:
    from experiments.rebuttal.prepare_vbench_prompts import unique_vbench_prompts
except ModuleNotFoundError:
    from prepare_vbench_prompts import unique_vbench_prompts


def read_prompts(path: Path):
    prompts = path.read_text(encoding="utf-8").splitlines()
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError(f"Prompt file is empty or contains blank lines: {path}")
    if len(prompts) != len(set(prompts)):
        raise ValueError(f"Prompt file contains duplicates: {path}")
    return prompts


def prepare_subset(
    full_info_path: Path,
    full_prompt_path: Path,
    full_manifest_path: Path,
    output_prompt_path: Path,
    output_manifest_path: Path,
    dimensions,
):
    if not dimensions:
        raise ValueError("At least one dimension is required")
    full_prompts = read_prompts(full_prompt_path)
    selected_set = set(unique_vbench_prompts(full_info_path, dimensions))
    selected_prompts = [prompt for prompt in full_prompts if prompt in selected_set]
    if set(selected_prompts) != selected_set:
        missing = sorted(selected_set - set(selected_prompts))
        raise ValueError(
            f"Full prompt file is missing selected VBench prompts: {missing[:8]}"
        )

    full_prompt_sha256 = hashlib.sha256(full_prompt_path.read_bytes()).hexdigest()
    records_by_prompt = defaultdict(list)
    with open(full_manifest_path, encoding="utf-8") as fp:
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
                    raise ValueError(
                        f"{full_manifest_path}:{line_number}: missing {key}"
                    )
            prompt_index = int(record["prompt_index"])
            if prompt_index < 0 or prompt_index >= len(full_prompts):
                raise IndexError(
                    f"{full_manifest_path}:{line_number}: invalid prompt_index"
                )
            prompt = full_prompts[prompt_index]
            if record["prompt"] != prompt:
                raise ValueError(
                    f"{full_manifest_path}:{line_number}: prompt/index mismatch"
                )
            if record["prompt_file_sha256"] != full_prompt_sha256:
                raise ValueError(
                    f"{full_manifest_path}:{line_number}: prompt hash mismatch"
                )
            records_by_prompt[prompt].append(record)
    if not records_by_prompt:
        raise ValueError(f"Full manifest is empty: {full_manifest_path}")

    output_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_prompt_path, "w", encoding="utf-8") as fp:
        for prompt in selected_prompts:
            fp.write(prompt + "\n")
    output_prompt_sha256 = hashlib.sha256(output_prompt_path.read_bytes()).hexdigest()

    selected_records = []
    for prompt_index, prompt in enumerate(selected_prompts):
        prompt_records = sorted(
            records_by_prompt.get(prompt, []),
            key=lambda record: int(record["sample_index"]),
        )
        if not prompt_records:
            raise ValueError(f"Full manifest has no records for prompt {prompt!r}")
        sample_indices = [int(record["sample_index"]) for record in prompt_records]
        if sample_indices != list(range(len(sample_indices))):
            raise ValueError(
                f"Prompt {prompt!r} has non-contiguous sample indices: "
                f"{sample_indices}"
            )
        for record in prompt_records:
            selected_records.append(
                {
                    **record,
                    "prompt_index": prompt_index,
                    "prompt_file_sha256": output_prompt_sha256,
                }
            )

    with open(output_manifest_path, "w", encoding="utf-8") as fp:
        for record in selected_records:
            fp.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(
        f"Wrote {len(selected_prompts)} prompts and {len(selected_records)} "
        f"seed-preserving records to {output_manifest_path}"
    )
    return selected_prompts, selected_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_info_path", required=True)
    parser.add_argument("--full_prompt_path", required=True)
    parser.add_argument("--full_manifest_path", required=True)
    parser.add_argument("--output_prompt_path", required=True)
    parser.add_argument("--output_manifest_path", required=True)
    parser.add_argument("--dimensions", nargs="+", required=True)
    args = parser.parse_args()

    prepare_subset(
        full_info_path=Path(args.full_info_path).resolve(),
        full_prompt_path=Path(args.full_prompt_path).resolve(),
        full_manifest_path=Path(args.full_manifest_path).resolve(),
        output_prompt_path=Path(args.output_prompt_path).resolve(),
        output_manifest_path=Path(args.output_manifest_path).resolve(),
        dimensions=args.dimensions,
    )


if __name__ == "__main__":
    main()
