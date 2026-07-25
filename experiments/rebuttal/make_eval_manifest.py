#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def read_lines(path):
    with open(path, encoding="utf-8") as fp:
        return [line.rstrip("\n") for line in fp]


def main():
    parser = argparse.ArgumentParser(
        description="Create a paired prompt/sample/seed manifest shared by every method."
    )
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--extended_prompt_path", default="")
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--naming",
        choices=["vbench", "index"],
        default="vbench",
        help="vbench preserves the exact prompt in the filename; index is for non-VBench metrics.",
    )
    args = parser.parse_args()

    if args.base_seed < 0:
        raise ValueError("--base_seed must be non-negative")
    if args.num_samples_per_prompt < 1:
        raise ValueError("--num_samples_per_prompt must be positive")

    prompts = read_lines(args.prompt_path)
    extended_prompts = (
        read_lines(args.extended_prompt_path) if args.extended_prompt_path else None
    )
    if extended_prompts is not None and len(extended_prompts) != len(prompts):
        raise ValueError("Prompt and extended-prompt files have different lengths")
    if args.limit > 0:
        prompts = prompts[:args.limit]
        if extended_prompts is not None:
            extended_prompts = extended_prompts[:args.limit]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_sha256 = hashlib.sha256(
        Path(args.prompt_path).read_bytes()
    ).hexdigest()
    output_names = set()
    records = []
    for prompt_index, prompt in enumerate(prompts):
        if not prompt.strip():
            raise ValueError(f"Prompt {prompt_index} is empty")
        for sample_index in range(args.num_samples_per_prompt):
            if args.naming == "vbench":
                if "/" in prompt or "\x00" in prompt:
                    raise ValueError(
                        f"Prompt {prompt_index} cannot be used as a VBench filename"
                    )
                output_name = f"{prompt}-{sample_index}.mp4"
                if len(output_name.encode("utf-8")) > 240:
                    raise ValueError(
                        f"Prompt {prompt_index} makes a filename longer than 240 bytes; "
                        "use --naming index only for non-VBench evaluation"
                    )
            else:
                output_name = (
                    f"prompt_{prompt_index:04d}_sample_{sample_index:02d}.mp4"
                )
            if output_name in output_names:
                raise ValueError(
                    f"Duplicate output filename {output_name!r}; prompts must be unique "
                    "when --naming vbench is used"
                )
            output_names.add(output_name)
            record = {
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "seed": (
                    args.base_seed
                    + prompt_index * args.num_samples_per_prompt
                    + sample_index
                ),
                "output_name": output_name,
                "prompt": prompt,
                "prompt_file_sha256": prompt_sha256,
            }
            if extended_prompts is not None:
                record["extended_prompt"] = extended_prompts[prompt_index]
            records.append(record)

    with open(output_path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(
        f"Wrote {len(prompts) * args.num_samples_per_prompt} paired samples to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
