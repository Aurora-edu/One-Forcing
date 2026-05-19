import argparse
import os
import subprocess
import sys
import zipfile

import torch


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]


def zip_directory(input_dir: str, output_zip: str) -> None:
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, input_dir)
                zf.write(full_path, rel_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", type=str, required=True)
    parser.add_argument("--full_info_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dimensions", nargs="*", default=DEFAULT_DIMENSIONS)
    parser.add_argument("--vbench_repo", type=str, default="")
    parser.add_argument("--zip_name", type=str, default="")
    args = parser.parse_args()

    try:
        from vbench import VBench
        from vbench.distributed import barrier, dist_init, get_rank
    except ImportError as exc:
        raise SystemExit(
            "vbench is not installed. Install it with `pip install vbench` and "
            "download `VBench_full_info.json` as described in the official README."
        ) from exc

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not torch.distributed.is_initialized():
        dist_init()

    rank = get_rank()
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed:
        barrier()

    evaluator = VBench(torch.device(args.device), args.full_info_path, args.output_dir)
    evaluator.evaluate(
        videos_path=args.videos_path,
        name=args.name,
        dimension_list=args.dimensions,
    )

    if distributed:
        barrier()

    if args.vbench_repo and rank == 0:
        zip_name = args.zip_name or f"{args.name}.zip"
        zip_path = os.path.join(args.output_dir, zip_name)
        zip_directory(args.output_dir, zip_path)
        script_path = os.path.join(args.vbench_repo, "scripts", "cal_final_score.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Cannot find {script_path}")
        subprocess.run(
            [
                sys.executable,
                script_path,
                "--zip_file",
                zip_path,
                "--model_name",
                args.name,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
