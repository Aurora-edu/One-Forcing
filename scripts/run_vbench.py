import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version as package_version
from pathlib import Path

import torch


VIDEO_SUFFIXES = {".mp4", ".gif", ".jpg", ".png"}
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


def validate_vbench_versions() -> None:
    expected = {
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "vbench": "0.1.5",
        "timm": "1.0.12",
        "transformers": "4.33.2",
        "numpy": "1.24.4",
    }
    mismatches = {}
    for package, wanted in expected.items():
        found = package_version(package).split("+", 1)[0]
        if found != wanted:
            mismatches[package] = (found, wanted)
    if mismatches:
        raise RuntimeError(
            "VBench environment does not match the audited versions: "
            + ", ".join(
                f"{name}={found} (expected {wanted})"
                for name, (found, wanted) in mismatches.items()
            )
        )


def discover_videos(videos_path: str) -> list[Path]:
    path = Path(videos_path).resolve()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            candidate.resolve()
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in VIDEO_SUFFIXES
        )
    else:
        raise FileNotFoundError(path)
    if not candidates:
        raise ValueError(f"No supported videos found in {path}")
    suffixes = {candidate.suffix.lower() for candidate in candidates}
    if len(suffixes) != 1:
        raise ValueError(
            "VBench requires one common input suffix, but found "
            f"{sorted(suffixes)} in {path}"
        )
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("VBench input video basenames must be unique")
    return candidates


def validate_standard_coverage(
    videos: list[Path],
    full_info_path: str,
    dimensions: list[str],
) -> None:
    with open(full_info_path, "r", encoding="utf-8") as fp:
        full_info = json.load(fp)
    if not isinstance(full_info, list) or not full_info:
        raise ValueError(f"Invalid or empty VBench full-info file: {full_info_path}")

    selected_dimensions = set(dimensions)
    suffix = videos[0].suffix
    available = {video.name for video in videos}
    expected = set()
    selected_prompts = 0
    for record in full_info:
        if not selected_dimensions.intersection(record["dimension"]):
            continue
        prompt = record["prompt_en"]
        selected_prompts += 1
        expected.update(f"{prompt}-{sample_index}{suffix}" for sample_index in range(5))

    if selected_prompts == 0:
        raise ValueError(
            "None of the requested dimensions occur in the VBench full-info file: "
            f"{sorted(selected_dimensions)}"
        )
    missing = sorted(expected - available)
    if missing:
        preview = "\n".join(f"  {name}" for name in missing[:20])
        remainder = len(missing) - min(len(missing), 20)
        suffix_message = (
            f"\n  ... and {remainder} more missing files" if remainder else ""
        )
        raise ValueError(
            "Incomplete VBench standard input. The official protocol requires five "
            "samples per selected prompt, named '<prompt>-0' through '<prompt>-4'. "
            f"Missing {len(missing)} of {len(expected)} required files:\n"
            f"{preview}{suffix_message}"
        )


def validate_result_json(
    result_path: Path,
    dimensions: list[str],
) -> None:
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with open(result_path, "r", encoding="utf-8") as fp:
        results = json.load(fp)

    def assert_finite(value, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite VBench result at {location}: {value}")
        if isinstance(value, list):
            for index, item in enumerate(value):
                assert_finite(item, f"{location}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                assert_finite(item, f"{location}.{key}")

    for dimension in dimensions:
        if dimension not in results:
            raise ValueError(
                f"VBench output is missing requested dimension {dimension!r}"
            )
        value = results[dimension]
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError(
                f"Unexpected VBench result format for {dimension!r}: {value!r}"
            )
        score, details = value[0], value[1]
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(
                f"VBench produced a non-finite score for {dimension!r}: {score!r}"
            )
        if not details:
            raise ValueError(
                f"VBench evaluated no videos for requested dimension {dimension!r}"
            )
        assert_finite(value, dimension)


def make_staging_directory(
    videos: list[Path],
    output_dir: str,
    name: str,
) -> Path:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{name}_inputs_", dir=output_dir)
    ).resolve()
    for video in videos:
        os.symlink(video, staging / video.name)
    return staging


def rewrite_staged_paths(
    output_dir: str,
    name: str,
    staging_path: str,
    videos: list[Path],
) -> None:
    staging = Path(staging_path)
    replacements = {
        str(staging / video.name): str(video.resolve())
        for video in videos
    }

    def rewrite(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for suffix in ("_full_info.json", "_eval_results.json"):
        path = Path(output_dir) / f"{name}{suffix}"
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(rewrite(payload), fp, indent=4, ensure_ascii=False)
            fp.write("\n")


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
    parser.add_argument(
        "--mode",
        choices=["vbench_standard", "custom_input"],
        default="vbench_standard",
        help=(
            "Use vbench_standard only for the official prompt set with five samples "
            "per prompt. Use custom_input for long-video windows or smoke tests."
        ),
    )
    parser.add_argument("--vbench_repo", type=str, default="")
    parser.add_argument("--zip_name", type=str, default="")
    args = parser.parse_args()

    validate_vbench_versions()
    videos = discover_videos(args.videos_path)
    if args.mode == "vbench_standard":
        validate_standard_coverage(
            videos=videos,
            full_info_path=args.full_info_path,
            dimensions=args.dimensions,
        )

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

    staging_path = None
    staging_holder = [None]
    if rank == 0:
        staging_holder[0] = str(
            make_staging_directory(
                videos=videos,
                output_dir=args.output_dir,
                name=args.name,
            )
        )
    if distributed:
        torch.distributed.broadcast_object_list(staging_holder, src=0)
        barrier()
    staging_path = staging_holder[0]

    try:
        evaluator = VBench(
            torch.device(args.device), args.full_info_path, args.output_dir
        )
        evaluator.evaluate(
            videos_path=staging_path,
            name=args.name,
            dimension_list=args.dimensions,
            mode=args.mode,
        )
        if distributed:
            barrier()
        if rank == 0:
            rewrite_staged_paths(
                output_dir=args.output_dir,
                name=args.name,
                staging_path=staging_path,
                videos=videos,
            )
    finally:
        if distributed:
            barrier()
        if rank == 0 and staging_path:
            shutil.rmtree(staging_path)

    result_path = Path(args.output_dir) / f"{args.name}_eval_results.json"
    if rank == 0:
        validate_result_json(result_path, args.dimensions)

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
