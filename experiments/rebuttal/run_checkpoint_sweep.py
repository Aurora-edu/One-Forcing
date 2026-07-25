#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.run_sharded_inference import parse_gpus


def discover_steps(run_dir):
    steps = []
    for path in run_dir.glob("checkpoint_model_*/model.pt"):
        try:
            steps.append(int(path.parent.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(steps))


def select_result_dimensions(source: Path, destination: Path, dimensions):
    with open(source, encoding="utf-8") as fp:
        payload = json.load(fp)
    missing = [dimension for dimension in dimensions if dimension not in payload]
    if missing:
        raise ValueError(f"{source} is missing requested dimensions: {missing}")
    selected = {dimension: payload[dimension] for dimension in dimensions}
    with open(destination, "w", encoding="utf-8") as fp:
        json.dump(selected, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Export and evaluate every requested training checkpoint with VBench."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--extended_prompt_path", default="")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--full_info_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", nargs="*", type=int, default=[])
    parser.add_argument(
        "--existing_result",
        action="append",
        default=[],
        metavar="STEP=RESULT_JSON",
        help=(
            "Append an already-computed checkpoint result without regenerating "
            "videos, e.g. 600=eval/final/main/vbench/result.json."
        ),
    )
    parser.add_argument("--schedule", choices=["all1", "ffe", "all4"], default="ffe")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPUs used concurrently for generation and VBench.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=[],
        help="Optional VBench dimension subset. Empty means all official dimensions.",
    )
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--vbench_python",
        default=os.environ.get("VBENCH_PYTHON", sys.executable),
        help="Python executable from the separate VBench environment.",
    )
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)

    run_dir = Path(args.run_dir).resolve()
    steps = sorted(set(args.steps)) if args.steps else discover_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"No checkpoint_model_*/model.pt checkpoints in {run_dir}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in steps:
        checkpoint = run_dir / f"checkpoint_model_{step:06d}" / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        step_root = output_root / f"step_{step:06d}"
        videos_dir = step_root / "videos"
        eval_dir = step_root / "vbench"
        eval_name = f"{args.model_name}_seed{args.seed}_step{step:06d}"

        infer_command = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "rebuttal" / "run_sharded_inference.py"),
            "--method",
            "framewise",
            "--schedule",
            args.schedule,
            "--config_path",
            os.path.abspath(args.config_path),
            "--checkpoint_path",
            str(checkpoint),
            "--prompt_path",
            os.path.abspath(args.prompt_path),
            "--manifest_path",
            os.path.abspath(args.manifest_path),
            "--output_folder",
            str(videos_dir),
            "--num_output_frames",
            str(args.num_output_frames),
            "--gpus",
            args.gpus,
        ]
        if args.extended_prompt_path:
            infer_command.extend(
                ["--extended_prompt_path", os.path.abspath(args.extended_prompt_path)]
            )
        if args.use_ema:
            infer_command.append("--use_ema")
        subprocess.run(infer_command, cwd=REPO_ROOT, check=True)

        vbench_command = [
            args.vbench_python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(gpus)}",
            str(REPO_ROOT / "scripts" / "run_vbench.py"),
            "--videos_path",
            str(videos_dir),
            "--full_info_path",
            os.path.abspath(args.full_info_path),
            "--output_dir",
            str(eval_dir),
            "--name",
            eval_name,
            "--device",
            "cuda",
        ]
        if args.dimensions:
            vbench_command.extend(["--dimensions", *args.dimensions])
        vbench_env = os.environ.copy()
        vbench_env["CUDA_VISIBLE_DEVICES"] = args.gpus
        subprocess.run(
            vbench_command,
            cwd=REPO_ROOT,
            env=vbench_env,
            check=True,
        )
        result_json = eval_dir / f"{eval_name}_eval_results.json"
        if not result_json.is_file():
            raise FileNotFoundError(f"VBench did not produce {result_json}")
        rows.append(
            {
                "model": args.model_name,
                "seed": args.seed,
                "step": step,
                "window": "full",
                "schedule": args.schedule,
                "result_json": str(result_json),
            }
        )

    for specification in args.existing_result:
        if "=" not in specification:
            raise ValueError(
                f"--existing_result must be STEP=RESULT_JSON, got {specification!r}"
            )
        step_text, result_text = specification.split("=", 1)
        step = int(step_text)
        result_json = Path(result_text).resolve()
        if step < 1:
            raise ValueError(
                f"--existing_result step must be positive, got {step}"
            )
        if not result_json.is_file():
            raise FileNotFoundError(result_json)
        if any(int(row["step"]) == step for row in rows):
            raise ValueError(
                f"step={step} occurs in both --steps and --existing_result"
            )
        if args.dimensions:
            filtered_result = output_root / (
                f"existing_step_{step:06d}_selected_dimensions.json"
            )
            result_json = select_result_dimensions(
                result_json,
                filtered_result,
                args.dimensions,
            )
        rows.append(
            {
                "model": args.model_name,
                "seed": args.seed,
                "step": step,
                "window": "full",
                "schedule": args.schedule,
                "result_json": str(result_json),
            }
        )

    rows.sort(key=lambda row: int(row["step"]))
    if not rows:
        raise ValueError("No checkpoint or existing-result rows were selected")
    manifest_path = output_root / "vbench_sweep.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
