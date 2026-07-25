#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LONG_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run VBench separately on exact early/middle/late rollout windows."
    )
    parser.add_argument("--windows_root", required=True)
    parser.add_argument("--full_info_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["early", "middle", "late"],
        choices=["early", "middle", "late"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=DEFAULT_LONG_DIMENSIONS,
        help="Custom-input-compatible temporal dimensions used on each rollout window.",
    )
    parser.add_argument(
        "--vbench_python",
        default=os.environ.get("VBENCH_PYTHON", sys.executable),
        help="Python executable from the separate VBench environment.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for position in dict.fromkeys(args.positions):
        videos_path = Path(args.windows_root).resolve() / position
        if not videos_path.is_dir():
            raise FileNotFoundError(videos_path)
        name = (
            f"{args.model_name}_seed{args.seed}_step{args.step:06d}_"
            f"{args.schedule}_{position}"
        )
        eval_dir = output_root / position
        command = [
            args.vbench_python,
            str(REPO_ROOT / "scripts" / "run_vbench.py"),
            "--videos_path",
            str(videos_path),
            "--full_info_path",
            str(Path(args.full_info_path).resolve()),
            "--output_dir",
            str(eval_dir),
            "--name",
            name,
            "--device",
            args.device,
            "--mode",
            "custom_input",
            "--dimensions",
            *args.dimensions,
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        result_json = eval_dir / f"{name}_eval_results.json"
        if not result_json.is_file():
            raise FileNotFoundError(result_json)
        rows.append(
            {
                "model": args.model_name,
                "seed": args.seed,
                "step": args.step,
                "window": position,
                "schedule": args.schedule,
                "result_json": str(result_json),
            }
        )

    manifest_path = output_root / "long_vbench.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
