"""Reevaluate the existing 1000-update endpoint-regression pair with FFE.

Default: print the two evaluation commands without loading models or creating
files. Execution reuses the original checkpoint/prompt/noise manifests and
writes a separate vbench_main_ffe directory. It never launches training.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "experiments/rebuttal"


def build_plan(args):
    root = Path(args.experiment_root).resolve()
    output = (Path(args.output).resolve() if getattr(args, "output", None)
              else root / "vbench_main_ffe")
    prompts = root / "manifests/vbench_prompts.txt"
    manifest = root / "manifests/vbench_single_sample_seed0.jsonl"
    audits, evaluations, results = [], [], []
    for arm in ("curved", "rectified"):
        checkpoint = root / "training" / arm / "checkpoint_model_001000/model.pt"
        name = f"{arm}_main_ffe_noema_single_sample"
        audits.append([
            args.python, str(SCRIPTS / "audit_noema_checkpoint.py"),
            "--checkpoint", str(checkpoint), "--expected_step", "1000",
            "--output_path", str(output / f"{arm}_checkpoint_audit.json"),
        ])
        evaluations.append([
            "bash", str(SCRIPTS / "run_vbench_condition.sh"),
            "--name", name,
            "--config_path", str(SCRIPTS / "configs/eval_all1.yaml"),
            "--checkpoint_path", str(checkpoint),
            "--method", "framewise", "--schedule", "main_ffe",
            "--prompt_path", str(prompts), "--manifest_path", str(manifest),
            "--full_info_path", str(Path(args.full_info_path).resolve()),
            "--output_root", str(output / name), "--gpus", args.gpus,
            "--vbench_python", args.vbench_python,
            "--samples_per_prompt", "1", "--require_no_ema",
            "--python", args.python,
        ])
        results += ["--result", f"{arm}_main_ffe=" + str(
            output / name / "vbench" / f"{name}_eval_results.json")]
    summary = [
        args.python, str(SCRIPTS / "summarize_single_seed_vbench.py"),
        *results, "--comparison",
        "rectification_gain_main_ffe=rectified_main_ffe,curved_main_ffe",
        "--output_path", str(output / "curvature_main_ffe_summary.json"),
    ]
    return output, audits + evaluations + [summary]


def validate_inputs(args):
    root = Path(args.experiment_root).resolve()
    for arm in ("curved", "rectified"):
        checkpoint = root / "training" / arm / "checkpoint_model_001000/model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    for path in (args.full_info_path, args.python, args.vbench_python):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Supply an existing absolute file path: {path}")
    prompts_path = root / "manifests/vbench_prompts.txt"
    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
    manifest = root / "manifests/vbench_single_sample_seed0.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(prompts) != 944 or len(records) != 944:
        raise ValueError("Reuse the original 944-prompts, one-sample manifest")
    prompt_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    for index, (prompt, record) in enumerate(zip(prompts, records)):
        expected = {
            "prompt_index": index, "sample_index": 0, "seed": index,
            "prompt": prompt, "output_name": f"{prompt}-0.mp4",
            "prompt_file_sha256": prompt_hash,
        }
        if record != expected:
            raise ValueError(f"Original seed-0 manifest mismatch at row {index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True,
                        help="Original curvature_control root, including training/ and manifests/")
    parser.add_argument("--full-info-path", required=True)
    parser.add_argument("--vbench-python", required=True)
    parser.add_argument("--output", help="New empty output directory; default: ROOT/vbench_main_ffe")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 8 or any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != 8:
        parser.error("Use eight distinct GPU IDs, matching the original evaluation")
    output, commands = build_plan(args)
    print("2 evaluations; step-1000 raw; original 944x1; framewise; "
          "first latent frame 4 steps, remaining 20 frames 1 step. No training.", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.execute:
        print("DRY RUN: no model/GPU access and no output files created.")
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("Run --execute in a dedicated tmux session")
    validate_inputs(args)
    if output.exists():
        raise FileExistsError(f"Preserve existing results; output already exists: {output}")
    # Check for other sessions before creating output or loading checkpoints.
    from experiments.rebuttal.run_sharded_inference import require_idle_gpus
    require_idle_gpus([int(gpu) for gpu in gpus])
    output.mkdir()
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    for command in commands:
        subprocess.run(command, cwd=REPO, check=True)
    print(f"Completed FFE evaluation: {output}")


if __name__ == "__main__":
    main()
