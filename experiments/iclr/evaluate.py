"""Plan/run raw paired VBench cells, including chunk curves and CD all1/all4."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.iclr.protocol import REPO, atomic_json, sha256_file


def cd_cells(original, rectified, config):
    return [{"name": f"cd_{arm}_{schedule}", "checkpoint": checkpoint,
             "config": config, "method": "framewise", "schedule": schedule}
            for arm, checkpoint in (("original", original), ("rectified", rectified))
            for schedule in ("all1", "all4")]


def curve_cells(index, config):
    cells = []
    for line in Path(index).read_text().splitlines():
        row = json.loads(line)
        if row["weight_source"] != "generator" or row["schedule"] != "all1":
            raise ValueError("matched curve requires raw/all1 checkpoints")
        if row["method"] not in ("framewise", "chunkwise"):
            raise ValueError("unknown curve method")
        cells.append({"name": f"step{row['step']:06d}_{row['method']}",
            "step": row["step"],
            "checkpoint": row["checkpoint"], "config": config,
            "method": row["method"], "schedule": "all1",
            "training_seconds": row["training_seconds"],
            "generator_updates": row["generator_updates"], "critic_updates": row["critic_updates"],
            "seed": row["seed"], "r1_weight": row["r1_weight"], "r2_weight": row["r2_weight"],
            "mode": row["mode"], "experiment_id": row["experiment_id"],
            "budget_seconds": row["budget_seconds"],
            "fractions": row["fractions"]})
    if len({cell["name"] for cell in cells}) != len(cells):
        raise ValueError("duplicate checkpoints in curve index; supply one run's index")
    return cells


def validate_manifest(path, prompt_path, *, expected_prompts=944):
    prompts = Path(prompt_path).read_text().splitlines()
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(prompts) != expected_prompts or len(rows) != expected_prompts:
        raise ValueError(f"expected {expected_prompts} original prompts and one sample each")
    names = set()
    for index, row in enumerate(rows):
        if (row["prompt_index"] != index or row["sample_index"] != 0
                or row["prompt"] != prompts[index] or row.get("extended_prompt") is not None
                or row["output_name"] != f"{prompts[index]}-0.mp4"):
            raise ValueError(f"unpaired or rewritten prompt at record {index}")
        if row["output_name"] in names or int(row["seed"]) < 0:
            raise ValueError("duplicate video name or invalid seed")
        names.add(row["output_name"])
    return {"manifest_sha256": sha256_file(path), "prompt_sha256": sha256_file(prompt_path),
            "num_prompts": len(prompts), "samples_per_prompt": 1, "conditioning": "original"}


def command_for_cell(cell, args):
    return ["bash", str(REPO / "experiments/rebuttal/run_vbench_condition.sh"),
        "--name", cell["name"], "--config_path", str(cell["config"]),
        "--checkpoint_path", str(cell["checkpoint"]), "--method", cell["method"],
        "--schedule", cell["schedule"], "--prompt_path", args.prompts,
        "--manifest_path", args.manifest, "--full_info_path", args.full_info,
        "--output_root", str(Path(args.output) / cell["name"]), "--gpus", args.gpus,
        "--vbench_python", args.vbench_python, "--python", args.python,
        "--samples_per_prompt", "1", "--require_no_ema"]


def record_cell(path, payload):
    """Never relabel existing evaluation outputs with a different experiment."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"evaluation directory belongs to another cell: {path}")
    else:
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError(f"existing outputs have no ICLR provenance: {path.parent}")
        atomic_json(path, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    kind = parser.add_subparsers(dest="kind", required=True)
    cd = kind.add_parser("cd", help="original/rectified x all1/all4, no FFE")
    cd.add_argument("--original", required=True)
    cd.add_argument("--rectified", required=True)
    curve = kind.add_parser("curve", help="timepoint index from one frame OR chunk run")
    curve.add_argument("--index", required=True)
    single = kind.add_parser("single", help="one checkpoint")
    single.add_argument("--checkpoint", required=True)
    single.add_argument("--method", choices=("framewise", "chunkwise"), required=True)
    for sub in (cd, curve, single):
        sub.add_argument("--config", default=str(REPO / "experiments/rebuttal/configs/eval_all1.yaml"))
        sub.add_argument("--prompts", required=True)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--full-info", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
        sub.add_argument("--python", default=sys.executable)
        sub.add_argument("--vbench-python", default="VBENCH_PYTHON")
        group = sub.add_mutually_exclusive_group()
        group.add_argument("--execute", action="store_true")
        group.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.kind == "cd":
        cells = cd_cells(args.original, args.rectified, args.config)
    elif args.kind == "curve":
        cells = curve_cells(args.index, args.config)
    else:
        cells = [{"name": f"{args.method}_all1", "config": args.config,
                  "checkpoint": args.checkpoint, "method": args.method, "schedule": "all1"}]
    commands = [command_for_cell(cell, args) for cell in cells]
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        print("DRY RUN: raw generator, original prompts, 1 sample/prompt; no GPU/model access")
        return
    from experiments.iclr.launch import parse_gpus, require_idle
    require_idle(parse_gpus(args.gpus))
    protocol = validate_manifest(args.manifest, args.prompts)
    protocol["full_info_sha256"] = sha256_file(args.full_info)
    for cell, command in zip(cells, commands):
        if not Path(cell["checkpoint"]).is_file():
            raise FileNotFoundError(cell["checkpoint"])
        root = Path(args.output) / cell["name"]
        # Keep time and update coordinates next to score outputs for curve plotting.
        record_cell(root / "iclr_cell.json", {**cell, **protocol, "use_ema": False})
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
