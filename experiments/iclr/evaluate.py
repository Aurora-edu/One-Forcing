"""Evaluate final step-200 EMA checkpoints and separate raw CD all1/all4 cells."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.iclr.protocol import MAIN_REFERENCE, REPO, atomic_json, sha256_file


def cd_cells(original, rectified, config):
    return [{"name": f"cd_{arm}_{schedule}", "checkpoint": checkpoint,
             "config": config, "method": "framewise", "schedule": schedule}
            for schedule in ("all1", "all4")
            for arm, checkpoint in (("original", original), ("rectified", rectified))]


def audit_cd_checkpoints(original, rectified, config_path, *, training_profile="endpoint_anchored_cd_v1"):
    import torch
    from omegaconf import OmegaConf
    from utils.config import load_config
    from experiments.iclr.cd_protocol import validate_checkpoint
    if training_profile == "original_adjacent_cd_9f_v1":
        from experiments.iclr.cd_original_protocol import validate_done_pair, audit_metrics
    elif training_profile == "endpoint_anchored_cd_v1":
        from experiments.iclr.cd_protocol import validate_done_pair
    else:
        raise ValueError("unknown CD training profile")
    paths = [Path(original), Path(rectified)]
    done = [json.loads((p.parent.parent / "training.done").read_text()) for p in paths]
    controls = validate_done_pair(*done)
    config = load_config(config_path)
    if (OmegaConf.to_container(config.model_kwargs) != controls["config"]["model_kwargs"]
            or config.num_frame_per_block != 1 or config.context_noise != 0
            or config.independent_first_frame or not config.warp_denoising_step):
        raise ValueError("CD evaluation model/history configuration differs from training protocol")
    for path, run in zip(paths, done):
        if training_profile == "original_adjacent_cd_9f_v1":
            audit_metrics(path.parent.parent / "metrics.jsonl", run)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        validate_checkpoint(checkpoint, run, 300)
        del checkpoint
    return {"cd_pair_controls": controls, "cd_checkpoint_step": 300}


def curve_cells(index, config, reference_step=200):
    if reference_step != 200:
        raise ValueError("frame/chunk VBench is restricted to the final step-200 checkpoint")
    cells = []
    for line in Path(index).read_text().splitlines():
        row = json.loads(line)
        if (row.get("profile") != "main_block_ablation_200_v2" or row["schedule"] != "all1"
                or row.get("stop_condition") != "fixed_steps"):
            raise ValueError("matched curve requires main-aligned fixed-step checkpoints; regenerate old configurations")
        if row["step"] != reference_step:
            continue
        initial = row["step"] == 0
        if not initial and "generator_ema" not in row["available_weights"]:
            raise ValueError("EMA is not initialized at this timepoint; do not silently use raw weights")
        if row["method"] not in ("framewise", "chunkwise"):
            raise ValueError("unknown curve method")
        cells.append({"name": f"step{row['step']:06d}_{row['method']}",
            "step": row["step"],
            "checkpoint": row["checkpoint"], "config": config,
            "method": row["method"], "schedule": "main_ffe",
            "use_ema": not initial, "initial_raw_reference": initial,
            "profile": row["profile"], "reference_step": reference_step,
            "training_seconds": row["training_seconds"],
            "generator_updates": row["generator_updates"], "critic_updates": row["critic_updates"],
            "seed": row["seed"], "r1_weight": row["r1_weight"], "r2_weight": row["r2_weight"],
            "mode": row["mode"], "experiment_id": row["experiment_id"],
            "max_steps": row["max_steps"], "stop_condition": row["stop_condition"],
            "milestones": row["milestones"]})
    if len({cell["name"] for cell in cells}) != len(cells):
        raise ValueError("duplicate checkpoints in curve index; supply one run's index")
    if not cells:
        raise ValueError("no matching checkpoints in index")
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
    command = ["bash", str(REPO / "experiments/rebuttal/run_vbench_condition.sh"),
        "--name", cell["name"], "--config_path", str(cell["config"]),
        "--checkpoint_path", str(cell["checkpoint"]), "--method", cell["method"],
        "--schedule", cell["schedule"], "--prompt_path", args.prompts,
        "--manifest_path", args.manifest, "--full_info_path", args.full_info,
        "--output_root", str(Path(args.output) / cell["name"]), "--gpus", args.gpus,
        "--vbench_python", args.vbench_python, "--python", args.python,
        "--samples_per_prompt", str(getattr(args, "samples_per_prompt", 1)),
        "--use_ema" if cell.get("use_ema", False) else "--require_no_ema"]
    if getattr(args, "extended_prompts", ""):
        command += ["--extended_prompt_path", args.extended_prompts]
    return command


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
    cd.add_argument("--training-profile", choices=("endpoint_anchored_cd_v1", "original_adjacent_cd_9f_v1"),
                    default="endpoint_anchored_cd_v1", help="explicitly select and audit the training recipe")
    curve = kind.add_parser("curve", help="final step-200 checkpoint from one frame OR chunk run (legacy command name)")
    curve.add_argument("--index", required=True)
    curve.add_argument("--reference-step", type=int, choices=(200,), default=200,
                       help="final checkpoint to evaluate; only step 200 is supported (default)")
    single = kind.add_parser("single", help="one checkpoint")
    single.add_argument("--checkpoint", required=True)
    single.add_argument("--method", choices=("framewise", "chunkwise"), required=True)
    for sub in (cd, curve, single):
        sub.add_argument("--config", default=str(MAIN_REFERENCE if sub is curve else
                         REPO / "experiments/rebuttal/configs/eval_all1.yaml"))
        sub.add_argument("--prompts", required=True)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--extended-prompts", default="")
        sub.add_argument("--samples-per-prompt", type=int, choices=(1, 5), default=5 if sub is curve else 1)
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
        cells = curve_cells(args.index, args.config, args.reference_step)
    else:
        cells = [{"name": f"{args.method}_all1", "config": args.config,
                  "checkpoint": args.checkpoint, "method": args.method, "schedule": "all1"}]
    commands = [command_for_cell(cell, args) for cell in cells]
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        print("DRY RUN: frame/chunk=step200/main_ffe/EMA/Qwen; CD=raw/all1+all4/original; no GPU/model access")
        return
    if args.kind == "cd":
        provenance = audit_cd_checkpoints(args.original, args.rectified, args.config,
                                         training_profile=args.training_profile)
        for cell in cells:
            cell.update(provenance)
    from experiments.iclr.launch import parse_gpus, require_idle
    require_idle(parse_gpus(args.gpus))
    if args.kind == "curve":
        if args.samples_per_prompt != 5 or not args.extended_prompts:
            raise ValueError("main-aligned endpoints require pinned Qwen rewrites and 5 samples per prompt")
        from experiments.rebuttal.paper_vbench_protocol import audit_files, verify_full_info
        verify_full_info(Path(args.full_info))
        protocol = audit_files(Path(args.prompts), Path(args.extended_prompts), Path(args.manifest), 5)
        protocol["num_prompts"] = protocol["prompt_count"]
    else:
        if args.extended_prompts or args.samples_per_prompt != 1:
            raise ValueError("CD/single protocol requires original prompts and one sample")
        protocol = validate_manifest(args.manifest, args.prompts)
    protocol["full_info_sha256"] = sha256_file(args.full_info)
    for cell, command in zip(cells, commands):
        if not Path(cell["checkpoint"]).is_file():
            raise FileNotFoundError(cell["checkpoint"])
        root = Path(args.output) / cell["name"]
        # Keep actual training costs and update counts next to endpoint scores.
        record_cell(root / "iclr_cell.json", {**cell, **protocol, "use_ema": cell.get("use_ema", False)})
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
