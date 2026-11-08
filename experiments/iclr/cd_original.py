"""Run the historical 9-frame pure-CD recipe; reuse hash-verified trajectories."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from experiments.iclr.cd import cd_command
from experiments.iclr.cd_original_protocol import (PROFILE, REFERENCE_SHA256, audit_metrics,
                                                  validate_config, validate_done_pair, verify_source)
from experiments.iclr.launch import parse_gpus, require_idle, start_tmux
from experiments.iclr.protocol import REPO, atomic_json, sha256_file
from utils.config import load_config

CONFIG = REPO / "experiments/iclr/configs/cd_original.yaml"


def preflight(data, checkpoint, config):
    """CPU-only audit; never regenerates trajectories or rewrites their manifest."""
    import torch
    from experiments.rebuttal.curvature_intervention import CurvatureTrajectoryDataset, MANIFEST_NAME, rectify_trajectory
    from experiments.rebuttal.audit_curvature_cd_design import resolve_eval_timesteps
    validate_config(config)
    data, checkpoint = Path(data).resolve(), Path(checkpoint).resolve()
    manifest_path = data / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    digest = verify_source(checkpoint, manifest)
    dataset = CurvatureTrajectoryDataset(str(data), "curved")
    size = len(dataset)
    differences = []
    try:
        if size < 8:
            raise ValueError("original CD requires at least the eight-row global batch")
        for index in range(min(8, size)):
            left = dataset[index]["trajectory"][:, :9]
            right = rectify_trajectory(left, dataset.selected_timesteps)
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError("nonfinite original CD trajectory")
            for point in (0, -2, -1):
                if not torch.equal(left[point], right[point]):
                    raise ValueError("CD treatment changed noise, endpoint or clean history")
            differences.append((left.float() - right.float()).square().flatten(1).mean(1))
    finally:
        dataset.env.close()
    mean_difference = torch.stack(differences).mean(0).tolist()
    if not all(value > 0 for value in mean_difference[1:4]):
        raise ValueError("trajectory rectification did not change all three interior states in the audited 9-frame window")
    schedules = {}
    for schedule in ("all1", "all4"):
        _, resolved = resolve_eval_timesteps(REPO / f"experiments/rebuttal/configs/eval_{schedule}.yaml")
        expected = manifest["selected_timesteps"][:1 if schedule == "all1" else 4]
        if len(resolved) != len(expected) or any(abs(a - b) > 1e-4 for a, b in zip(resolved, expected)):
            raise ValueError("original CD evaluation schedule differs from trajectory inputs")
        schedules[schedule] = resolved
    return {"status": "pass", "iclr_cd_profile": PROFILE,
            "reference_config_sha256": REFERENCE_SHA256, "config": OmegaConf.to_container(config, resolve=True),
            "trainer_source_sha256": sha256_file(REPO / "experiments/rebuttal/train_curvature_cd.py"),
            "initial_generator_sha256": digest, "dataset_manifest_sha256": sha256_file(manifest_path),
            "data_path": str(data), "initial_generator_ckpt": str(checkpoint),
            "num_training_rows": size, "training_frames": 9, "stored_frames": 21,
            "rows_audited": len(differences), "mean_intervention_mse": mean_difference,
            "resolved_eval_timesteps": schedules, "trajectories_reused": True,
            "historical_initialization_match": "unverified; matching the stored source proves reuse compatibility only",
            "historical_score_reproduction_guaranteed": False}


def completed_pair(output, *, mode="formal"):
    done = [json.loads((Path(output) / arm / "training.done").read_text()) for arm in ("curved", "rectified")]
    controls = validate_done_pair(*done, mode=mode)
    metrics = {arm: audit_metrics(Path(output) / arm / "metrics.jsonl", run)
               for arm, run in zip(("curved", "rectified"), done)}
    return {"status": "pass", "controls": controls, "actual_updates": metrics,
            "reference_config_sha256": REFERENCE_SHA256,
            "historical_initialization_match": "unverified", "historical_score_reproduction_guaranteed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="existing paired trajectory LMDB; source hash must match initialization")
    parser.add_argument("--generator-ckpt", required=True, help="explicit raw source checkpoint; never substitute an unrelated initialization")
    parser.add_argument("--output", required=True, help="new output root, separate from the 21-frame endpoint experiment")
    parser.add_argument("--prompt-cache", default="")
    parser.add_argument("--smoke", action="store_true", help="10 updates per arm; separate directory, not formal results")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--session", default="iclr_cd_original")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--preflight", action="store_true", help="only verify existing assets on CPU and write preflight.json")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    config = load_config(str(CONFIG))
    config.prompt_embedding_cache_path = args.prompt_cache
    if args.smoke:
        config.update(max_steps=10, checkpoint_interval=5, print_interval=1,
                      iclr_cd_mode="smoke", save_initial_checkpoint=True)
    validate_config(config)
    output = Path(args.output).resolve()
    config_path = output / "configs" / "cd_original.yaml"
    commands = [cd_command(args.python, config_path, args.data, args.generator_ckpt,
                           output / arm, arm, 2026) for arm in ("curved", "rectified")]
    for command in commands:
        print(shlex.join(command))
    print(f"ORIGINAL RECIPE: 9 frames; pure CD; endpoint=0; holdout=0; seed=2026; "
          f"{config.max_steps} updates/arm; target EMA=.95; raw all1/all4 evaluation; NO resume")
    if not (args.execute or args.preflight or args.worker):
        print("DRY RUN: two runs only; no asset reads, GPU queries or process launches")
        return
    if config_path.exists():
        raise FileExistsError("original CD output already used; choose a new directory, never pretend to resume")
    for arm in ("curved", "rectified"):
        if (output / arm).exists():
            raise FileExistsError("original CD arm directory already exists; use a fresh output")
    if not args.worker:
        audit = preflight(args.data, args.generator_ckpt, config)
        record = output / "preflight.json"
        if record.exists() and json.loads(record.read_text()) != audit:
            raise ValueError("output belongs to different original CD inputs or mode")
        atomic_json(record, audit)
        print(f"Asset audit PASS; reuse {audit['num_training_rows']} trajectories, train on first 9 frames.")
        if args.preflight:
            return
        require_idle(gpus)
        argv = [args.python, "-m", "experiments.iclr.cd_original", "--data", str(Path(args.data).resolve()),
                "--generator-ckpt", str(Path(args.generator_ckpt).resolve()), "--output", str(output),
                "--gpus", args.gpus, "--python", args.python, "--worker"]
        if args.smoke:
            argv.append("--smoke")
        if args.prompt_cache:
            argv += ["--prompt-cache", args.prompt_cache]
        start_tmux(args.session, argv, output / "logs" / "train.log")
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("original CD worker requires tmux")
    recorded = json.loads((output / "preflight.json").read_text())
    if (recorded["config"] != OmegaConf.to_container(config, resolve=True)
            or recorded["data_path"] != str(Path(args.data).resolve())
            or recorded["initial_generator_ckpt"] != str(Path(args.generator_ckpt).resolve())):
        raise ValueError("original CD worker inputs differ from preflight")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_path)
    for command in commands:
        require_idle(gpus)
        subprocess.run(command, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}, check=True)
    atomic_json(output / "paired_training_audit.json", completed_pair(output, mode=config.iclr_cd_mode))


if __name__ == "__main__":
    main()
