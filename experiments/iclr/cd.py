"""Endpoint-anchored paired CD: pinned AR init, 21 frames, 300 updates per arm.

The reviewer CD trainer does not save optimizer or EMA-target restart state.
This command therefore NEVER offers resume or extends an old 300-step run.
Every update includes highest-noise endpoint supervision. History remains
teacher-forced to preserve the strict paired intervention; diagnose it separately.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from experiments.iclr.launch import parse_gpus, require_idle, start_tmux
from experiments.iclr.protocol import REPO
from experiments.iclr.cd_protocol import AR_DEFAULT, validate_one_step_config, validate_done_pair
from utils.config import load_config


def cd_command(python, config, data, checkpoint, output, intervention, seed):
    return [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
        str(REPO / "experiments/rebuttal/train_curvature_cd.py"), "--config_path", str(config),
        "--data_path", str(data), "--generator_ckpt", str(checkpoint),
        "--output_dir", str(output), "--intervention", intervention, "--seed", str(seed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "experiments/iclr/configs/cd_one_step.yaml"))
    parser.add_argument("--data", required=True, help="paired ODE trajectory LMDB with manifest")
    parser.add_argument("--generator-ckpt", default=str(REPO / AR_DEFAULT),
                        help="published Causal-Forcing framewise/ar_diffusion.pt; SHA256 verified")
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/cd")
    parser.add_argument("--steps", type=int, choices=(300,), default=300,
                        help="fixed formal budget; do not tune it against VBench")
    parser.add_argument("--smoke", action="store_true", help="two 10-step smoke runs, never formal results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-cache", default="")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--session", default="iclr_cd")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    config.max_steps = 10 if args.smoke else args.steps
    config.checkpoint_interval = 5 if args.smoke else 50
    config.iclr_cd_mode = "smoke" if args.smoke else "formal"
    config.print_interval = 10
    config.seed = args.seed
    config.prompt_embedding_cache_path = args.prompt_cache
    validate_one_step_config(config)
    config_path = Path(args.output).resolve() / "configs" / "cd.yaml"
    commands = [cd_command(args.python, config_path, args.data, args.generator_ckpt,
        Path(args.output).resolve() / arm, arm, args.seed) for arm in ("curved", "rectified")]
    for command in commands:
        print(shlex.join(command))
    print(f"FROM SCRATCH: {config.max_steps} updates per arm; saves every {config.checkpoint_interval}; "
          f"training frames={config.training_num_frames}; endpoint weight=1; holdout=64; pinned AR init; NO resume")
    if not args.execute and not args.worker:
        return
    if not args.worker:
        require_idle(parse_gpus(args.gpus))
        argv = [args.python, "-m", "experiments.iclr.cd", "--config", str(Path(args.config).resolve()),
            "--data", str(Path(args.data).resolve()), "--generator-ckpt", str(Path(args.generator_ckpt).resolve()),
            "--output", str(Path(args.output).resolve()), "--steps", str(args.steps), "--seed", str(args.seed),
            "--python", args.python, "--gpus", args.gpus, "--worker"]
        if args.prompt_cache:
            argv += ["--prompt-cache", str(Path(args.prompt_cache).resolve())]
        if args.smoke:
            argv += ["--smoke"]
        start_tmux(args.session, argv, Path(args.output).resolve() / "logs" / "train.log")
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("CD training worker requires tmux")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        raise FileExistsError("CD output already exists: use a new output root, never pretend to resume")
    OmegaConf.save(config, config_path)
    for command in commands:
        require_idle(parse_gpus(args.gpus))
        subprocess.run(command, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}, check=True)
    if not args.smoke:
        from experiments.iclr.protocol import atomic_json
        root = Path(args.output).resolve()
        controls = validate_done_pair(*[json.loads((root / arm / "training.done").read_text())
                                       for arm in ("curved", "rectified")])
        atomic_json(root / "paired_training_audit.json", {"status": "pass", "controls": controls})


if __name__ == "__main__":
    main()
