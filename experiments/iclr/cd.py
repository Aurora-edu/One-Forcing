"""Paired CD training from scratch to 300/600/900, reusing reviewer trainer.

The reviewer CD trainer does not save optimizer or EMA-target restart state.
This command therefore NEVER offers resume or extends an old 300-step run.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from experiments.iclr.launch import parse_gpus, require_idle, start_tmux
from experiments.iclr.protocol import REPO
from utils.config import load_config


def cd_command(python, config, data, checkpoint, output, intervention, seed):
    return [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
        str(REPO / "experiments/rebuttal/train_curvature_cd.py"), "--config_path", str(config),
        "--data_path", str(data), "--generator_ckpt", str(checkpoint),
        "--output_dir", str(output), "--intervention", intervention, "--seed", str(seed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "experiments/rebuttal/configs/train_curvature_cd_small.yaml"))
    parser.add_argument("--data", required=True, help="paired ODE trajectory LMDB with manifest")
    parser.add_argument("--generator-ckpt", required=True)
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/cd")
    parser.add_argument("--steps", type=int, choices=(300, 600, 900), default=900)
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
    config.max_steps = args.steps
    config.checkpoint_interval = 300
    config.print_interval = 10
    config.seed = args.seed
    config.prompt_embedding_cache_path = args.prompt_cache
    config_path = Path(args.output).resolve() / "configs" / "cd.yaml"
    commands = [cd_command(args.python, config_path, args.data, args.generator_ckpt,
        Path(args.output).resolve() / arm, arm, args.seed) for arm in ("curved", "rectified")]
    for command in commands:
        print(shlex.join(command))
    print(f"FROM SCRATCH: {args.steps} updates per arm; saves every 300; training frames={config.training_num_frames}; NO resume")
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


if __name__ == "__main__":
    main()
