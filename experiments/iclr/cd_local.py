"""Train the paired high-noise local progressive CD experiment; evaluate all1 only."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from omegaconf import OmegaConf
from experiments.iclr.cd import cd_command
from experiments.iclr.cd_local_protocol import PROFILE, validate_config
from experiments.iclr.cd_original import preflight, completed_pair
from experiments.iclr.launch import parse_gpus, require_idle, start_tmux
from experiments.iclr.protocol import REPO, atomic_json
from utils.config import load_config
CONFIG = REPO / "experiments/iclr/configs/cd_local_all1.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="existing paired trajectory LMDB; source hash must match initialization")
    parser.add_argument("--generator-ckpt", required=True, help="explicit raw source checkpoint; never substitute an unrelated initialization")
    parser.add_argument("--output", required=True, help="new output root, separate from the 21-frame endpoint experiment")
    parser.add_argument("--seed", type=int, choices=(2026, 2027), default=2026)
    parser.add_argument("--prompt-cache", default="")
    parser.add_argument("--smoke", action="store_true", help="10 updates per arm; separate directory, not formal results")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--session", default="iclr_cd_local_all1")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--preflight", action="store_true", help="only verify existing assets on CPU and write preflight.json")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    config = load_config(str(CONFIG))
    config.prompt_embedding_cache_path = args.prompt_cache
    config.seed = args.seed
    if args.smoke:
        config.update(max_steps=10, checkpoint_interval=5, print_interval=1,
                      iclr_cd_mode="smoke", save_initial_checkpoint=True)
    validate_config(config)
    output = Path(args.output).resolve()
    config_path = output / "configs" / "cd_local_all1.yaml"
    commands = [cd_command(args.python, config_path, args.data, args.generator_ckpt,
                           output / arm, arm, args.seed) for arm in ("curved", "rectified")]
    for command in commands:
        print(shlex.join(command))
    print(f"LOCAL ALL1: 9 frames; pure CD; endpoint=0; holdout=0; seed={args.seed}; "
          f"{config.max_steps} updates/arm; target EMA=.95; raw all1 ONLY; no FFE; ramp=100; NO resume")
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
        argv = [args.python, "-m", "experiments.iclr.cd_local", "--data", str(Path(args.data).resolve()),
                "--generator-ckpt", str(Path(args.generator_ckpt).resolve()), "--output", str(output),
                "--gpus", args.gpus, "--python", args.python, "--seed", str(args.seed), "--worker"]
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
    atomic_json(output / "paired_training_audit.json", completed_pair(output, mode=config.iclr_cd_mode, profile=PROFILE))


if __name__ == "__main__":
    main()
