"""ICLR training entry point. --dry-run validates config without loading models."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from experiments.iclr.protocol import atomic_json, validate_config
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default="", help="complete ICLR checkpoint DIRECTORY only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config.logdir = str(Path(args.output).resolve())
    config.resume_ckpt = args.resume
    config.config_name = config.experiment_id
    validate_config(config)
    if args.dry_run:
        print(json.dumps({"config": args.config, "mode": config.iclr_mode,
            "expected_world_size": config.iclr_expected_world_size,
            "gradient_frames": config.gradient_num_frames,
            "method": "framewise" if config.num_frame_per_block == 1 else "chunkwise",
            "budget_seconds": config.iclr_budget_seconds, "resume": args.resume,
            "weight_source": "generator", "gpu_execution": False}, indent=2))
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("all ICLR training must run in tmux; use experiments.iclr.launch")
    if int(os.environ.get("RANK", "0")) == 0:
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"use a fresh output directory, including for full resume: {output}")
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output / "resolved_config.yaml", resolve=True)
        atomic_json(output / "source.json", {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()})
    from experiments.iclr.trainer import ICLRTrainer
    trainer = ICLRTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
