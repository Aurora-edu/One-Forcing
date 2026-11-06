"""Plan or launch a sequential 8-GPU matrix in tmux. Default is CPU dry-run."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from experiments.iclr.protocol import REPO, validate_matrix
from utils.config import load_config


def parse_gpus(value):
    parts = value.split(",")
    if len(parts) != 8 or any(not x.isdigit() for x in parts) or len(set(parts)) != 8:
        raise ValueError("ICLR launcher requires 8 distinct GPU IDs")
    return parts


def require_idle(gpus):
    # Fail closed on nvidia-smi errors; never signal another session's PID.
    for gpu in gpus:
        result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        if result.stdout.strip():
            raise RuntimeError(f"GPU {gpu} busy: {result.stdout.strip()}; no processes were stopped")


def start_tmux(session, argv, log):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("unsafe tmux session name")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise FileExistsError(f"tmux session already exists: {session}")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    # Quote every shell argument; logging stays outside the run's data directory.
    shell = "cd " + shlex.quote(str(REPO)) + " && exec " + shlex.join(argv)
    shell += " >> " + shlex.quote(str(log)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"tmux attach -t {session}\nlog: {log}")


def train_command(config, output, python, resume=""):
    command = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               "--module", "experiments.iclr.train", "--config", str(config), "--output", str(output)]
    if resume:
        command += ["--resume", str(resume)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/runs")
    parser.add_argument("--run", default="all", help="one matrix run name, or all (12 sequential runs)")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--session", default="iclr_matrix")
    parser.add_argument("--resume", default="", help="complete ICLR checkpoint; requires --run NAME")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    matrix_path = Path(args.matrix).resolve()
    manifest = json.loads(matrix_path.read_text())
    configs = [load_config(str(matrix_path.parent / r["config"])) for r in manifest["runs"]]
    validate_matrix(configs)
    if manifest.get("configs") != [dict_config(c) for c in configs]:
        raise ValueError("matrix manifest and YAML configs disagree; regenerate the matrix")
    selected = [(r, c) for r, c in zip(manifest["runs"], configs)
                if args.run == "all" or r["name"] == args.run]
    if not selected or (args.resume and args.run == "all"):
        raise ValueError("select one known run for resume, or all for the 12-run matrix")
    commands = [train_command(matrix_path.parent / r["config"], Path(args.output) / r["name"],
                              args.python, args.resume) for r, _ in selected]
    if not args.execute and not args.worker:
        for command in commands:
            print(shlex.join(command))
        print(f"DRY RUN: {len(commands)} runs; no GPU queries, model loads or processes launched")
        return
    if not args.worker:
        require_idle(gpus)
        argv = [args.python, "-m", "experiments.iclr.launch", "--matrix", str(matrix_path),
                "--output", str(Path(args.output).resolve()), "--run", args.run,
                "--gpus", args.gpus, "--python", args.python, "--worker"]
        if args.resume:
            argv += ["--resume", str(Path(args.resume).resolve())]
        start_tmux(args.session, argv, Path(args.output).resolve() / "logs" / f"{args.session}.log")
        return
    if not os.environ.get("TMUX"):
        raise RuntimeError("training worker requires tmux")
    for (record, config), command in zip(selected, commands):
        require_idle(gpus)
        for path in (Path(config.generator_ckpt), Path(config.teacher_model_path),
                     Path(config.data_path) / "data.mdb"):
            if not path.exists():
                raise FileNotFoundError(path)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}
        print(f"START {record['name']}; monitor steps 1-10 before leaving unattended", flush=True)
        subprocess.run(command, cwd=REPO, env=environment, check=True)


def dict_config(config):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(config, resolve=True)


if __name__ == "__main__":
    main()
