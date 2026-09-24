#!/usr/bin/env python3
"""Launch one arm on pinned, unmodified GitHub main code in tmux."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.official_main_pair import (
    CONFIG_NAMES, OFFICIAL_COMMIT, check, official_command,
)


def _path(path: Path, kind: str) -> Path:
    path = path.expanduser().resolve()
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(path)
    if kind == "dir" and not path.is_dir():
        raise NotADirectoryError(path)
    return path


def _gpu_ids(value: str) -> list[str]:
    ids = value.split(",")
    if len(ids) != 8 or len(set(ids)) != 8 or any(not re.fullmatch(r"\d+", item) for item in ids):
        raise ValueError("Use eight distinct numeric GPU IDs to retain the official global batch")
    for gpu in ids:
        result = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True,
        )
        if result.stdout.strip():
            raise RuntimeError(f"GPU {gpu} already has compute process(es): {result.stdout.strip()}")
    return ids


def launch(args: argparse.Namespace) -> dict:
    root = _path(args.official_root, "dir")
    pair_dir = _path(args.pair_dir, "dir")
    manifest = check(root, pair_dir)
    python = str(_path(args.python, "file"))
    generator_ckpt = _path(args.generator_ckpt, "file")
    teacher = _path(args.teacher_model_path, "dir")
    data = _path(args.data_path, "dir")
    _path(data / "data.mdb", "file")
    wan_1_3b = _path(args.wan_1_3b_path, "dir")
    _path(wan_1_3b / "models_t5_umt5-xxl-enc-bf16.pth", "file")
    _path(wan_1_3b / "Wan2.1_VAE.pth", "file")
    gpu_ids = _gpu_ids(args.gpus)
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required for long training")

    # The official source uses this relative Wan 1.3B path. An ignored asset
    # symlink is allowed; no tracked official source file is changed.
    official_asset = root / "wan_models/Wan2.1-T2V-1.3B"
    if official_asset.exists() or official_asset.is_symlink():
        if official_asset.resolve() != wan_1_3b:
            raise ValueError(f"Official Wan 1.3B asset resolves elsewhere: {official_asset}")
    else:
        official_asset.parent.mkdir(parents=True, exist_ok=True)
        official_asset.symlink_to(wan_1_3b, target_is_directory=True)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing existing run directory: {run_dir}")
    session = args.session or f"of_official_{args.arm}_s{manifest['shared_effective_seed']}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session):
        raise ValueError("Invalid tmux session name")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise FileExistsError(f"tmux session already exists: {session}")

    asset_paths = {
        "generator_ckpt": str(generator_ckpt),
        "teacher_model_path": str(teacher),
        "data_path": str(data),
        "wan_1_3b_path": str(wan_1_3b),
        "cuda_visible_devices": ",".join(gpu_ids),
    }
    config_path = pair_dir / CONFIG_NAMES[args.arm]
    command = official_command(python, config_path, run_dir, asset_paths)
    run_dir.mkdir(parents=True)
    launch_record = {
        "schema_version": 1,
        "arm": args.arm,
        "official_main_commit": OFFICIAL_COMMIT,
        "config_path": str(config_path.resolve()),
        "config_sha256": manifest["configs_sha256"][args.arm],
        "world_size": 8,
        "run_dir": str(run_dir),
        "python": python,
        "asset_paths": asset_paths,
        "asset_stats": {
            "generator_ckpt_size": generator_ckpt.stat().st_size,
            "data_mdb_size": (data / "data.mdb").stat().st_size,
        },
        "command": command,
        "session": session,
        "rebuttal_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (run_dir / "official_launch.json").write_text(
        json.dumps(launch_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shell_command = (
        "set -o pipefail; "
        f"cd {shlex.quote(str(root))}; "
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(gpu_ids))}; "
        f"{shlex.join(command)} 2>&1 | tee {shlex.quote(str(run_dir / 'train.log'))}"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_command], check=True)
    return {"session": session, "run_dir": str(run_dir), "log": str(run_dir / "train.log")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("full", "dmd"), required=True)
    parser.add_argument("--official_root", type=Path, required=True)
    parser.add_argument("--pair_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--generator_ckpt", type=Path, required=True)
    parser.add_argument("--teacher_model_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--wan_1_3b_path", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--session", default="")
    print(json.dumps(launch(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
