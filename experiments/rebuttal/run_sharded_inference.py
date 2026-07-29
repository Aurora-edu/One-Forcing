#!/usr/bin/env python3
"""Export one fixed evaluation manifest concurrently over multiple GPUs."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gpus(value: str):
    parts = value.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("--gpus must be comma-separated non-negative integers")
    gpus = [int(part) for part in parts]
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"--gpus contains duplicates: {value}")
    return gpus


def require_idle_gpus(gpus):
    for gpu in gpus:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if pids:
            raise RuntimeError(
                f"GPU {gpu} already has compute process(es) {pids}; refusing "
                "to overlap another session."
            )


def count_selected_manifest_records(path: Path, limit: int):
    count = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if limit > 0:
        count = min(count, limit)
    if count < 1:
        raise ValueError(f"Manifest contains no selected records: {path}")
    return count


def build_intent(args, gpus, selected_records):
    config_path = Path(args.config_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    prompt_path = Path(args.prompt_path).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    extended_prompt_path = (
        Path(args.extended_prompt_path).resolve()
        if args.extended_prompt_path
        else None
    )
    resolved_config = OmegaConf.to_yaml(
        load_config(str(config_path)),
        resolve=True,
        sort_keys=True,
    )
    return {
        "schema_version": 1,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config_sha256": hashlib.sha256(
            resolved_config.encode("utf-8")
        ).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "extended_prompt_path": (
            str(extended_prompt_path) if extended_prompt_path is not None else None
        ),
        "extended_prompt_sha256": (
            sha256_file(extended_prompt_path)
            if extended_prompt_path is not None
            else None
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_manifest_records": selected_records,
        "method": args.method,
        "schedule": args.schedule,
        "num_output_frames": args.num_output_frames,
        "fps": args.fps,
        "use_ema": bool(args.use_ema),
        "limit": args.limit,
        "num_shards": len(gpus),
    }


def initialize_or_validate_intent(output_folder: Path, intent):
    output_folder.mkdir(parents=True, exist_ok=True)
    intent_path = output_folder / "export.intent.json"
    existing_videos = list(output_folder.glob("*.mp4"))
    if intent_path.is_file():
        existing = json.loads(intent_path.read_text(encoding="utf-8"))
        if existing != intent:
            raise ValueError(
                f"{output_folder} belongs to a different export. Existing intent "
                "does not match this checkpoint/config/manifest."
            )
        return
    if existing_videos:
        raise ValueError(
            f"{output_folder} contains {len(existing_videos)} videos but no "
            "export.intent.json; refusing to reuse videos with unknown provenance."
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(intent_path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as fp:
        json.dump(intent, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())


def terminate_children(children):
    for process, _ in children:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process, _ in children:
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--extended_prompt_path", default="")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--method", choices=["framewise", "chunkwise"], default="framewise")
    parser.add_argument("--schedule", choices=["all1", "ffe", "all4"], default="ffe")
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--prompt_embedding_cache_path", default="")
    parser.add_argument("--offload_generator_before_decode", action="store_true")
    parser.add_argument("--streaming_decode", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.num_output_frames < 1 or args.fps < 1:
        raise ValueError("--num_output_frames and --fps must be positive")
    if args.streaming_decode and not args.offload_generator_before_decode:
        raise ValueError(
            "--streaming_decode requires --offload_generator_before_decode"
        )
    paths = [
        Path(args.config_path),
        Path(args.checkpoint_path),
        Path(args.prompt_path),
        Path(args.manifest_path),
    ]
    if args.extended_prompt_path:
        paths.append(Path(args.extended_prompt_path))
    if args.prompt_embedding_cache_path:
        paths.append(Path(args.prompt_embedding_cache_path) / "data.mdb")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sharded-inference inputs: {missing}")

    gpus = parse_gpus(args.gpus)
    require_idle_gpus(gpus)
    manifest_path = Path(args.manifest_path).resolve()
    selected_records = count_selected_manifest_records(manifest_path, args.limit)
    output_folder = Path(args.output_folder).resolve()
    intent = build_intent(args, gpus, selected_records)
    initialize_or_validate_intent(output_folder, intent)

    children = []
    try:
        for shard_index, gpu in enumerate(gpus):
            log_path = output_folder / (
                f"export.shard_{shard_index:02d}_of_{len(gpus):02d}.log"
            )
            command = [
                "bash",
                str(REPO_ROOT / "scripts" / "infer.sh"),
                "--method",
                args.method,
                "--schedule",
                args.schedule,
                "--config_path",
                str(Path(args.config_path).resolve()),
                "--checkpoint_path",
                str(Path(args.checkpoint_path).resolve()),
                "--prompt_path",
                str(Path(args.prompt_path).resolve()),
                "--manifest_path",
                str(manifest_path),
                "--output_folder",
                str(output_folder),
                "--num_output_frames",
                str(args.num_output_frames),
                "--fps",
                str(args.fps),
                "--limit",
                str(args.limit),
                "--shard_index",
                str(shard_index),
                "--num_shards",
                str(len(gpus)),
                "--gpu_id",
                str(gpu),
            ]
            if args.extended_prompt_path:
                command.extend(
                    [
                        "--extended_prompt_path",
                        str(Path(args.extended_prompt_path).resolve()),
                    ]
                )
            if args.use_ema:
                command.append("--use_ema")
            if args.prompt_embedding_cache_path:
                command.extend(
                    [
                        "--prompt_embedding_cache_path",
                        str(Path(args.prompt_embedding_cache_path).resolve()),
                    ]
                )
            if args.offload_generator_before_decode:
                command.append("--offload_generator_before_decode")
            if args.streaming_decode:
                command.append("--streaming_decode")
            environment = os.environ.copy()
            environment["PYTHON_BIN"] = args.python
            log_fp = open(log_path, "a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append((process, log_fp))
            print(
                f"Started shard {shard_index}/{len(gpus)} on GPU {gpu}: "
                f"pid={process.pid}, log={log_path}",
                flush=True,
            )

        failed = []
        while True:
            pending = 0
            for shard_index, (process, _) in enumerate(children):
                return_code = process.poll()
                if return_code is None:
                    pending += 1
                elif return_code != 0:
                    failed.append((shard_index, return_code))
            if failed:
                terminate_children(children)
                break
            if pending == 0:
                break
            time.sleep(1)
        if failed:
            raise subprocess.CalledProcessError(
                failed[0][1],
                f"inference shard {failed[0][0]}",
            )
    except BaseException:
        terminate_children(children)
        raise
    finally:
        for _, log_fp in children:
            log_fp.close()

    validation_command = [
        args.python,
        str(SCRIPT_DIR / "validate_sharded_export.py"),
        "--output_folder",
        str(output_folder),
        "--manifest_path",
        str(manifest_path),
        "--checkpoint_path",
        str(Path(args.checkpoint_path).resolve()),
        "--num_shards",
        str(len(gpus)),
        "--num_output_frames",
        str(args.num_output_frames),
        "--fps",
        str(args.fps),
        "--limit",
        str(args.limit),
        "--expected_weight_source",
        "generator_ema" if args.use_ema else "generator",
    ]
    subprocess.run(validation_command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
