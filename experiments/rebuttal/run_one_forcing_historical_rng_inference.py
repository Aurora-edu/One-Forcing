#!/usr/bin/env python3
"""Run raw One-Forcing on all GPUs while matching historical SF RNG streams."""

from __future__ import annotations

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

from experiments.rebuttal.resolve_all_gpus import (  # noqa: E402
    detect_gpus,
    resolve_requested,
)
from utils.config import load_config  # noqa: E402


RNG_PROTOCOL = "self_forcing_two_shard_seed0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gpus(value: str) -> list[int]:
    parts = value.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("--gpus must contain comma-separated physical GPU indices")
    selected = [int(part) for part in parts]
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate GPU indices: {value}")
    return selected


def audit_all4_config(path: Path) -> str:
    config = load_config(str(path))
    checks = {
        "denoising_step_list": (list(config.denoising_step_list), [1000, 750, 500, 250]),
        "num_frame_per_block": (int(config.num_frame_per_block), 1),
        "rollout_schedule": (str(config.rollout_schedule), "fixed"),
        "local_attn_size": (int(config.model_kwargs.local_attn_size), 21),
        "sink_size": (int(config.model_kwargs.sink_size), 0),
    }
    mismatches = {
        key: values for key, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"One-Forcing all4 config mismatch: {mismatches}")
    resolved = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def count_manifest(path: Path) -> int:
    count = sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if count != 944:
        raise ValueError(f"Expected 944 manifest records, found {count}")
    return count


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def terminate_children(children: list[tuple[subprocess.Popen, object]]) -> None:
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


def validate_export(
    python: Path,
    output_folder: Path,
    manifest: Path,
    checkpoint: Path,
    num_shards: int,
) -> None:
    subprocess.run(
        [
            str(python),
            str(SCRIPT_DIR / "validate_sharded_export.py"),
            "--output_folder",
            str(output_folder),
            "--manifest_path",
            str(manifest),
            "--checkpoint_path",
            str(checkpoint),
            "--num_shards",
            str(num_shards),
            "--num_output_frames",
            "21",
            "--fps",
            "16",
            "--expected_weight_source",
            "generator",
            "--expected_rng_protocol",
            RNG_PROTOCOL,
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--qwen_rewrite_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    paths = {
        "config": Path(args.config_path).resolve(),
        "checkpoint": Path(args.checkpoint_path).resolve(),
        "prompt": Path(args.prompt_path).resolve(),
        "rewrite": Path(args.qwen_rewrite_path).resolve(),
        "manifest": Path(args.manifest_path).resolve(),
        "python": Path(args.python).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    requested_gpus = parse_gpus(args.gpus)
    inventory = detect_gpus()
    detected_indices = [gpu["index"] for gpu in inventory]
    gpus = resolve_requested(args.gpus, detected_indices)
    if requested_gpus != gpus:
        raise ValueError(
            f"GPU order must be canonical detected order: {requested_gpus} != {gpus}"
        )
    if len(gpus) < 2 or len(gpus) % 2:
        raise ValueError(
            "Historical two-stream matching requires an even number of physical GPUs"
        )
    busy = {
        gpu["index"]: gpu["compute_pids_before_launch"]
        for gpu in inventory
        if gpu["compute_pids_before_launch"]
    }
    if busy:
        raise RuntimeError(
            "Some GPUs belong to existing sessions; refusing to interrupt or overlap: "
            f"{busy}"
        )

    resolved_config_sha256 = audit_all4_config(paths["config"])
    selected_records = count_manifest(paths["manifest"])
    output_folder = Path(args.output_folder).resolve()
    intent = {
        "schema_version": 2,
        "config_path": str(paths["config"]),
        "config_sha256": sha256_file(paths["config"]),
        "resolved_config_sha256": resolved_config_sha256,
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_size_bytes": paths["checkpoint"].stat().st_size,
        "checkpoint_mtime_ns": paths["checkpoint"].stat().st_mtime_ns,
        "prompt_path": str(paths["prompt"]),
        "prompt_sha256": sha256_file(paths["prompt"]),
        "extended_prompt_path": str(paths["rewrite"]),
        "extended_prompt_sha256": sha256_file(paths["rewrite"]),
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "selected_manifest_records": selected_records,
        "method": "framewise",
        "schedule": "all4",
        "num_output_frames": 21,
        "fps": 16,
        "use_ema": False,
        "num_shards": len(gpus),
        "gpu_indices": gpus,
        "historical_num_rng_streams": 2,
        "prompt_sharding": "all_gpu_stride_preserving_even_odd_rng_streams",
        "rng_protocol": RNG_PROTOCOL,
        "rng_state_reset_per_record": True,
        "process_seed": 0,
        "initial_noise_seed_scope": "index_within_historical_even_odd_shard",
    }
    intent_path = output_folder / "export.intent.json"
    done_path = output_folder / "export.done"
    if intent_path.is_file():
        existing = json.loads(intent_path.read_text(encoding="utf-8"))
        if existing != intent:
            raise ValueError("Existing export intent differs from this invocation")
    else:
        if output_folder.exists() and any(output_folder.iterdir()):
            raise RuntimeError(
                f"Output directory has files but no audited intent: {output_folder}"
            )
        output_folder.mkdir(parents=True, exist_ok=True)
        atomic_write_json(intent_path, intent)
    if done_path.is_file():
        validate_export(
            paths["python"], output_folder, paths["manifest"], paths["checkpoint"], len(gpus)
        )
        print(f"PASS: existing complete all-GPU export revalidated: {output_folder}")
        return

    children: list[tuple[subprocess.Popen, object]] = []
    try:
        for shard_index, gpu in enumerate(gpus):
            log_path = output_folder / (
                f"export.shard_{shard_index:02d}_of_{len(gpus):02d}.log"
            )
            command = [
                str(paths["python"]),
                str(REPO_ROOT / "scripts" / "export_videos.py"),
                "--config_path",
                str(paths["config"]),
                "--checkpoint_path",
                str(paths["checkpoint"]),
                "--prompt_path",
                str(paths["prompt"]),
                "--extended_prompt_path",
                str(paths["rewrite"]),
                "--manifest_path",
                str(paths["manifest"]),
                "--output_folder",
                str(output_folder),
                "--num_output_frames",
                "21",
                "--seed",
                "0",
                "--num_samples_per_prompt",
                "1",
                "--shard_index",
                str(shard_index),
                "--num_shards",
                str(len(gpus)),
                "--naming",
                "raw_prompt_index",
                "--fps",
                "16",
                "--rng_protocol",
                RNG_PROTOCOL,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_stream = open(log_path, "a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append((process, log_stream))
            print(
                f"Started OF historical-RNG shard {shard_index}/{len(gpus)} "
                f"on GPU {gpu}: pid={process.pid}, log={log_path}",
                flush=True,
            )

        while True:
            states = [process.poll() for process, _ in children]
            failures = [code for code in states if code not in (None, 0)]
            if failures:
                raise subprocess.CalledProcessError(failures[0], "Qwen all4 export")
            if all(code == 0 for code in states):
                break
            time.sleep(1)
    except BaseException:
        terminate_children(children)
        raise
    finally:
        for _, stream in children:
            stream.close()

    validate_export(
        paths["python"], output_folder, paths["manifest"], paths["checkpoint"], len(gpus)
    )
    print(f"PASS: raw/no-EMA all-GPU historical-seed export complete: {output_folder}")


if __name__ == "__main__":
    main()
