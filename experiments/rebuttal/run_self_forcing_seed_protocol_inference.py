#!/usr/bin/env python3
"""Run raw One-Forcing with the historical two-shard Self-Forcing RNG protocol."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise ValueError(
            "Historical Self-Forcing matching requires exactly two GPU IDs, e.g. 0,1"
        )
    gpus = [int(part) for part in parts]
    if len(set(gpus)) != 2:
        raise ValueError("The two GPU IDs must be different")
    return gpus


def require_idle_gpus(gpus: list[int]) -> None:
    for gpu in gpus:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if pids:
            raise RuntimeError(
                f"GPU {gpu} has compute process(es) {pids}; refusing to overlap another session"
            )


def audit_all4_config(path: Path) -> None:
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


def count_manifest(path: Path) -> int:
    count = sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if count != 944:
        raise ValueError(f"Expected 944 manifest records, found {count}")
    return count


def atomic_write_json(path: Path, payload: dict) -> None:
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
    python: Path, output_folder: Path, manifest: Path, checkpoint: Path
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
            "2",
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
    gpus = parse_gpus(args.gpus)
    audit_all4_config(paths["config"])
    selected_records = count_manifest(paths["manifest"])

    output_folder = Path(args.output_folder).resolve()
    intent = {
        "schema_version": 1,
        "config_path": str(paths["config"]),
        "config_sha256": sha256_file(paths["config"]),
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
        "num_shards": 2,
        "prompt_sharding": "even_odd",
        "rng_protocol": RNG_PROTOCOL,
        "process_seed": 0,
    }
    intent_path = output_folder / "export.intent.json"
    done_path = output_folder / "export.done"
    if done_path.is_file():
        existing = json.loads(intent_path.read_text(encoding="utf-8"))
        if existing != intent:
            raise ValueError("Completed export intent differs from this invocation")
        validate_export(
            paths["python"], output_folder, paths["manifest"], paths["checkpoint"]
        )
        print(f"PASS: existing complete export revalidated: {output_folder}")
        return
    if output_folder.exists() and any(output_folder.iterdir()):
        raise RuntimeError(
            "Sequential two-shard RNG exports cannot safely resume from a partial "
            f"directory; use a new empty --output_folder: {output_folder}"
        )
    output_folder.mkdir(parents=True, exist_ok=True)
    atomic_write_json(intent_path, intent)
    require_idle_gpus(gpus)

    children: list[tuple[subprocess.Popen, object]] = []
    try:
        for shard_index, gpu in enumerate(gpus):
            log_path = output_folder / f"export.shard_{shard_index:02d}_of_02.log"
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
                "2",
                "--naming",
                "raw_prompt_index",
                "--fps",
                "16",
                "--rng_protocol",
                RNG_PROTOCOL,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_stream = open(log_path, "x", encoding="utf-8")
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
                f"Started historical-protocol shard {shard_index}/2 on GPU {gpu}: "
                f"pid={process.pid}, log={log_path}",
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
        paths["python"], output_folder, paths["manifest"], paths["checkpoint"]
    )
    print(f"PASS: raw/no-EMA historical-seed export complete: {output_folder}")


if __name__ == "__main__":
    main()
