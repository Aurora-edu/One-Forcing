#!/usr/bin/env python3
"""Official Self Forcing/Causal-Forcing + first-block FFE + fixed Qwen VBench."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from official_baseline_common import (
    HERE, LOCK, LOCK_PATH, check_completed_video, check_source, immutable_json,
    json_write, selected_records, sha256, validate_output_name,
)
from paper_vbench_protocol import ASSETS, audit_inputs, prepare, protocol_paths
from consolidate_results import load_vbench_results, official_totals

REPO = HERE.parents[1]


def models(args):
    return list(LOCK) if args.model == "both" else [args.model]


def locations(root, model):
    return {
        "source": root / "source" / model,
        "checkpoint": root / "weights" / model / LOCK[model]["hf_filename"],
        "runtime": root / "runtime" / model,
    }


def prepare_sources(args):
    prepare(args.work_root / "inputs", 5)
    for model in models(args):
        spec = LOCK[model]
        paths = locations(args.work_root, model)
        if not paths["source"].exists():
            paths["source"].parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", spec["repository"], str(paths["source"])], check=True)
            subprocess.run(["git", "-C", str(paths["source"]), "checkout", "--detach", spec["commit"]], check=True)
        check_source(paths["source"], model)
        if args.wan_path:
            wan_path = args.wan_path.resolve(strict=True)
            for relative in ["config.json", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl"]:
                if not (wan_path / relative).exists():
                    raise FileNotFoundError(wan_path / relative)
            link = paths["runtime"] / "wan_models/Wan2.1-T2V-1.3B"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                if link.resolve() != wan_path:
                    raise ValueError(f"Existing Wan asset link points elsewhere: {link}")
            else:
                link.symlink_to(wan_path, target_is_directory=True)
        if args.download_weights:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=spec["hf_repository"], filename=spec["hf_filename"],
                            revision=spec["hf_revision"], local_dir=args.work_root / "weights" / model)
        if paths["checkpoint"].exists():
            verify_checkpoint(paths["checkpoint"], model)
        print(f"Prepared {model}: {paths['source']}; checkpoint: {paths['checkpoint']}")
    json_write(args.work_root / "official_sources.json", {model: LOCK[model] for model in models(args)})


def verify_checkpoint(path, model):
    if sha256(path) != LOCK[model]["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint is not the pinned official release: {path}")


def gpu_ids(value):
    ids = value.split(",")
    if not ids or any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("--gpus requires unique comma-separated GPU indices")
    return ids


def require_idle(ids):
    for gpu in ids:
        output = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if output:
            raise RuntimeError(f"GPU {gpu} is busy ({output}); choose idle GPUs without stopping other tasks")


def output_root(args, model):
    return args.work_root / "results" / model / args.schedule / args.phase


def build_intent(args, model):
    paths = locations(args.work_root, model)
    check_source(paths["source"], model)
    protocol = audit_inputs(args.work_root / "inputs", 5)
    manifest = protocol_paths(args.work_root / "inputs", 5)["manifest"]
    rows = selected_records(manifest, args.phase)
    for record in rows:
        validate_output_name(record["output_name"])
    code = [Path(__file__).resolve(), HERE / "official_baseline_common.py", HERE / "official_baseline_worker.py", LOCK_PATH]
    return {
        "schema_version": 1, "model": model, "source_root": str(paths["source"]),
        "runtime_root": str(paths["runtime"]), "checkpoint": str(paths["checkpoint"]),
        "official_release": LOCK[model], "weight_key": LOCK[model]["weight_key"],
        "schedule": args.schedule, "phase": args.phase,
        "protocol": protocol, "records": rows,
        "latent_frames": 21, "pixel_frames": 81, "fps": 16, "resolution": [832, 480],
        "frames_per_block": 3, "first_block_steps": [1000, 750, 500, 250],
        "later_block_steps": [1000, 750, 500, 250] if args.schedule == "all4" else [1000],
        "evaluation_code_sha256": {str(p): sha256(p) for p in code},
        "native_checkpoint_training_schedule": "four-step; this is a sampling evaluation, no retraining",
    }


def audit_export(root):
    intent_path = root / "intent.json"
    intent = json.loads(intent_path.read_text())
    expected = {r["output_name"] for r in intent["records"]}
    actual = {p.name for p in (root / "videos").glob("*.mp4")}
    if expected != actual:
        raise ValueError(f"Incomplete/unexpected export: missing={len(expected-actual)}, extra={len(actual-expected)}")
    intent_hash = sha256(intent_path)
    for row in intent["records"]:
        if not check_completed_video(root / "videos" / row["output_name"], root / "metadata" / (row["output_name"] + ".json"), row, intent_hash):
            raise ValueError(f"Missing video: {row['output_name']}")
    return intent


def infer(args):
    ids = gpu_ids(args.gpus)
    for model in models(args):
        intent = build_intent(args, model)
        if args.dry_run:
            print(json.dumps({k: v for k, v in intent.items() if k != "records"}, indent=2))
            print(f"{model}: {len(intent['records'])} videos; {len(ids)} shards; no GPU launched")
            continue
        paths = locations(args.work_root, model)
        verify_checkpoint(paths["checkpoint"], model)
        if not (paths["runtime"] / "wan_models/Wan2.1-T2V-1.3B").is_dir():
            raise FileNotFoundError("Run prepare --wan-path /actual/Wan2.1-T2V-1.3B first")
        require_idle(ids)
        root = output_root(args, model)
        for directory in ["videos", "metadata", "audit", "logs", "previews"]:
            (root / directory).mkdir(parents=True, exist_ok=True)
        # Only one controller may write this experiment; changing GPU count on resume is safe.
        with (root / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            immutable_json(root / "intent.json", intent)
            children = []
            try:
                for rank, gpu in enumerate(ids):
                    log = (root / "logs" / f"shard{rank}.log").open("a")
                    command = [args.python, str(HERE / "official_baseline_worker.py"),
                               "--intent", str(root / "intent.json"), "--shard", str(rank), "--shards", str(len(ids))]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    children.append((process, log))
                while any(p.poll() is None for p, _ in children):
                    failures = [p.returncode for p, _ in children if p.poll() not in (None, 0)]
                    if failures:
                        raise RuntimeError(f"Official inference worker failed: {failures}; inspect {root/'logs'}")
                    time.sleep(1)
                if any(p.returncode != 0 for p, _ in children):
                    raise RuntimeError(f"Official inference worker failed; inspect {root/'logs'}")
            finally:
                # These Popen objects are the only processes owned by this invocation.
                for process, _ in children:
                    if process.poll() is None:
                        process.terminate()
                for process, log in children:
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    log.close()
            audit_export(root)
            json_write(root / "export.done.json", {"intent_sha256": sha256(root / "intent.json"), "videos": len(intent["records"])})
            print(f"PASS {model}: {len(intent['records'])} videos, previews at {root/'previews'}", flush=True)


def score(args):
    if args.phase != "full" or args.schedule != "ffe":
        raise ValueError("The main VBench comparison requires full 944x5 FFE export")
    ids = gpu_ids(args.gpus)
    for model in models(args):
        root = output_root(args, model)
        intent = audit_export(root)
        if len(intent["records"]) != 4720:
            raise ValueError("Expected exactly 944x5 videos")
        require_idle(ids)
        scorer = REPO / "scripts/run_vbench.py"
        command = [args.vbench_python, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={len(ids)}", str(scorer), "--videos_path", str(root / "videos"),
                   "--full_info_path", str(ASSETS / "VBench_full_info.json"),
                   "--output_dir", str(root / "vbench"), "--name", model,
                   "--device", "cuda", "--samples_per_prompt", "5"]
        provenance = {"command": command, "scorer_sha256": sha256(scorer),
                      "export_intent_sha256": sha256(root / "intent.json")}
        wrapper = Path(args.vbench_python)
        if wrapper.is_file():
            provenance["vbench_python_or_wrapper_sha256"] = sha256(wrapper)
        json_write(root / "score_provenance.json", provenance)
        subprocess.run(command, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(ids)), cwd=REPO)
    summarize(args)


def summarize(args):
    rows = []
    for model in models(args):
        root = args.work_root / "results" / model / "ffe/full"
        audit_export(root)
        result = root / "vbench" / f"{model}_eval_results.json"
        scores = load_vbench_results(result)
        totals = official_totals(scores)
        if totals is None or len(scores) != 16:
            raise ValueError(f"Expected all 16 VBench dimensions: {result}")
        rows.append({"model": model, "weight_key": LOCK[model]["weight_key"],
                     "total": 100 * totals["total_score"], "quality": 100 * totals["quality_score"],
                     "semantic": 100 * totals["semantic_score"], "prompts": "Qwen 944x5",
                     "sampling": "first 3-latent block 4 steps; subsequent blocks 1 step",
                     "checkpoint_sha256": LOCK[model]["checkpoint_sha256"], "result_path": str(result)})
    json_write(args.work_root / "summary.json", rows)
    with (args.work_root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "infer", "score", "summarize"])
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model", choices=["both", *LOCK], default="both")
    parser.add_argument("--wan-path", type=Path)
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--schedule", choices=["ffe", "all4"], default="ffe")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vbench-python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.work_root = args.work_root.resolve()
    if args.action == "score" and not args.vbench_python:
        parser.error("score requires --vbench-python (existing B200 VBench wrapper is supported)")
    {"prepare": prepare_sources, "infer": infer, "score": score, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()
