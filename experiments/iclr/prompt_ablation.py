"""Evaluate an existing checkpoint on original/pinned-Qwen prompts; no training.

Run after the nPEg experiments. The default is a read-only command plan.
This is a new paired evaluation, not a replication of the historical 83.76.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from experiments.iclr.launch import parse_gpus, require_idle
from experiments.iclr.protocol import REPO, atomic_json, sha256_file
from experiments.iclr.summarize import write_csv
from experiments.rebuttal.consolidate_results import load_vbench_results, official_totals
from experiments.rebuttal.paper_vbench_protocol import ASSETS, expected_payloads, pinned_inputs
from utils.config import load_config


def input_payloads(samples):
    prompts, rewrites = pinned_inputs()
    qwen = expected_payloads(prompts, rewrites, samples)
    original = []
    for line in qwen["manifest"].splitlines():
        record = json.loads(line)
        record.pop("extended_prompt")
        record.pop("rewrite_file_sha256")
        original.append(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"prompts.txt": qwen["prompts"], "rewrites.txt": qwen["rewrites"],
            "original.jsonl": "".join(original), "qwen.jsonl": qwen["manifest"]}


def prepare_inputs(root, samples):
    directory = Path(root) / "manifests"
    payloads = input_payloads(samples)
    # Validate every existing input before writing any missing file.
    for name, content in payloads.items():
        path = directory / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f"different prompt protocol already exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = directory / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)


def conditions(args):
    return ("original", "qwen") if args.conditioning == "paired" else ("original",)


def command_for_condition(args, condition):
    root = Path(args.output).resolve()
    inputs = root / "manifests"
    command = ["bash", str(REPO / "experiments/rebuttal/run_vbench_condition.sh"),
        "--name", condition, "--config_path", str(Path(args.config).resolve()),
        "--checkpoint_path", str(Path(args.checkpoint).resolve()),
        "--method", args.method, "--schedule", args.schedule,
        "--prompt_path", str(inputs / "prompts.txt"),
        "--manifest_path", str(inputs / f"{condition}.jsonl"),
        "--full_info_path", str(ASSETS / "VBench_full_info.json"),
        "--output_root", str(root / condition), "--gpus", args.gpus,
        "--python", args.python, "--vbench_python", args.vbench_python,
        "--samples_per_prompt", str(args.samples_per_prompt)]
    if condition == "qwen":
        command += ["--extended_prompt_path", str(inputs / "rewrites.txt")]
    command += ["--use_ema" if args.weight_source == "generator_ema" else "--require_no_ema"]
    return command


def expected_intent(args, condition):
    inputs = Path(args.output).resolve() / "manifests"
    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config).resolve()
    resolved = OmegaConf.to_yaml(load_config(str(config)), resolve=True, sort_keys=True)
    return {"checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "config_path": str(config), "config_sha256": sha256_file(config),
        "resolved_config_sha256": hashlib.sha256(resolved.encode()).hexdigest(),
        "prompt_sha256": sha256_file(inputs / "prompts.txt"),
        "extended_prompt_sha256": sha256_file(inputs / "rewrites.txt") if condition == "qwen" else None,
        "manifest_sha256": sha256_file(inputs / f"{condition}.jsonl"),
        "selected_manifest_records": 944 * args.samples_per_prompt,
        "method": args.method, "schedule": args.schedule, "num_output_frames": 21,
        "fps": 16, "use_ema": args.weight_source == "generator_ema"}


def summarize(args):
    root = Path(args.output).resolve()
    # Recheck immutable inputs, including pairing of prompt/sample/noise seeds.
    for name, content in input_payloads(args.samples_per_prompt).items():
        if (root / "manifests" / name).read_text() != content:
            raise ValueError(f"prompt input changed: {name}")
    rows = []
    provenance = {}
    for condition in conditions(args):
        directory = root / condition
        intent = json.loads((directory / "videos/export.intent.json").read_text())
        expected = expected_intent(args, condition)
        if any(intent.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{condition}: export intent differs from paired protocol")
        done = json.loads((directory / "videos/export.done").read_text())
        required = {"status": "complete", "num_videos": 944 * args.samples_per_prompt,
            "weight_source": args.weight_source, "use_ema": expected["use_ema"],
            "manifest_sha256": expected["manifest_sha256"],
            "checkpoint_path": expected["checkpoint_path"],
            "checkpoint_size_bytes": expected["checkpoint_size_bytes"],
            "latent_frames_per_video": 21, "rgb_frames_per_video": 81, "fps": 16}
        if any(done.get(k) != v for k, v in required.items()):
            raise ValueError(f"{condition}: incomplete or mismatched export")
        vbench = directory / "vbench"
        scoring = json.loads((vbench / f"{condition}_protocol.json").read_text())
        scores = load_vbench_results(vbench / f"{condition}_eval_results.json")
        totals = official_totals(scores)
        if (scoring.get("mode") != "vbench_standard"
                or scoring.get("samples_per_prompt") != args.samples_per_prompt
                or scoring.get("official_five_sample_protocol") is not (args.samples_per_prompt == 5)
                or len(scoring.get("dimensions", [])) != 16
                or set(scoring["dimensions"]) != set(scores) or totals is None):
            raise ValueError(f"{condition}: mismatched scoring protocol or missing dimensions")
        scores.update(totals)
        if not all(math.isfinite(score) for score in scores.values()):
            raise ValueError("non-finite VBench score")
        provenance[condition] = expected
        rows.append({"conditioning": condition, **scores})
    deltas = []
    if len(rows) == 2:
        deltas = [{"metric": k, "qwen_minus_original": rows[1][k] - value}
                  for k, value in rows[0].items() if k != "conditioning"]
    return {"scores": rows, "gains": deltas, "provenance": provenance,
        "samples_per_prompt": args.samples_per_prompt, "weight_source": args.weight_source,
        "full_info_sha256": sha256_file(ASSETS / "VBench_full_info.json"),
        "interpretation": "same checkpoint and schedule; conditioning changes only; not training-seed variance",
        "historical_headline_replication": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(REPO / "experiments/iclr/configs/main_reference.yaml"))
    parser.add_argument("--method", choices=("framewise", "chunkwise"), default="framewise")
    parser.add_argument("--schedule", choices=("all1", "ffe", "all4", "main_ffe"), default="main_ffe")
    parser.add_argument("--weight-source", choices=("generator", "generator_ema"), default="generator_ema")
    parser.add_argument("--conditioning", choices=("original", "paired"), default="paired")
    parser.add_argument("--samples-per-prompt", type=int, choices=(1, 5), default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vbench-python", default="VBENCH_PYTHON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.method == "chunkwise" and args.schedule == "ffe":
        parser.error("this exporter supports FFE only for framewise models")
    for condition in conditions(args):
        print(shlex.join(command_for_condition(args, condition)))
    if not (args.execute or args.prepare_only or args.summarize_only):
        print("DRY RUN: evaluation only, no training/GPU access; run after nPEg priorities")
        return
    if args.execute and not os.environ.get("TMUX"):
        raise RuntimeError("long evaluations require an independent tmux session")
    prepare_inputs(args.output, args.samples_per_prompt)
    if args.prepare_only:
        return
    if args.execute:
        gpus = parse_gpus(args.gpus)
        for condition in conditions(args):
            require_idle(gpus)
            subprocess.run(command_for_condition(args, condition), cwd=REPO, check=True)
    result = summarize(args)
    root = Path(args.output)
    atomic_json(root / "prompt_comparison.json", result)
    write_csv(root / "prompt_scores.csv", result["scores"])
    if result["gains"]:
        write_csv(root / "prompt_gains.csv", result["gains"])
    print(f"Wrote {root}; inspect actual MP4 motion before using the results")


if __name__ == "__main__":
    main()
