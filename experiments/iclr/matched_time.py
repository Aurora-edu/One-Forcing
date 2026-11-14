"""Reuse historical seed0 frame200; train/evaluate only equal-time chunkwise.

The existing frame200 clock fixes the budget at 4261.134528471157 seconds.
Reuse its scores; only chunk gets a new EMA/Qwen944x5/main_ffe endpoint export.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from omegaconf import OmegaConf

from experiments.iclr.launch import parse_gpus, require_idle, start_tmux, train_command
from experiments.iclr.make_configs import make_matrix
from experiments.iclr.protocol import (
    REPO, LEGACY_PROFILE, atomic_json, config_dict, resume_signature, sha256_file, validate_config,
)
from experiments.iclr import prompt_ablation as evaluation
from experiments.iclr.state import TrainingTimeClock
from utils.config import load_config

PROFILE = "matched_time_seed0_v1"
HISTORICAL_COMMIT = "1ee818ee8535052f43868c9eecc8e4d6c4732406"
INIT_SHA256 = "58bd2f6c3b757d6582541c4354eff2626ea0ba4cd1fcf48c729a3d24518379bf"
CAP = 1000
ARMS = ("frame", "chunk")
ACTIVE_ARMS = ("chunk",)
SCOPE = "historical_frame200_chunk_only"
FRAME_ANCHOR = REPO / "experiments/iclr/assets/historical_frame200_seed0.json"
PROTECTED_DIRS = ("model", "pipeline", "trainer", "utils", "wan")
CONTROL_FILES = (
    "experiments/iclr/matched_time.py", "experiments/iclr/trainer.py",
    "experiments/iclr/state.py", "experiments/iclr/protocol.py", "experiments/iclr/train.py",
    "experiments/iclr/configs/base.yaml", "experiments/iclr/configs/main_reference.yaml",
    "experiments/iclr/prompt_ablation.py", "experiments/rebuttal/run_sharded_inference.py",
    "experiments/rebuttal/run_vbench_condition.sh", "scripts/infer.sh", "inference.py",
    "experiments/iclr/assets/historical_frame200_seed0.json",
)
HISTORICAL = {
    "source_commit": HISTORICAL_COMMIT,
    "frame": {"step": 200, "training_seconds": 4261.134528471157, "total_percent": 83.4519},
    "chunk": {"step": 200, "training_seconds": 3268.672115027439, "total_percent": 81.6281},
    "note": "Reuse historical frame200 score and time. Train and evaluate only a new equal-time chunk endpoint.",
}


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def frame_anchor():
    anchor = read(FRAME_ANCHOR)
    if (anchor["source_commit"] != HISTORICAL_COMMIT or anchor["seed"] != 0
            or anchor["training"]["step"] != 200
            or anchor["training"]["training_seconds"] != HISTORICAL["frame"]["training_seconds"]
            or anchor["initialization"]["generator_checkpoint_sha256"] != INIT_SHA256):
        raise ValueError("Historical frame200 anchor changed")
    return anchor


def historical_budget():
    anchor = frame_anchor()
    return {"training_seconds": anchor["training"]["training_seconds"],
            "anchor_experiment": anchor["experiment_id"], "anchor_step": 200,
            "anchor_source": "historical_reused", "anchor_metadata_sha256": sha256_file(FRAME_ANCHOR)}


def require_chunk(arm):
    if arm not in ACTIVE_ARMS:
        raise ValueError("Only chunk is authorized; reuse historical frame200 without retraining or reevaluation")


def source_fingerprint():
    # Model, optimizer, data, EMA and inference implementations match the old run.
    changes = git("diff", "--name-only", HISTORICAL_COMMIT, "--", *PROTECTED_DIRS)
    if changes:
        raise ValueError(f"Historical model/training implementations changed: {changes}")
    return {"historical_commit": HISTORICAL_COMMIT,
            "controls": {p: sha256_file(REPO / p) for p in CONTROL_FILES}}


def validate_time_config(config):
    c = config_dict(config)
    block = c.get("num_frame_per_block")
    if c.get("seed") != 0 or c.get("iclr_mode") != "formal" or block not in (1, 3):
        raise ValueError("matched time requires formal seed0 frame/chunk runs")
    if block == 1:
        if c.get("iclr_stop_condition") != "fixed_steps" or "iclr_budget_seconds" in c:
            raise ValueError("frame is the fixed-200 time anchor")
    else:
        if c.get("iclr_stop_condition") != "training_seconds" or c.get("max_steps") != CAP:
            raise ValueError("chunk must use measured training_seconds, with a 1000-step safety cap")
        if c.get("iclr_save_steps") != list(range(0, CAP + 1, 50)):
            raise ValueError("chunk checkpoints must be every 50 steps plus the terminal iteration")
        TrainingTimeClock(CAP, tuple(c["iclr_save_steps"]), budget_seconds=c.get("iclr_budget_seconds", 0))
        c.pop("iclr_budget_seconds")
        c.update(max_steps=200, iclr_stop_condition="fixed_steps", iclr_save_steps=[0, 50, 100, 150, 200])
    # Reuse the strict historical scientific-settings validator. Only the time
    # budget fields are relaxed, never lr, losses, EMA, windows, offload or seed.
    c["iclr_profile"] = LEGACY_PROFILE
    validate_config(OmegaConf.create(c))


def config_for(assets, arm, budget=None):
    if arm not in ARMS:
        raise ValueError(arm)
    c = make_matrix(load_config(str(REPO / "experiments/iclr/configs/base.yaml")),
                    generator_ckpt=assets["generator"], teacher=assets["teacher"],
                    data=assets["data"], prompt_cache="")[ARMS.index(arm)]
    c.iclr_profile = PROFILE
    if arm == "chunk":
        c.update(iclr_stop_condition="training_seconds", iclr_budget_seconds=budget,
                 max_steps=CAP, iclr_save_steps=list(range(0, CAP + 1, 50)))
    validate_config(c)
    return c


def immutable_json(path, payload):
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Immutable task input changed: {path}")
    else:
        atomic_json(path, payload)


def prepare(output, args):
    if output.exists():
        raise FileExistsError(f"Use a fresh directory; preserve earlier results: {output}")
    source = source_fingerprint()
    assets = {key: str(Path(getattr(args, key)).resolve()) for key in ("generator", "teacher", "data")}
    budget = historical_budget()
    config = config_for(assets, "chunk", budget["training_seconds"])
    output.mkdir(parents=True)
    OmegaConf.save(config, output / "chunk.yaml")
    immutable_json(output / "budget.json", budget)
    immutable_json(output / "historical_frame200.json", frame_anchor())
    evaluation.prepare_inputs(output / "inputs", 5)
    manifests = output / "inputs/manifests"
    records = [json.loads(line) for line in (manifests / "qwen.jsonl").read_text().splitlines()]
    chosen = [r for r in records if r["sample_index"] == 0 and r["prompt_index"] in (0, 30, 100, 200, 300, 500, 700, 900)]
    (manifests / "preview.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen))
    atomic_json(output / "task.json", {"profile": PROFILE, "scope": SCOPE, "active_arms": list(ACTIVE_ARMS), "assets": assets,
        "source": source, "historical_reference": HISTORICAL,
        "inputs": {p.name: sha256_file(p) for p in manifests.iterdir()},
        "time_definition": "sum of CUDA-synchronized per-iteration durations, MAX across 8 ranks; excludes checkpoint I/O, logging/GC and initialization",
        "budget_rule": "reuse historical frame200 training_seconds=4261.134528471157; no frame training/evaluation"})


def check(output):
    task = read(output / "task.json")
    if (task["profile"] != PROFILE or task.get("scope") != SCOPE
            or task.get("active_arms") != list(ACTIVE_ARMS) or task["source"] != source_fingerprint()):
        raise ValueError("Task source or identity changed; use a fresh task for a changed recipe")
    if read(output / "historical_frame200.json") != frame_anchor():
        raise ValueError("Historical frame200 evidence changed")
    if read(output / "budget.json") != historical_budget():
        raise ValueError("Historical frame200 time budget changed")
    manifests = output / "inputs/manifests"
    if any(sha256_file(manifests / p) != h for p, h in task["inputs"].items()):
        raise ValueError("Pinned evaluation inputs changed")
    budget = historical_budget()["training_seconds"]
    if config_dict(load_config(str(output / "chunk.yaml"))) != config_dict(config_for(task["assets"], "chunk", budget)):
        raise ValueError("chunk time-budget config changed")
    return task


def attempts(output, arm):
    return sorted((output / "runs" / arm).glob("attempt_[0-9][0-9][0-9]"))


def endpoint(output, arm):
    require_chunk(arm)
    runs = attempts(output, arm)
    if not runs or not (runs[-1] / "training.done").is_file():
        raise FileNotFoundError(f"{arm}: terminal training.done is not available")
    run = runs[-1]
    done = read(run / "training.done")
    if done.get("formal_result") is not True:
        raise ValueError(f"{arm}: time budget was not reached (safety cap is not a result)")
    config = load_config(str(output / f"{arm}.yaml"))
    if resume_signature(load_config(str(run / "resolved_config.yaml"))) != resume_signature(config):
        raise ValueError(f"{arm}: actual training config changed")
    step = done["step"]
    if arm == "frame" and step != 200:
        raise ValueError("frame anchor must finish exactly 200 steps")
    checkpoint = run / f"checkpoint_model_{step:06d}/model.pt"
    complete = read(checkpoint.parent / "complete.json")
    if (complete["config"] != resume_signature(config)
            or complete["clock"]["step"] != step
            or complete["clock"]["training_seconds"] != done["training_seconds"]
            or checkpoint.stat().st_size != complete["files"]["model.pt"]):
        raise ValueError(f"{arm}: checkpoint does not match terminal clock/config")
    rows = [json.loads(x) for x in (run / "training_clock.jsonl").read_text().splitlines() if x.strip()]
    # The first crossing must be the endpoint. Supports resume at step N.
    if arm == "chunk":
        budget = read(output / "budget.json")["training_seconds"]
        previous = rows[-2]["training_seconds"] if len(rows) > 1 else read(run / "launch.json")["resume_training_seconds"]
        if (done.get("time_budget_reached") is not True or done.get("budget_seconds") != budget
                or not previous < budget <= done["training_seconds"]):
            raise ValueError("chunk endpoint is not the first complete iteration reaching the budget")
    return run, checkpoint, done


def freeze_budget(output):
    check(output)
    return read(output / "budget.json")


def hardware(gpus):
    import torch
    return {"hostname": socket.gethostname(), "gpus": gpus, "world_size": 8,
            "python": sys.version, "python_executable": sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "devices": subprocess.check_output(["nvidia-smi", "-i", gpus,
                "--query-gpu=uuid,name,memory.total,power.limit", "--format=csv,noheader,nounits"], text=True).splitlines()}


def asset_identity(assets):
    # Same physical files throughout this paired run; the large LMDB is not
    # reread merely to hash it. Initialization itself has a historical SHA256.
    paths = [Path(assets["generator"]), Path(assets["data"]) / "data.mdb"]
    for root in (Path(assets["teacher"]), REPO / "wan_models/Wan2.1-T2V-1.3B"):
        if not root.is_dir():
            raise FileNotFoundError(f"Historical model assets unavailable: {root}")
        paths.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in (".pth", ".bin", ".safetensors", ".json"))
    return {str(p.resolve()): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(set(paths))}


def next_attempt(output, arm):
    runs = attempts(output, arm)
    resume = ""
    if runs:
        candidates = [(read(p)["clock"]["step"], p.parent) for run in runs
                      for p in run.glob("checkpoint_model_*/complete.json")]
        if candidates:
            resume = str(max(candidates, key=lambda x: x[0])[1])
    return output / "runs" / arm / f"attempt_{len(runs):03d}", resume


def train_arm(output, arm, args):
    require_chunk(arm)
    task = check(output)
    if arm == "chunk":
        freeze_budget(output)
    if attempts(output, arm) and (attempts(output, arm)[-1] / "training.done").exists():
        endpoint(output, arm)
        print(f"{arm}: already complete; preserving existing result", flush=True)
        return
    require_idle(parse_gpus(args.gpus))
    immutable_json(output / "hardware.json", hardware(args.gpus))
    if git("diff", "--name-only", "HEAD"):
        raise ValueError("Commit the training source before execution")
    assets = task["assets"]
    for path in (Path(assets["data"]) / "data.mdb", Path(assets["teacher"]), Path(assets["generator"])):
        if not path.exists():
            raise FileNotFoundError(path)
    # The seed-0 run's raw ODE initialization is pinned, not one_forcing.pt.
    if sha256_file(assets["generator"]) != INIT_SHA256:
        raise ValueError("Generator initialization differs from the historical seed0 experiment")
    immutable_json(output / "asset_identity.json", asset_identity(assets))
    run, resume = next_attempt(output, arm)
    command = train_command(output / f"{arm}.yaml", run, args.python, resume)
    started = time.time()
    receipt = {"command": command, "git_commit": git("rev-parse", "HEAD"),
               "start_unix": started, "resume": resume,
               "resume_training_seconds": read(Path(resume) / "complete.json")["clock"]["training_seconds"] if resume else 0.}
    print(f"START {arm}; observe at least the first 10 steps. {shlex.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}, check=True)
    finally:
        # Keep metadata outside the output until train.py creates its fresh directory.
        receipt["end_unix"] = time.time()
        receipt["process_wall_seconds"] = receipt["end_unix"] - started
        if run.exists():
            atomic_json(run / "launch.json", receipt)
    endpoint(output, arm)


def monitor(output, arm):
    runs = attempts(output, arm)
    if not runs or not (runs[-1] / "metrics.jsonl").is_file():
        return {"arm": arm, "status": "not started or still initializing", "step10_observed": False}
    run = runs[-1]
    # Only consume newline-terminated records when the trainer is appending.
    lines = (run / "metrics.jsonl").read_text().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]
    if not rows:
        return {"arm": arm, "status": "no complete metric records", "step10_observed": False}
    if [r["step"] for r in rows] != list(range(rows[0]["step"], rows[-1]["step"] + 1)):
        raise ValueError(f"{arm}: missing or duplicate training iterations")
    for r in rows:
        if any(not math.isfinite(v) for v in r.values() if isinstance(v, (float, int))):
            raise ValueError(f"{arm}: nonfinite metric at step {r['step']}")
        keys = ["critic_loss", "fake_score_loss", "gan_d_loss", "critic_grad_norm"]
        if (r["step"] - 1) % 5 == 0:
            keys += ["generator_loss", "dmd_loss", "gan_g_loss", "generator_grad_norm"]
        if any(k not in r for k in keys) or any(r.get(k, 0) != 0 for k in ("r1_loss", "r2_loss")):
            raise ValueError(f"{arm}: missing expected losses or regularization enabled at step {r['step']}")
    result = {"arm": arm, "run": str(run), "last_step": rows[-1]["step"],
              "step10_observed": len(rows) >= 10, "metrics_finite": True,
              "last_metrics": rows[-1], "training_done": (run / "training.done").exists()}
    if len(rows) >= 10:
        atomic_json(run / "observed_step10.json", result)
    return result


def eval_args(output, arm, args):
    _, checkpoint, _ = endpoint(output, arm)
    return SimpleNamespace(output=str(output / "eval" / arm), config=str(output / f"{arm}.yaml"),
        checkpoint=str(checkpoint), method="framewise" if arm == "frame" else "chunkwise",
        schedule="main_ffe", weight_source="generator_ema", conditioning="qwen", samples_per_prompt=5,
        gpus=args.gpus, python=args.python, vbench_python=args.vbench_python)


def preview_command(output, arm, args):
    ea = eval_args(output, arm, args)
    m = output / "inputs/manifests"
    return [args.python, str(REPO / "experiments/rebuttal/run_sharded_inference.py"),
            "--config_path", ea.config, "--checkpoint_path", ea.checkpoint,
            "--method", ea.method, "--schedule", "main_ffe", "--use_ema",
            "--prompt_path", str(m / "prompts.txt"), "--extended_prompt_path", str(m / "rewrites.txt"),
            "--manifest_path", str(m / "preview.jsonl"), "--output_folder", str(output / "preview" / arm),
            "--gpus", args.gpus, "--python", args.python, "--num_output_frames", "21", "--fps", "16"]


def report(output, args):
    check(output)
    anchor = frame_anchor()
    clock = anchor["training"]
    rows = [{"method": "frame", "source": "historical_reused", "seed": 0, "step": 200,
             "generator_updates": clock["generator_updates"], "critic_updates": clock["critic_updates"],
             "training_minutes": clock["training_seconds"] / 60,
             "checkpoint_minutes": clock["checkpoint_seconds"] / 60,
             "overhead_minutes": clock["overhead_seconds"] / 60,
             "process_minutes_including_recovery": None, "attempts": None, "budget_overshoot_seconds": 0.,
             **{k: 100 * anchor["evaluation"]["scores"][k]
                for k in ("total_score", "quality_score", "semantic_score")}}]
    inits = [{k: anchor["initialization"][k]
              for k in ("generator_tensor_sha256", "critic_tensor_sha256", "initialization_seed")}]
    uninterrupted = True
    for arm in ACTIVE_ARMS:
        run, _, done = endpoint(output, arm)
        init = read(run / "initialization.json")
        if init["generator_checkpoint_sha256"] != INIT_SHA256:
            raise ValueError("Initialization changed")
        inits.append({k: init[k] for k in ("generator_tensor_sha256", "critic_tensor_sha256", "initialization_seed")})
        score = evaluation.summarize(eval_args(output, arm, args))["scores"][0]
        arm_attempts = attempts(output, arm)
        receipts = [read(p / "launch.json") for p in arm_attempts if (p / "launch.json").exists()]
        uninterrupted = uninterrupted and len(arm_attempts) == len(receipts) == 1 and not receipts[0]["resume"]
        process_minutes = (sum(r["process_wall_seconds"] for r in receipts) / 60
                           if len(receipts) == len(arm_attempts) else None)
        rows.append({"method": arm, "source": "new_training", "seed": 0, "step": done["step"],
                     "generator_updates": done["generator_updates"], "critic_updates": done["critic_updates"],
                     "training_minutes": done["training_seconds"] / 60,
                     "checkpoint_minutes": done["checkpoint_seconds"] / 60,
                     "overhead_minutes": done["overhead_seconds"] / 60,
                     "process_minutes_including_recovery": process_minutes,
                     "attempts": len(arm_attempts),
                     "budget_overshoot_seconds": done.get("budget_overshoot_seconds", 0.),
                     **{k: 100 * score[k] for k in ("total_score", "quality_score", "semantic_score")}})
    if inits[0] != inits[1]:
        raise ValueError("Generator/critic initialization differs between arms")
    result = {"rows": rows, "frame_minus_chunk": {k: rows[0][k] - rows[1][k]
                 for k in ("total_score", "quality_score", "semantic_score")},
              "budget": read(output / "budget.json"), "hardware": read(output / "hardware.json"),
              "historical_reference": HISTORICAL,
              "protocol": "seed0; R1/R2 off; EMA; main_ffe; pinned Qwen 944x5; endpoint only",
              "time_definition": read(output / "task.json")["time_definition"],
              "scope": SCOPE, "uninterrupted_chunk": uninterrupted,
              "recovery_note": (None if uninterrupted else
                  "Checkpoint clocks exclude discarded work after an interruption. Preserve these results and retry only chunk in a fresh task if needed; always reuse historical frame200.")}
    atomic_json(output / "matched_time_results.json", result)
    evaluation.write_csv(output / "matched_time_results.csv", rows)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "check", "train", "monitor", "preview", "evaluate", "report"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ACTIVE_ARMS, default="chunk")
    for name in ("generator", "teacher", "data"):
        p.add_argument(f"--{name}")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--vbench-python", default="VBENCH_PYTHON")
    p.add_argument("--session", default="iclr_seed0_matched_time")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = output = args.output.resolve()
    if args.action == "prepare":
        if any(not getattr(args, key) for key in ("generator", "teacher", "data")):
            p.error("prepare requires --generator, --teacher and --data from the historical seed0 run")
        prepare(output, args)
        print(f"Prepared {output}. Chunk only, historical frame200 budget = {historical_budget()['training_seconds']} seconds.")
        return
    check(output)
    if args.action == "check":
        print("PASS: historical frame200 reused; only chunk trains for 4261.134528471157s; seed0/EMA/Qwen944x5/main_ffe")
        return
    if args.action == "report":
        report(output, args)
        return
    selected = (args.arm,)
    if args.action == "monitor":
        print(json.dumps([monitor(output, arm) for arm in selected], indent=2))
        return
    parse_gpus(args.gpus)
    if args.action == "train":
        if not args.execute and not args.worker:
            print("DRY RUN: no GPU queries or jobs launched. Reuse historical frame200 score 83.4519.")
            print("Train ONLY chunk from historical initialization until training_seconds >= 4261.134528471157.")
            print("Budget: 71.0189 minutes. Do not retrain/reevaluate frame or hardcode ~261 steps.")
            return
        if not args.worker:
            require_idle(parse_gpus(args.gpus))
            argv = [args.python, "-m", "experiments.iclr.matched_time", "train", "--output", str(output),
                    "--arm", args.arm, "--gpus", args.gpus, "--python", args.python, "--worker"]
            start_tmux(args.session, argv, output / "logs" / f"{args.session}.log")
            return
        if not os.environ.get("TMUX"):
            raise RuntimeError("Training requires independent tmux")
        for arm in selected:
            train_arm(output, arm, args)
        return
    commands = []
    for arm in selected:
        if args.action == "preview":
            command = preview_command(output, arm, args)
        else:
            ea = eval_args(output, arm, args)
            if args.execute:
                evaluation.prepare_inputs(ea.output, 5)
            command = evaluation.command_for_condition(ea, "qwen")
        print(shlex.join(command), flush=True)
        commands.append(command)
    if args.execute:
        if not os.environ.get("TMUX"):
            raise RuntimeError("Long exports/evaluations require independent tmux")
        for command in commands:
            require_idle(parse_gpus(args.gpus))
            subprocess.run(command, cwd=REPO, check=True)
        print("Inspect the actual MP4s, including motion. Keep both successful and poor samples.")


if __name__ == "__main__":
    main()
