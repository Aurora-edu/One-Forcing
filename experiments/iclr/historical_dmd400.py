"""Reproduce the historical DMD-only recipe for 400 continuous iterations.

The 83.93 GAN reference used a legacy four-frame first block. Keep that
sampler and the historical trainer in separate pinned, unmodified worktrees.
This entry never resumes weights and never trains a new GAN arm.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from utils.config import load_config
from experiments.rebuttal import paper_vbench_protocol as protocol
from experiments.rebuttal.summarize_paper_aligned_vbench import summarize

REPO = Path(__file__).resolve().parents[2]
TRAIN_COMMIT = "d0abec6518e7d7dc567da9eb3abbefadab0f89b2"
EVAL_COMMIT = "54841ba66cda33b77ba018d78f47ee2d052cc224"
RUN_NAME = "historical_dmd_only_400"
REFERENCE = {
    "total_score": 83.93, "quality_score": 85.41, "semantic_score": 78.04,
    "checkpoint_md5": "3bb7732b43167b0ad5c2fd0ee4f9f957",
    "checkpoint_suffix": "train_1step_one_forcing_600/seed_0/checkpoint_model_000400/model.pt",
    "report": "experiments/rebuttal/results/ICLR2027_APPENDIX_GAN_ABLATION.md",
    "precision": "published rounded percentages, not full-precision aggregates",
}
TRAIN_FLAGS = ["--fake_score_cpu_offload", "--manual_generator_backward",
               "--generator_optimizer_state_cpu_offload", "--rank0_preload_generator_ckpt"]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_config(path):
    return OmegaConf.to_container(load_config(str(path)), resolve=True)


def expected_config(train_root):
    cfg = read_config(train_root / "experiments/rebuttal/configs/train_1step_dmd_only.yaml")
    cfg["max_steps"] = 400
    return cfg


def verify_source(root, commit):
    if git(root, "rev-parse", "HEAD") != commit or git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Historical source must be clean and pinned at {commit}: {root}")


def prepare(output):
    if output.exists():
        raise FileExistsError(f"Use a new task directory; existing results are preserved: {output}")
    # Resolve both objects before creating anything. Fetch the historical review
    # and iclr2027-rebuttal-gan-ablation-audit branches if a clone is shallow.
    for commit in (TRAIN_COMMIT, EVAL_COMMIT):
        git(REPO, "cat-file", "-e", f"{commit}^{{commit}}")
    output.mkdir(parents=True)
    for name, commit in (("training", TRAIN_COMMIT), ("evaluation", EVAL_COMMIT)):
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach",
                        str(output / name), commit], check=True)
    OmegaConf.save(OmegaConf.create(expected_config(output / "training")), output / "dmd400.yaml")
    (output / "reference_report.md").write_text(
        (output / "evaluation" / REFERENCE["report"]).read_text())
    protocol.prepare(output / "inputs", 5)
    # Fixed, stratified manifest positions; these previews never enter VBench.
    paths = protocol.protocol_paths(output / "inputs", 5)
    records = [json.loads(line) for line in paths["manifest"].read_text().splitlines()]
    selected = [r for r in records if r["sample_index"] == 0 and
                r["prompt_index"] in (0, 30, 100, 200, 300, 500, 700, 900)]
    (output / "inputs/preview.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    write_json(output / "task.json", {
        "training_commit": TRAIN_COMMIT, "evaluation_commit": EVAL_COMMIT,
        "seed": 0, "iterations": 400, "generator_updates": 80,
        "reference": REFERENCE, "training_config_sha256": protocol.sha256_file(output / "dmd400.yaml"),
        "protocol": protocol.audit_inputs(output / "inputs", 5),
    })
    check(output)


def check(output):
    task = json.loads((output / "task.json").read_text())
    for name, commit in (("training", TRAIN_COMMIT), ("evaluation", EVAL_COMMIT)):
        verify_source(output / name, commit)
    cfg = read_config(output / "dmd400.yaml")
    if cfg != expected_config(output / "training"):
        raise ValueError("DMD400 config differs from the historical recipe plus max_steps=400")
    if task.get("reference") != REFERENCE or task.get("seed") != 0 or task.get("iterations") != 400:
        raise ValueError("Task identity changed")
    if protocol.sha256_file(output / "dmd400.yaml") != task["training_config_sha256"]:
        raise ValueError("Training config checksum changed")
    if protocol.audit_inputs(output / "inputs", 5) != task["protocol"]:
        raise ValueError("Evaluation protocol changed")
    ffe = read_config(output / "evaluation/experiments/rebuttal/configs/eval_ffe.yaml")
    if (ffe["rollout_schedule"], ffe["first_rollout_num_frames"],
        ffe["first_frame_denoising_step_list"], ffe["denoising_step_list"]) != (
            "first4then1", 4, [1000, 750, 500, 250], [1000]):
        raise ValueError("The historical four-frame FFE sampler changed")
    return task


def run_dir(output):
    return output / "runs" / RUN_NAME / "seed_0"


def clean_env():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    return env


def eight_gpus(value):
    ids = value.split(",")
    if len(ids) != 8 or len(set(ids)) != 8 or any(not x.isdigit() for x in ids):
        raise ValueError("Use exactly eight distinct GPU IDs, matching the historical global batch")
    return value


def require_idle(value):
    for gpu in eight_gpus(value).split(","):
        pids = subprocess.check_output([
            "nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], text=True).strip()
        if pids:
            raise RuntimeError(f"GPU {gpu} is occupied; wait for an idle allocation, never kill other sessions")


def train_command(output, args):
    return ["bash", str(output / "training/experiments/rebuttal/launch_train.sh"),
            "--config_path", str(output / "dmd400.yaml"), "--run_name", RUN_NAME,
            "--seed", "0", "--max_steps", "400", "--gpus", eight_gpus(args.gpus),
            "--generator_ckpt", str(Path(args.generator).resolve()),
            "--teacher_model_path", str(Path(args.teacher).resolve()),
            "--data_path", str(Path(args.data).resolve()),
            "--prompt_embedding_cache_path", str(Path(args.prompt_cache).resolve()),
            "--output_root", str(output / "runs"), "--session", args.session,
            "--python", args.python, *TRAIN_FLAGS]


def link_assets(output, wan_root):
    target = Path(wan_root).resolve()
    if not (target / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth").is_file():
        raise FileNotFoundError(f"--wan-root must contain Wan2.1-T2V-1.3B assets: {target}")
    for name in ("training", "evaluation"):
        link = output / name / "wan_models"
        if link.exists() or link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"Existing assets point elsewhere: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)


def audit_training(output, *, final):
    check(output)
    run = run_dir(output)
    cfg = read_config(run / "resolved_config.yaml")
    expected = read_config(output / "dmd400.yaml")
    launch = json.loads((output / "launch.json").read_text())
    command = launch["command"]
    for key, flag in (("generator_ckpt", "--generator_ckpt"), ("teacher_model_path", "--teacher_model_path"),
                      ("data_path", "--data_path"), ("prompt_embedding_cache_path", "--prompt_embedding_cache_path")):
        expected[key] = command[command.index(flag) + 1]
    expected.update(fake_score_cpu_offload=True, manual_generator_backward=True,
                    generator_optimizer_state_cpu_offload=True, rank0_preload_generator_ckpt=True)
    differences = {k: [cfg.get(k), v] for k, v in expected.items() if cfg.get(k) != v}
    if differences or cfg.get("resume_ckpt") or cfg.get("generator_ckpt_after_resume"):
        raise ValueError(f"Training recipe changed or resumed: {differences}")
    if int((run / "runtime_seed.txt").read_text()) != 0:
        raise ValueError("The historical seed is literal zero, not a random seed")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    needed = 400 if final else 10
    if len(rows) < needed or len(rows) > 400 or [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Expected continuous, unique metric records from step 1")
    for row in rows:
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise ValueError(f"Nonfinite training metric at step {row['step']}")
        required = ["critic_loss", "fake_score_loss", "gan_d_loss", "r1_loss", "r2_loss"]
        generator_step = (row["step"] - 1) % 5 == 0
        if generator_step:
            required += ["generator_loss", "dmd_loss", "gan_g_loss"]
        if any(k not in row for k in required) or ("generator_loss" in row) != generator_step:
            raise ValueError(f"Missing losses or incorrect G:D update ratio at step {row['step']}")
        if any(row.get(k, 0) != 0 for k in ("gan_g_loss", "gan_d_loss", "r1_loss", "r2_loss")):
            raise ValueError("Nonzero GAN/regularization loss in the DMD-only arm")
    metadata = json.loads((run / "run_metadata.json").read_text())
    if metadata.get("git_commit") != TRAIN_COMMIT or metadata.get("git_status_porcelain"):
        raise ValueError("Actual training source was not the clean historical snapshot")
    if final:
        done = json.loads((run / "training.done").read_text())
        if done != {"final_step": 400, "max_steps": 400}:
            raise ValueError("Training did not finish all 400 iterations")
        for step in (100, 200, 300, 400):
            if not (run / f"checkpoint_model_{step:06d}/model.pt").is_file():
                raise FileNotFoundError(f"Missing checkpoint at step {step}")
        from experiments.rebuttal.audit_noema_checkpoint import audit_checkpoint
        audit_checkpoint(run / "checkpoint_model_000400/model.pt", expected_step=400)
    return {"last_step": len(rows), "critic_updates": len(rows),
            "generator_updates": sum("generator_loss" in r for r in rows),
            "continuous_from_zero": True, "seed": 0}


def eval_command(output, args, preview=False):
    root = output / "evaluation/experiments/rebuttal"
    checkpoint = run_dir(output) / "checkpoint_model_000400/model.pt"
    if preview:
        return [args.python, str(root / "run_sharded_inference.py"),
                "--config_path", str(root / "configs/eval_ffe.yaml"),
                "--checkpoint_path", str(checkpoint), "--method", "framewise", "--schedule", "ffe",
                "--prompt_path", str(output / "inputs/vbench_prompts.txt"),
                "--extended_prompt_path", str(output / "inputs/vbench_qwen_rewrites.txt"),
                "--manifest_path", str(output / "inputs/preview.jsonl"),
                "--output_folder", str(output / "preview/videos"),
                "--gpus", eight_gpus(args.gpus), "--python", args.python,
                "--num_output_frames", "21", "--fps", "16"]
    return ["bash", str(root / "run_paper_aligned_vbench.sh"), "--name", "dmd400_ffe",
            "--checkpoint_path", str(checkpoint), "--schedule", "ffe", "--model", "one_forcing",
            "--samples_per_prompt", "5", "--output_root", str(output / "eval/dmd400_ffe"),
            "--gpus", eight_gpus(args.gpus), "--python", args.python, "--vbench_python", args.vbench_python]


def start_tmux(command, cwd, session, logfile):
    if not session or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in session):
        raise ValueError("Use a simple, unique tmux session name")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise FileExistsError(f"tmux session exists: {session}")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    script = f"set -o pipefail; cd {shlex.quote(str(cwd))}; unset PYTHONPATH PYTHONHOME; "
    script += f"{shlex.join(command)} 2>&1 | tee {shlex.quote(str(logfile))}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session,
                    shlex.join(["bash", "-lc", script])], check=True, env=clean_env())


def validated_summary(path, output):
    # Recalculate all 16 dimensions and verify exact conditioning/RNG hashes.
    value = summarize([f"cell={path}"], [])
    cell = value["runs"]["cell"]
    if value["protocol"] != protocol.audit_inputs(output / "inputs", 5):
        raise ValueError("Result does not use the historical Qwen 944x5 protocol")
    if (cell["weight_source"], cell["method"], cell["schedule"]) != ("generator", "framewise", "ffe"):
        raise ValueError("Expected raw framewise legacy FFE, not EMA/main_ffe/all1")
    if any(not math.isfinite(v) for v in cell["normalized_aggregates"].values()):
        raise ValueError("Nonfinite evaluation score")
    return cell


def report(output, reference_summary=None):
    health = audit_training(output, final=True)
    path = output / "eval/dmd400_ffe/dmd400_ffe_paper_protocol_summary.json"
    dmd = validated_summary(path, output)
    expected_checkpoint = run_dir(output) / "checkpoint_model_000400/model.pt"
    if Path(dmd["checkpoint_path"]).resolve() != expected_checkpoint.resolve():
        raise ValueError("DMD result is not the new continuous step400 checkpoint")
    ref_scores = {k: REFERENCE[k] for k in ("total_score", "quality_score", "semantic_score")}
    precision = REFERENCE["precision"]
    if reference_summary:
        ref = validated_summary(reference_summary, output)
        ref_scores = {k: v * 100 for k, v in ref["normalized_aggregates"].items()}
        if any(round(ref_scores[k], 2) != REFERENCE[k] for k in ref_scores):
            raise ValueError("Reference summary is not the archived 83.93/85.41/78.04 result")
        if not str(ref["checkpoint_path"]).endswith(REFERENCE["checkpoint_suffix"]):
            raise ValueError("GAN reference checkpoint identity does not match the historical run")
        precision = "full-precision audited historical summary"
    dmd_scores = {k: v * 100 for k, v in dmd["normalized_aggregates"].items()}
    result = {"training": health, "gan_reference": ref_scores, "dmd400": dmd_scores,
              "gan_minus_dmd_percentage_points": {k: ref_scores[k] - dmd_scores[k] for k in ref_scores},
              "reference_precision": precision, "reference_summary": str(reference_summary) if reference_summary else None,
              "reference_report_commit": EVAL_COMMIT,
              "historical_200_rounded": {"dmd_only": 77.21, "dmd_gan": 80.58},
              "visual_review": "See operator_notes.md; metric audit does not certify video quality."}
    write_json(output / "comparison.json", result)
    rows = ["| Model | Step | Total | Quality | Semantic |", "|---|---:|---:|---:|---:|"]
    for label, scores in (("Historical DMD+GAN", ref_scores), ("New DMD-only", dmd_scores)):
        rows.append(f"| {label} | 400 | {scores['total_score']:.2f} | {scores['quality_score']:.2f} | {scores['semantic_score']:.2f} |")
    delta = result["gan_minus_dmd_percentage_points"]["total_score"]
    (output / "comparison.md").write_text("\n".join(rows) +
        f"\n\nGAN − DMD-only Total: {delta:+.2f} points. Reference: {precision}.\n" +
        "Both use raw weights, pinned Qwen 944×5, legacy four-frame FFE.\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "check", "train", "health", "preview", "evaluate", "report"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name in ("train", "preview", "evaluate"):
            p.add_argument("--python", default=sys.executable)
            p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
            p.add_argument("--session", default=f"of_hist_dmd400_{name}")
            p.add_argument("--execute", action="store_true", help="Without this flag, only print the command")
        if name == "train":
            for key in ("generator", "teacher", "data", "prompt-cache", "wan-root"):
                p.add_argument(f"--{key}", required=True)
        if name == "evaluate":
            p.add_argument("--vbench-python", required=True)
        if name == "health":
            p.add_argument("--final", action="store_true")
        if name == "report":
            p.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if args.action == "prepare":
        prepare(output)
        print(f"Prepared historical DMD-only400: {output}")
        return
    check(output)
    if args.action == "check":
        print("PASS: pinned historical trainer, seed0/400 iterations, raw Qwen944x5 legacy FFE")
    elif args.action == "health":
        print(json.dumps(audit_training(output, final=args.final), indent=2))
    elif args.action == "report":
        print(json.dumps(report(output, args.reference_summary), indent=2))
    elif args.action == "train":
        command = train_command(output, args)
        print(shlex.join(command), flush=True)
        if args.execute:
            if run_dir(output).exists():
                raise FileExistsError("Training already launched; never resume/reuse this output directory")
            require_idle(args.gpus)
            link_assets(output, args.wan_root)
            # Fingerprint the initialization; large data/cache asset identity must
            # additionally be reconciled with historical run records by the operator.
            launch = {"command": command,
                "generator_sha256": protocol.sha256_file(Path(args.generator)),
                "wan_root": str(Path(args.wan_root).resolve())}
            if (output / "launch.json").exists() and json.loads((output / "launch.json").read_text()) != launch:
                raise ValueError("Previous preflight used different assets or commands; use a new task directory")
            write_json(output / "launch.json", launch)
            subprocess.run(command, cwd=output / "training", check=True, env=clean_env())
    else:
        command = eval_command(output, args, preview=args.action == "preview")
        print(shlex.join(command), flush=True)
        if args.execute:
            health = audit_training(output, final=True)
            require_idle(args.gpus)
            write_json(output / "training_audit.json", health)
            start_tmux(command, output / "evaluation", args.session,
                       output / f"logs/{args.action}.log")


if __name__ == "__main__":
    main()
