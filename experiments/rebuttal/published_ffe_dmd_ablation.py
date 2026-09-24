#!/usr/bin/env python3
"""Audit a paired FFE-aligned One-Forcing vs DMD-only experiment.

Both arms are newly trained from the same ODE checkpoint. The manuscript's
83.76 headline is displayed only as an external reproduction check; the GAN
effect is calculated exclusively from the two newly evaluated checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.audit_noema_checkpoint import audit_checkpoint
from experiments.rebuttal.paper_vbench_protocol import audit_run
from experiments.rebuttal.summarize_paper_aligned_vbench import PAIRED_FIELDS
from utils.config import load_config


BASE_CONFIG = REPO_ROOT / "ffe_config.yaml"
CONFIGS = REPO_ROOT / "experiments/rebuttal/configs"
FULL_CONFIG = CONFIGS / "train_one_forcing_published_ffe.yaml"
DMD_CONFIG = CONFIGS / "train_dmd_only_published_ffe.yaml"
RUN_NAMES = {
    "full": "one_forcing_paired_ffe",
    "dmd": "dmd_only_paired_ffe",
}
CONFIG_PATHS = {"full": FULL_CONFIG, "dmd": DMD_CONFIG}
FIELDS = ("total_score", "quality_score", "semantic_score")
PAPER_REFERENCE = {"total_score": 83.76, "quality_score": 85.22, "semantic_score": 77.91}

# Values that train.py/launch_train.sh add at launch time. Every field here,
# except logdir/config_name, must still match across the two executed runs.
RUNTIME_FIELDS = {
    "generator_ckpt", "teacher_model_path", "data_path", "prompt_embedding_cache_path",
    "logdir", "wandb_save_dir", "disable_wandb", "no_save", "no_visualize",
    "config_name", "text_encoder_cpu_offload", "real_score_cpu_offload",
    "fake_score_cpu_offload", "manual_generator_backward",
    "generator_optimizer_state_cpu_offload", "rank0_preload_generator_ckpt",
}
ARM_SPECIFIC_FIELDS = {"gan_g_weight", "gan_d_weight", "logdir", "config_name"}


def _mapping(path: Path) -> dict:
    value = OmegaConf.to_container(load_config(str(path)), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in configuration {path}")
    return value


def _assert_equal(actual: dict, expected: dict, description: str) -> None:
    if actual != expected:
        changes = {
            key: (expected.get(key), actual.get(key))
            for key in sorted(set(expected) | set(actual))
            if actual.get(key) != expected.get(key)
        }
        raise ValueError(f"{description} differs: {changes}")


def verify_recipe() -> dict:
    base, full, dmd = (_mapping(path) for path in (BASE_CONFIG, FULL_CONFIG, DMD_CONFIG))
    if base.get("gan_g_weight") != 0.03 or base.get("gan_d_weight") != 0.03:
        raise ValueError("Full One-Forcing recipe must retain GAN weights 0.03/0.03")
    expected_full = dict(base, dataset_type="clean_latent_lmdb")
    _assert_equal(full, expected_full, "Full arm versus documented FFE recipe")
    expected_dmd = dict(expected_full, gan_g_weight=0.0, gan_d_weight=0.0)
    _assert_equal(dmd, expected_dmd, "DMD-only arm versus full One-Forcing")
    fixed = {
        "rollout_schedule": "first4then1",
        "first_rollout_num_frames": 4,
        "num_frame_per_block": 1,
        "max_steps": 200,
        "log_iters": 50,
        "seed": 0,
        "randomize_seed": False,
        "dfake_gen_update_ratio": 5,
        "r1_weight": 0.0,
        "r2_weight": 0.0,
    }
    for key, value in fixed.items():
        if full.get(key) != value:
            raise ValueError(f"Documented FFE recipe has unexpected {key}: {full.get(key)!r}")
    if full.get("first_frame_denoising_step_list") != [1000, 750, 500, 250]:
        raise ValueError("First FFE block must have four denoising updates")
    if full.get("denoising_step_list") != [1000]:
        raise ValueError("Subsequent frames must have one denoising update")
    return {
        "documented_one_forcing_recipe": str(BASE_CONFIG),
        "full_config": str(FULL_CONFIG),
        "dmd_config": str(DMD_CONFIG),
        "only_arm_difference": ["gan_g_weight", "gan_d_weight"],
        "training_steps_per_arm": 200,
    }


def audit_training(label: str, run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    config_path = CONFIG_PATHS[label]
    expected = _mapping(config_path)
    resolved = _mapping(run_dir / "resolved_config.yaml")
    for key, value in expected.items():
        if key not in RUNTIME_FIELDS and resolved.get(key) != value:
            raise ValueError(f"{label}: executed training changed {key}: {resolved.get(key)!r} != {value!r}")
    unexpected = set(resolved) - set(expected) - RUNTIME_FIELDS
    if unexpected:
        raise ValueError(f"{label}: unexpected executed settings: {sorted(unexpected)}")
    if resolved.get("no_save") or not resolved.get("disable_wandb") or resolved.get("seed") != 0:
        raise ValueError(f"{label}: expected saved checkpoints, fixed seed 0, and no W&B")

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("world_size") != 8:
        raise ValueError(f"{label}: expected published 8-rank global batch, got {metadata.get('world_size')!r}")
    if Path(metadata.get("config_path", "")).resolve() != config_path.resolve():
        raise ValueError(f"{label}: training did not load its audited configuration")
    if not metadata.get("git_commit") or metadata.get("git_status_porcelain") != []:
        raise ValueError(f"{label}: training must use a recorded clean code revision")

    completion = json.loads((run_dir / "training.done").read_text(encoding="utf-8"))
    if completion != {"final_step": 200, "max_steps": 200}:
        raise ValueError(f"{label}: training did not complete exactly 200 steps: {completion}")
    checkpoint = run_dir / "checkpoint_model_000200/model.pt"
    ckpt_audit = audit_checkpoint(checkpoint, expected_step=200)
    return {
        "run_dir": str(run_dir),
        "resolved_config": resolved,
        "metadata": metadata,
        "checkpoint_audit": ckpt_audit,
    }


def audit_evaluation(label: str, eval_root: Path, checkpoint_audit: dict) -> dict:
    evaluated = audit_run(eval_root.resolve(), RUN_NAMES[label], 5, "ffe", "generator")
    if Path(evaluated["checkpoint_path"]).resolve() != Path(checkpoint_audit["checkpoint_path"]).resolve():
        raise ValueError(f"{label}: evaluation did not use its own step-200 checkpoint")
    if evaluated["checkpoint_size_bytes"] != checkpoint_audit["checkpoint_size_bytes"]:
        raise ValueError(f"{label}: evaluated checkpoint size changed")
    if evaluated["method"] != "framewise" or len(evaluated["scores"]) != 16:
        raise ValueError(f"{label}: expected complete 16-dimensional framewise evaluation")
    protocol = evaluated["protocol"]
    if (protocol.get("prompt_count") != 944 or protocol.get("samples_per_prompt") != 5
            or protocol.get("conditioning") != "pinned_qwen_rewrite"
            or protocol.get("scoring_prompt") != "original_vbench_prompt"):
        raise ValueError(f"{label}: not the complete Qwen-conditioned VBench protocol")
    scores = {key: evaluated["normalized_aggregates"][key] * 100 for key in FIELDS}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError(f"{label}: non-finite VBench aggregate")
    return evaluated


def build_report(full_run_dir: Path, dmd_run_dir: Path, full_eval_root: Path, dmd_eval_root: Path) -> dict:
    recipe = verify_recipe()
    training = {
        "full": audit_training("full", full_run_dir),
        "dmd": audit_training("dmd", dmd_run_dir),
    }
    full_config = training["full"]["resolved_config"]
    dmd_config = training["dmd"]["resolved_config"]
    for key in sorted(set(full_config) | set(dmd_config)):
        if key not in ARM_SPECIFIC_FIELDS and full_config.get(key) != dmd_config.get(key):
            raise ValueError(f"Executed training arms differ in {key}: {full_config.get(key)!r} != {dmd_config.get(key)!r}")
    full_metadata = training["full"]["metadata"]
    dmd_metadata = training["dmd"]["metadata"]
    if full_metadata["git_commit"] != dmd_metadata["git_commit"]:
        raise ValueError("Paired training arms used different code revisions")

    evaluations = {
        "full": audit_evaluation("full", full_eval_root, training["full"]["checkpoint_audit"]),
        "dmd": audit_evaluation("dmd", dmd_eval_root, training["dmd"]["checkpoint_audit"]),
    }
    full_protocol = evaluations["full"]["protocol"]
    dmd_protocol = evaluations["dmd"]["protocol"]
    for field in (*PAIRED_FIELDS, "official_five_sample_protocol"):
        if full_protocol.get(field) != dmd_protocol.get(field):
            raise ValueError(f"Evaluations are not paired: {field} differs")
    if set(evaluations["full"]["scores"]) != set(evaluations["dmd"]["scores"]):
        raise ValueError("VBench dimension sets differ")

    scores = {
        label: {key: record["normalized_aggregates"][key] * 100 for key in FIELDS}
        for label, record in evaluations.items()
    }
    gain = {key: scores["full"][key] - scores["dmd"][key] for key in FIELDS}
    dimension_gain = {
        dimension: evaluations["full"]["scores"][dimension] - evaluations["dmd"]["scores"][dimension]
        for dimension in sorted(evaluations["full"]["scores"])
    }
    return {
        "status": "pass",
        "comparison_type": "newly_trained_paired_objective_ablation",
        "full_one_forcing": scores["full"],
        "dmd_only": scores["dmd"],
        "gan_gain": gain,
        "dimension_gain": dimension_gain,
        "paper_headline_reference_not_used_for_gain": PAPER_REFERENCE,
        "full_minus_paper_reference": {
            key: scores["full"][key] - PAPER_REFERENCE[key] for key in FIELDS
        },
        "recipe": recipe,
        "protocol": full_protocol,
        "training": training,
        "evaluations": evaluations,
    }


def render_markdown(report: dict) -> str:
    def cells(scores: dict, signed: bool = False) -> str:
        style = "+.2f" if signed else ".2f"
        return " | ".join(format(scores[key], style) for key in FIELDS)

    return "\n".join([
        "# Paired FFE One-Forcing GAN ablation", "",
        "| Arm | Total | Quality | Semantic |", "|---|---:|---:|---:|",
        f"| DMD+GAN (new One-Forcing run) | {cells(report['full_one_forcing'])} |",
        f"| DMD-only (new run) | {cells(report['dmd_only'])} |",
        f"| GAN gain (paired full − DMD-only) | {cells(report['gan_gain'], signed=True)} |", "",
        "Both arms were trained from the same configured ODE checkpoint on the same clean-latent "
        "data with seed 0, eight ranks, 200 steps and FFE first-4-then-1 rollout. "
        "Only the GAN objective weights differ. Evaluation uses the same pinned Qwen "
        "rewrite and generation manifest, five samples for each of 944 prompts, all "
        "16 VBench dimensions, FFE, and raw generator weights.", "",
        "The manuscript headline 83.76 / 85.22 / 77.91 is a reproduction check, "
        "not one side of this GAN-gain calculation. New DMD+GAN minus headline: "
        f"{cells(report['full_minus_paper_reference'], signed=True)} (total / quality / semantic).", "",
        f"Full checkpoint: `{report['training']['full']['checkpoint_audit']['checkpoint_path']}`  ",
        f"DMD-only checkpoint: `{report['training']['dmd']['checkpoint_audit']['checkpoint_path']}`", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check_recipe")
    report = sub.add_parser("report")
    report.add_argument("--full_run_dir", required=True)
    report.add_argument("--dmd_run_dir", required=True)
    report.add_argument("--full_eval_root", required=True)
    report.add_argument("--dmd_eval_root", required=True)
    report.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    if args.command == "check_recipe":
        print(json.dumps(verify_recipe(), indent=2, sort_keys=True))
    else:
        result = build_report(
            Path(args.full_run_dir), Path(args.dmd_run_dir),
            Path(args.full_eval_root), Path(args.dmd_eval_root),
        )
        prefix = Path(args.output_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prefix.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
        print(f"PASS: paired GAN ablation: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
