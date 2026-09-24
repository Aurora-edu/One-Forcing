#!/usr/bin/env python3
"""Audit the published-FFE DMD-only recipe and report its VBench result.

The published One-Forcing 83.76 is a manuscript reference, not a paired
re-evaluation on this script's generation manifest.
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

from utils.config import load_config
from experiments.rebuttal.audit_noema_checkpoint import audit_checkpoint
from experiments.rebuttal.paper_vbench_protocol import audit_run


BASE_CONFIG = REPO_ROOT / "ffe_config.yaml"
DMD_CONFIG = REPO_ROOT / "experiments/rebuttal/configs/train_dmd_only_published_ffe.yaml"
RUN_NAME = "dmd_only_published_ffe"
PUBLISHED_SCORES = {"total_score": 83.76, "quality_score": 85.22, "semantic_score": 77.91}
FIELDS = ("total_score", "quality_score", "semantic_score")


def verify_recipe() -> dict:
    base = OmegaConf.to_container(load_config(str(BASE_CONFIG)), resolve=True)
    dmd = OmegaConf.to_container(load_config(str(DMD_CONFIG)), resolve=True)
    if not isinstance(base, dict) or not isinstance(dmd, dict):
        raise ValueError("Training configurations must be mappings")
    if dmd.get("dataset_type") != "clean_latent_lmdb":
        raise ValueError("DMD training must use clean-latent LMDB")
    dmd.pop("dataset_type")  # train.py receives this same setting from the launcher.
    expected = dict(base)
    expected.update(gan_g_weight=0.0, gan_d_weight=0.0)
    if dmd != expected:
        changes = {
            key: (expected.get(key), dmd.get(key))
            for key in sorted(expected.keys() | dmd.keys())
            if expected.get(key) != dmd.get(key)
        }
        raise ValueError(f"DMD recipe differs from FFE baseline beyond GAN weights: {changes}")
    if base.get("gan_g_weight", 0) <= 0 or base.get("gan_d_weight", 0) <= 0:
        raise ValueError("Published baseline must have both GAN terms enabled")
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
        if dmd.get(key) != value:
            raise ValueError(f"Unexpected {key}: {dmd.get(key)!r}, expected {value!r}")
    if dmd.get("first_frame_denoising_step_list") != [1000, 750, 500, 250]:
        raise ValueError("First FFE block must have four denoising updates")
    if dmd.get("denoising_step_list") != [1000]:
        raise ValueError("Subsequent frames must have one denoising update")
    return {
        "published_recipe": str(BASE_CONFIG),
        "dmd_recipe": str(DMD_CONFIG),
        "only_training_objective_difference": ["gan_g_weight", "gan_d_weight"],
        "gan_weights": [0.0, 0.0],
        "training_steps": 200,
    }


def build_report(run_dir: Path, eval_root: Path) -> dict:
    recipe = verify_recipe()
    run_dir = run_dir.resolve()
    eval_root = eval_root.resolve()
    resolved = OmegaConf.to_container(load_config(str(run_dir / "resolved_config.yaml")), resolve=True)
    expected = OmegaConf.to_container(load_config(str(DMD_CONFIG)), resolve=True)
    # Only launch-time paths, metadata, and hardware flags may differ.
    runtime_fields = {
        "generator_ckpt", "teacher_model_path", "data_path", "prompt_embedding_cache_path",
        "logdir", "wandb_save_dir", "disable_wandb", "no_save", "no_visualize",
        "config_name", "text_encoder_cpu_offload", "real_score_cpu_offload",
        "fake_score_cpu_offload", "manual_generator_backward",
        "generator_optimizer_state_cpu_offload", "rank0_preload_generator_ckpt",
    }
    for key, value in expected.items():
        if key not in runtime_fields and resolved.get(key) != value:
            raise ValueError(f"Executed training changed {key}: {resolved.get(key)!r} != {value!r}")
    unexpected = set(resolved) - set(expected) - runtime_fields
    if unexpected:
        raise ValueError(f"Unexpected executed training settings: {sorted(unexpected)}")
    if resolved.get("no_save") or not resolved.get("disable_wandb") or resolved.get("seed") != 0:
        raise ValueError("Run must save weights, use fixed seed 0, and disable W&B")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("world_size") != 8:
        raise ValueError(f"Expected published 8-rank global batch, got {metadata.get('world_size')!r}")
    if Path(metadata.get("config_path", "")).resolve() != DMD_CONFIG.resolve():
        raise ValueError("Training did not load the audited DMD-only configuration")
    completion = json.loads((run_dir / "training.done").read_text(encoding="utf-8"))
    if completion != {"final_step": 200, "max_steps": 200}:
        raise ValueError(f"DMD training did not finish exactly 200 steps: {completion}")
    checkpoint = run_dir / "checkpoint_model_000200/model.pt"
    ckpt_audit = audit_checkpoint(checkpoint, expected_step=200)
    evaluated = audit_run(eval_root, RUN_NAME, 5, "ffe", "generator")
    if Path(evaluated["checkpoint_path"]).resolve() != checkpoint.resolve():
        raise ValueError("Evaluation used a checkpoint other than the DMD-only step-200 output")
    if evaluated["checkpoint_size_bytes"] != ckpt_audit["checkpoint_size_bytes"]:
        raise ValueError("Evaluated checkpoint size has changed")
    if evaluated["method"] != "framewise" or len(evaluated["scores"]) != 16:
        raise ValueError("Expected complete 16-dimensional framewise evaluation")
    protocol = evaluated["protocol"]
    if (protocol.get("prompt_count") != 944 or protocol.get("samples_per_prompt") != 5
            or protocol.get("conditioning") != "pinned_qwen_rewrite"
            or protocol.get("scoring_prompt") != "original_vbench_prompt"):
        raise ValueError("VBench evaluation is not the full Qwen-conditioned protocol")
    dmd_scores = {key: evaluated["normalized_aggregates"][key] * 100 for key in FIELDS}
    if not all(math.isfinite(value) for value in dmd_scores.values()):
        raise ValueError("Non-finite DMD-only VBench aggregate")
    return {
        "status": "pass",
        "comparison_type": "published_reference_vs_new_run_not_paired_generation_seeds",
        "published_one_forcing": PUBLISHED_SCORES,
        "dmd_only": dmd_scores,
        "published_minus_dmd": {key: PUBLISHED_SCORES[key] - dmd_scores[key] for key in FIELDS},
        "recipe": recipe,
        "training_run_dir": str(run_dir),
        "training_metadata": metadata,
        "checkpoint_audit": ckpt_audit,
        "vbench_summary": evaluated,
        "note": "83.76/85.22/77.91 are the paper's reported One-Forcing scores; its exact generation manifest is not archived here. These differences are cross-run, not paired-seed estimates.",
    }


def render_markdown(report: dict) -> str:
    published = report["published_one_forcing"]
    dmd = report["dmd_only"]
    delta = report["published_minus_dmd"]
    cells = lambda scores: " | ".join(f"{scores[key]:.2f}" for key in FIELDS)
    return "\n".join([
        "# Published-FFE GAN ablation", "",
        "| Method | Total | Quality | Semantic |", "|---|---:|---:|---:|",
        f"| One-Forcing (paper reference) | {cells(published)} |",
        f"| DMD-only (new FFE-aligned run) | {cells(dmd)} |",
        f"| Difference (reference − DMD-only) | {cells(delta)} |", "",
        "Both training configurations use the same FFE framewise recipe and configured ODE initialization, "
        "clean-latent data type, seed 0, and 200-step budget; only the two GAN weights "
        "are zeroed in DMD-only. The new evaluation uses 944 Qwen-rewritten prompts, "
        "five samples per prompt, all 16 VBench dimensions, FFE, and generator weights.", "",
        report["note"], "",
        f"DMD-only checkpoint: `{report['checkpoint_audit']['checkpoint_path']}`", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check_recipe")
    report = sub.add_parser("report")
    report.add_argument("--run_dir", required=True)
    report.add_argument("--eval_root", required=True)
    report.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    if args.command == "check_recipe":
        print(json.dumps(verify_recipe(), indent=2, sort_keys=True))
    else:
        result = build_report(Path(args.run_dir), Path(args.eval_root))
        prefix = Path(args.output_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prefix.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
        print(f"PASS: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
