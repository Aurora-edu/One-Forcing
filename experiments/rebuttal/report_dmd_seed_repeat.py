#!/usr/bin/env python3
"""Audit a DMD-only training-seed repeat, without inventing a new GAN pair."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal import official_main_pair as pair
from experiments.rebuttal.consolidate_results import official_totals

BASELINE = REPO_ROOT / "experiments/rebuttal/results/official_main_gan_ablation.json"


def audit_training_log(run_dir: Path, min_step: int = 200) -> dict:
    """Check official stdout at step 10 or completion; do not launch/stop jobs."""
    if not 1 <= min_step <= 200:
        raise ValueError("min_step must be between 1 and 200")
    records = []
    for line in (run_dir / "train.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("{'step':") and "'critic_loss':" in line:
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError) as error:
                raise ValueError("Malformed/non-finite training record; check train.log") from error
            if any(
                not math.isfinite(value) for value in record.values()
                if isinstance(value, (float, int))
            ):
                raise ValueError(f"Non-finite training record at step {record.get('step')}")
            records.append(record)
    steps = [record.get("step") for record in records]
    if not min_step <= len(records) <= 200 or steps != list(range(1, len(records) + 1)):
        raise ValueError(f"Expected contiguous steps 1 through at least {min_step}, at most 200")

    generator_steps = []
    for record in records:
        step = record["step"]
        for key in ("gan_d_loss", "r1_loss", "r2_loss"):
            if record.get(key) != 0.0:
                raise ValueError(f"DMD-only step {step}: {key} must be present and zero")
        for key in ("critic_loss", "critic_grad_norm"):
            if not isinstance(record.get(key), (int, float)):
                raise ValueError(f"Step {step}: missing {key}")
        # Official training increments the printed step AFTER the update.
        is_generator_step = (step - 1) % 5 == 0
        if ("generator_loss" in record) != is_generator_step:
            raise ValueError(f"Unexpected generator update at printed step {step}")
        if is_generator_step:
            if record.get("gan_g_loss") != 0.0:
                raise ValueError(f"DMD-only step {step}: gan_g_loss must be present and zero")
            for key in ("dmd_loss", "generator_grad_norm", "generator_loss"):
                if not isinstance(record.get(key), (int, float)):
                    raise ValueError(f"Step {step}: missing {key}")
            if not math.isclose(record["generator_loss"], record["dmd_loss"], rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Step {step}: generator loss differs from DMD loss")
            generator_steps.append(step)
    if not any(record.get("generator_grad_norm", 0) > 0 for record in records):
        raise ValueError("No nonzero generator gradient observed")
    if not any(record["critic_grad_norm"] > 0 for record in records):
        raise ValueError("No nonzero critic gradient observed")
    return {
        "last_step": steps[-1],
        "generator_updates_logged": len(generator_steps),
        "critic_updates_logged": len(records),
        "all_gan_losses_zero": True,
        "all_logged_scalars_finite": True,
    }


def load_baseline(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    source = f"Aurora-edu/One-Forcing main@{pair.OFFICIAL_COMMIT}"
    if result.get("status") != "pass" or result.get("training_source") != source:
        raise ValueError("Baseline must be the audited official-main experiment")
    if result.get("shared_effective_seed") != 48491:
        raise ValueError("Expected the original seed-48491 DMD-only baseline")
    dimensions = result.get("dimensions", {}).get("dmd", {})
    totals = official_totals(dimensions)
    if len(dimensions) != 16 or totals is None or any(
        not math.isfinite(value) for value in dimensions.values()
    ):
        raise ValueError("Baseline requires 16 finite VBench dimensions")
    for key in pair.SCORE_KEYS:
        if not math.isclose(result["dmd_only"][key], totals[key] * 100, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Baseline aggregate does not match dimensions: {key}")
    return result


def build_report(
    official_root: Path, pair_dir: Path, run_dir: Path, eval_root: Path,
    baseline_path: Path = BASELINE,
) -> dict:
    baseline = load_baseline(baseline_path)
    manifest = pair.check(official_root, pair_dir)
    seed = manifest["shared_effective_seed"]
    if seed != 1:
        raise ValueError("This repeat preselects training seed 1; do not search for a favorable seed")
    training = audit_training_log(run_dir, min_step=200)
    # Only audit the newly trained DMD arm. No full-arm run/evaluation is needed.
    arm = pair._audit_arm("dmd", run_dir, eval_root, pair_dir, manifest)
    evaluation = arm["evaluation"]
    protocol = evaluation["protocol"]
    for key in (*pair.PAIRED_FIELDS, "prompt_count", "official_five_sample_protocol"):
        if protocol.get(key) is None or protocol[key] != baseline["eval_protocol"].get(key):
            raise ValueError(f"Repeat changed evaluation protocol: {key}")
    previous_dimensions = baseline["dimensions"]["dmd"]
    if set(evaluation["scores"]) != set(previous_dimensions):
        raise ValueError("Repeat changed VBench dimensions")
    scores = {key: evaluation["normalized_aggregates"][key] * 100 for key in pair.SCORE_KEYS}
    return {
        "schema_version": 1,
        "status": "pass",
        "comparison_type": "dmd_only_training_seed_repeat_not_gan_ablation",
        "training_source": baseline["training_source"],
        "baseline_training_seed": 48491,
        "repeat_training_seed": seed,
        "baseline_report_sha256": pair._hash(baseline_path),
        "baseline_dmd_only": baseline["dmd_only"],
        "repeat_dmd_only": scores,
        "delta_repeat_minus_baseline": {
            key: scores[key] - baseline["dmd_only"][key] for key in pair.SCORE_KEYS
        },
        "dimensions": {"baseline": previous_dimensions, "repeat": evaluation["scores"]},
        "eval_protocol": protocol,
        "checkpoint": arm["checkpoint"],
        "training_log_audit": training,
        "disclosed_environment_deviations": evaluation.get("disclosed_environment_deviations", {}),
        "note": "DMD+GAN was not retrained. This report does not estimate a same-seed GAN gain.",
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Official-main DMD-only: training-seed repeat", "",
        f"Training code: `{result['training_source']}`. Only DMD-only is retrained, "
        "from the original ODE initialization for 200 iterations (40 generator updates). "
        "Training seed changes from 48491 to 1; evaluation generation seeds stay fixed. "
        "Both evaluations use 944 × 5 Qwen-conditioned prompts, FFE, the generator "
        "weights (not EMA), and all 16 VBench dimensions.", "",
        "| DMD-only run | Total | Quality | Semantic |", "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Original seed 48491", "baseline_dmd_only"),
        ("Repeat seed 1", "repeat_dmd_only"),
        ("Difference (1 − 48491)", "delta_repeat_minus_baseline"),
    ):
        lines.append(f"| {label} | " + " | ".join(f"{result[key][field]:.2f}" for field in pair.SCORE_KEYS) + " |")
    lines.extend(["", result["note"], "", "| Dimension (raw score) | Seed 48491 | Seed 1 | Difference |", "|---|---:|---:|---:|"])
    for dimension, previous in sorted(result["dimensions"]["baseline"].items()):
        current = result["dimensions"]["repeat"][dimension]
        lines.append(f"| {dimension} | {previous:.6f} | {current:.6f} | {current - previous:+.6f} |")
    return "\n".join([*lines, ""])


def write_report(result: dict, prefix: Path) -> None:
    outputs = {
        Path(str(prefix) + ".json"): json.dumps(result, indent=2, sort_keys=True) + "\n",
        Path(str(prefix) + ".md"): render_markdown(result),
    }
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite report: {path}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    health = sub.add_parser("check-training")
    health.add_argument("--run_dir", type=Path, required=True)
    health.add_argument("--min_step", type=int, default=10)
    report = sub.add_parser("report")
    for name in ("official_root", "pair_dir", "run_dir", "eval_root", "output_prefix"):
        report.add_argument(f"--{name}", type=Path, required=True)
    report.add_argument("--baseline_json", type=Path, default=BASELINE)
    args = parser.parse_args()
    if args.command == "check-training":
        result = audit_training_log(args.run_dir.resolve(), args.min_step)
    else:
        result = build_report(
            args.official_root, args.pair_dir, args.run_dir.resolve(),
            args.eval_root, args.baseline_json,
        )
        write_report(result, args.output_prefix.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
