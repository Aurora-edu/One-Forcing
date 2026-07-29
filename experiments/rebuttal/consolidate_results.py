#!/usr/bin/env python3
"""Consolidate official rebuttal outputs into exactly three upload files:

    experiments/rebuttal/results/final_metrics.json
    experiments/rebuttal/results/final_metrics.csv
    experiments/rebuttal/results/RUN_REPORT.md

Reads only aggregated result JSONs produced by the official pipeline. No raw
videos, per-sample records, checkpoints, or logs are copied into the results.
"""
import argparse
import csv
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Official VBench leaderboard constants (Vchitect/VBench_Leaderboard).
QUALITY_LIST = [
    "subject consistency", "background consistency", "temporal flickering",
    "motion smoothness", "aesthetic quality", "imaging quality", "dynamic degree",
]
SEMANTIC_LIST = [
    "object class", "multiple objects", "human action", "color",
    "spatial relationship", "scene", "appearance style", "temporal style",
    "overall consistency",
]
DIM_WEIGHT = {
    "subject consistency": 1, "background consistency": 1, "temporal flickering": 1,
    "motion smoothness": 1, "aesthetic quality": 1, "imaging quality": 1,
    "dynamic degree": 0.5, "object class": 1, "multiple objects": 1,
    "human action": 1, "color": 1, "spatial relationship": 1, "scene": 1,
    "appearance style": 1, "temporal style": 1, "overall consistency": 1,
}
NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}
QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1


def dim_key(dimension):
    return dimension.replace("_", " ")


def load_vbench_results(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = {}
    for dimension, value in payload.items():
        if isinstance(value, list):
            value = value[0]
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite {dimension}")
        scores[dimension] = value
    return scores


def normalized(dimension, score):
    rng = NORMALIZE_DIC[dim_key(dimension)]
    return (score - rng["Min"]) / (rng["Max"] - rng["Min"])


def official_totals(scores):
    """Official VBench leaderboard aggregation (weighted normalized means,
    total = (4*quality + 1*semantic) / 5). Requires all 16 dimensions."""
    keys = {dim_key(d) for d in scores}
    if not (set(QUALITY_LIST) <= keys and set(SEMANTIC_LIST) <= keys):
        return None
    by_key = {dim_key(d): s for d, s in scores.items()}

    def weighted(dims):
        num = sum(normalized_by_key(k, by_key[k]) * DIM_WEIGHT[k] for k in dims)
        den = sum(DIM_WEIGHT[k] for k in dims)
        return num / den

    def normalized_by_key(key, score):
        rng = NORMALIZE_DIC[key]
        return (score - rng["Min"]) / (rng["Max"] - rng["Min"])

    quality = weighted(QUALITY_LIST)
    semantic = weighted(SEMANTIC_LIST)
    total = (QUALITY_WEIGHT * quality + SEMANTIC_WEIGHT * semantic) / (
        QUALITY_WEIGHT + SEMANTIC_WEIGHT
    )
    return {
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": total,
    }


def latency_summary(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = payload["summary"]
    keep = {}
    for name, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            keep[name] = {
                k: stats[k] for k in ("mean", "std", "min", "max") if k in stats
            }
    trials = payload.get("trials")
    num_trials = len(trials) if isinstance(trials, list) else trials
    return {
        "hardware": payload.get("hardware"),
        "trials": num_trials,
        "warmup": payload.get("warmup"),
        "num_output_frames": payload.get("num_output_frames"),
        "include_vae": payload.get("include_vae"),
        "nfe": payload.get("nfe") or payload.get("num_function_evaluations"),
        "summary_ms": keep,
    }


def training_summary(run_dir, max_steps):
    run_dir = Path(run_dir)
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    assert records[-1]["step"] == max_steps
    ckpts = sorted(run_dir.glob("checkpoint_model_*/model.pt"))
    times = [c.stat().st_mtime for c in ckpts]
    started = (run_dir / "resolved_config.yaml").stat().st_mtime
    per_iter = [
        float(r["per_iteration_time"]) for r in records if "per_iteration_time" in r
    ]
    gen_records = [r for r in records if "generator_loss" in r]
    return {
        "run_dir": str(run_dir.relative_to(REPO)),
        "seed": int((run_dir / "runtime_seed.txt").read_text().strip()),
        "final_step": records[-1]["step"],
        "num_metric_records": len(records),
        "checkpoints": [c.parent.name for c in ckpts],
        "wallclock_hours_first_to_done": round(
            ((run_dir / "training.done").stat().st_mtime - started) / 3600, 3
        ),
        "mean_seconds_per_logged_iteration": (
            round(statistics.fmean(per_iter), 3) if per_iter else None
        ),
        "final_generator_loss": (
            gen_records[-1].get("generator_loss") if gen_records else None
        ),
        "final_critic_loss": records[-1].get("critic_loss"),
        "all_metrics_finite": all(
            math.isfinite(v)
            for r in records
            for v in r.values()
            if isinstance(v, (int, float))
        ),
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="experiments/rebuttal/results")
    args = parser.parse_args()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
        capture_output=True, text=True,
    ).stdout.strip()

    clock = (REPO / "logs_setup/DEADLINE_CLOCK_START.txt").read_text().splitlines()

    final_conditions = {
        "main_step600_ffe": "eval/final/main_step600_ffe/vbench/main_step600_ffe_eval_results.json",
        "main_step600_all1": "eval/final/main_step600_all1/vbench/main_step600_all1_eval_results.json",
        "of4_step300_all4": "eval/final/of4_step300_all4/vbench/of4_step300_all4_eval_results.json",
        "sf4_all4": "eval/final/sf4_all4/vbench/sf4_all4_eval_results.json",
    }
    gan_conditions = {
        "main_step200_ffe_gan7": "eval/gan/main_step200_ffe_gan7/vbench/main_step200_ffe_gan7_eval_results.json",
        "dmd_step200_ffe_gan7": "eval/gan/dmd_step200_ffe_gan7/vbench/dmd_step200_ffe_gan7_eval_results.json",
    }

    vbench_final = {}
    for name, rel in final_conditions.items():
        scores = load_vbench_results(REPO / rel)
        vbench_final[name] = {
            "per_dimension": scores,
            "official": official_totals(scores),
            "num_prompts": 944,
            "samples_per_prompt": 5,
            "num_videos": 4720,
            "generation_seed_base": 0,
        }

    vbench_gan = {}
    for name, rel in gan_conditions.items():
        scores = load_vbench_results(REPO / rel)
        vbench_gan[name] = {
            "per_dimension": scores,
            "unweighted_mean_of_7_dims": statistics.fmean(scores.values()),
            "num_prompts": 326,
            "samples_per_prompt": 5,
            "num_videos": 1630,
        }
    gan_delta = {
        d: vbench_gan["main_step200_ffe_gan7"]["per_dimension"][d]
        - vbench_gan["dmd_step200_ffe_gan7"]["per_dimension"][d]
        for d in vbench_gan["main_step200_ffe_gan7"]["per_dimension"]
    }

    # Stability sweep
    sweep_rows = list(
        csv.DictReader(open(REPO / "eval/stability/main_ffe/vbench_sweep.csv"))
    )
    stability = {}
    for row in sweep_rows:
        result_path = Path(row["result_json"])
        if not result_path.is_absolute():
            result_path = (REPO / "eval/stability/main_ffe" / result_path).resolve()
        stability[int(row["step"])] = load_vbench_results(result_path)

    diversity = {
        "main_step200": read_json(REPO / "eval/diversity/main_step200.json"),
        "dmd_step200": read_json(REPO / "eval/diversity/dmd_step200.json"),
    }
    diversity_out = {
        name: {
            k: payload.get(k)
            for k in (
                "mean_pairwise_lpips",
                "sample_std_over_prompts",
                "standard_error",
                "bootstrap_95ci",
                "num_prompts",
                "samples_per_prompt",
                "metric",
            )
        }
        for name, payload in diversity.items()
    }

    fvd = {}
    for name in ("main_step200", "dmd_step200"):
        payload = read_json(REPO / f"eval/fvd/{name}.json")
        fvd[name] = {
            k: payload.get(k)
            for k in (
                "fvd",
                "num_real_videos",
                "num_fake_videos",
                "num_sampled_frames",
                "feature_distribution_metrics",
                "bootstrap",
            )
        }

    latency = {
        "main600_all1": latency_summary(REPO / "eval/latency/main600_all1.json"),
        "main600_ffe": latency_summary(REPO / "eval/latency/main600_ffe.json"),
        "main600_all4": latency_summary(REPO / "eval/latency/main600_all4.json"),
        "of4_step300_all4": latency_summary(
            REPO / "eval/latency/of4_step300_all4.json"
        ),
    }

    training = {
        "train_1step_one_forcing_600": training_summary(
            REPO / "runs/rebuttal/train_1step_one_forcing_600/seed_0", 600
        ),
        "train_1step_dmd_only_200": training_summary(
            REPO / "runs/rebuttal/train_1step_dmd_only_200/seed_0", 200
        ),
        "train_4step_one_forcing_300": training_summary(
            REPO / "runs/rebuttal/train_4step_one_forcing_300/seed_0", 300
        ),
    }

    hardware = {
        "training_node": "a3u50asia-a3u-44 (Slurm job 9558), 8x NVIDIA H200 141GB, driver 570.172.08",
        "eval_nodes": [
            "a3u50asia-a3u-44 (track A)",
            "a3u50asia-a3u-38 (track B, Slurm job 9759)",
            "a3u50asia-a3u-0 (track C, Slurm job 9760)",
        ],
        "gpu_ids": "0,1,2,3,4,5,6,7 on each node (all verified idle before use)",
        "torch": "2.5.1+cu124",
        "note": (
            "README's 8xA100 assumption replaced by measured 8xH200; evaluation "
            "was fanned out to three identical H200 nodes on user instruction, "
            "each condition still 8-way sharded with identical manifests/seeds."
        ),
    }

    # ---- Optional no-EMA ablation (raw generator weights, post-deadline
    # operator-requested addendum; separate *_noema outputs) ----
    noema = None
    noema_final_conditions = {
        "main_step600_ffe_noema": "eval/final_noema/main_step600_ffe/vbench/main_step600_ffe_noema_eval_results.json",
        "main_step600_all1_noema": "eval/final_noema/main_step600_all1/vbench/main_step600_all1_noema_eval_results.json",
        "of4_step300_all4_noema": "eval/final_noema/of4_step300_all4/vbench/of4_step300_all4_noema_eval_results.json",
    }
    if all((REPO / rel).is_file() for rel in noema_final_conditions.values()):
        noema_final = {}
        for name, rel in noema_final_conditions.items():
            scores = load_vbench_results(REPO / rel)
            noema_final[name] = {
                "per_dimension": scores,
                "official": official_totals(scores),
                "num_prompts": 944,
                "samples_per_prompt": 5,
                "num_videos": 4720,
            }
        noema_gan = {
            name: load_vbench_results(REPO / rel)
            for name, rel in {
                "main_step200_ffe_gan7_noema": "eval/gan_noema/main_step200_ffe_gan7/vbench/main_step200_ffe_gan7_noema_eval_results.json",
                "dmd_step200_ffe_gan7_noema": "eval/gan_noema/dmd_step200_ffe_gan7/vbench/dmd_step200_ffe_gan7_noema_eval_results.json",
            }.items()
        }
        noema_stability = {}
        sweep_csv = REPO / "eval/stability_noema/main_ffe/vbench_sweep.csv"
        for row in csv.DictReader(open(sweep_csv)):
            rp = Path(row["result_json"])
            if not rp.is_absolute():
                rp = (sweep_csv.parent / rp).resolve()
            noema_stability[int(row["step"])] = load_vbench_results(rp)
        noema = {
            "note": (
                "Ablation with raw (non-EMA) generator weights, requested after "
                "the official EMA results were finalized. Same manifests, seeds, "
                "schedules, and pipeline; only --use_ema removed. SF4 no-EMA is "
                "impossible: the released self_forcing_dmd.pt contains only "
                "generator_ema."
            ),
            "vbench_final": noema_final,
            "gan_ablation_vbench7_step200_paired": {
                "conditions": noema_gan,
                "delta_main_minus_dmd": {
                    d: noema_gan["main_step200_ffe_gan7_noema"][d]
                    - noema_gan["dmd_step200_ffe_gan7_noema"][d]
                    for d in noema_gan["main_step200_ffe_gan7_noema"]
                },
            },
            "stability_vbench5_steps": noema_stability,
            "diversity_lpips": {
                name: {
                    k: read_json(REPO / f"eval/diversity_noema/{name}.json").get(k)
                    for k in ("mean_pairwise_lpips", "bootstrap_95ci")
                }
                for name in ("main_step200", "dmd_step200")
            },
            "fvd_prdc_256": {
                name: {
                    k: read_json(REPO / f"eval/fvd_noema/{name}.json").get(k)
                    for k in ("fvd", "feature_distribution_metrics")
                }
                for name in ("main_step200", "dmd_step200")
            },
            "latency_h200_21latents": {
                name: latency_summary(REPO / f"eval/latency_noema/{name}.json")
                for name in ("main600_all1", "main600_ffe", "main600_all4",
                             "of4_step300_all4")
            },
        }

    # ---- Optional ordered no-EMA follow-up experiments (NOEMA_FOLLOWUP.md) ----
    followup = {}
    p = REPO / "eval/followup/step200_1v4/step200_fourstep_comparison.json"
    if p.is_file():
        d = read_json(p)
        followup["step200_one_vs_four_step"] = {
            "protocol": d.get("protocol"),
            "runs": {
                name: block.get("normalized_aggregates")
                for name, block in d.get("runs", {}).items()
            },
            "comparisons": {
                name: block.get("normalized_aggregates")
                for name, block in d.get("comparisons", {}).items()
            },
        }
    p = REPO / "eval/followup/raw_step200_step400/raw_step200_step400_full_vbench.json"
    if p.is_file():
        d = read_json(p)
        followup["raw_step200_step400_full_vbench"] = {
            "protocol": d.get("protocol"),
            "runs": {
                name: block.get("normalized_aggregates")
                for name, block in d.get("runs", {}).items()
            },
            "comparisons": {
                name: {
                    "normalized_aggregates": block.get("normalized_aggregates"),
                    "per_dimension": block.get("scores"),
                }
                for name, block in d.get("comparisons", {}).items()
            },
        }
    p = REPO / "eval/followup/curvature_control/curvature_causal_summary.json"
    if p.is_file():
        followup["curvature_causal_intervention"] = read_json(p)
    cf = {}
    for sched in ("all1", "all4"):
        p = REPO / f"eval/diversity_noema/cf_init_{sched}.json"
        if p.is_file():
            d = read_json(p)
            cf[sched] = {
                "mean_pairwise_lpips": d.get("mean_pairwise_lpips"),
                "bootstrap_95ci": d.get("bootstrap_95ci"),
            }
    if cf:
        followup["causal_forcing_init_diversity_lpips"] = cf

    # ---- Reviewer runbook experiments (REVIEWER_EXPERIMENTS_RUNBOOK.md) ----
    reviewer = {}
    for key, rel in {
        "qwen_matched_4step": "eval/reviewer/qwen_matched_4step_all_gpu/qwen_matched_4step_summary.json",
        "dmd_only_step200_full16": "eval/reviewer/dmd_only_step200_full16/dmd_only_step200_full16_summary.json",
        "curvature_cd_small": "eval/reviewer/curvature_cd_small/curvature_cd_small_summary.json",
    }.items():
        p = REPO / rel
        if p.is_file():
            reviewer[key] = read_json(p)

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_branch": branch,
        "git_commit": commit,
        "deadline_clock": {
            "start_utc": clock[0],
            "deadline_utc": clock[1].split(": ", 1)[-1],
        },
        "hardware": hardware,
        "training_seed": 0,
        "single_seed_disclosure": (
            "All training runs use seed 0 (single seed). VBench 5 samples/prompt "
            "and diversity 4 samples/prompt are generation seeds, not training "
            "seeds. The stability curve uses four checkpoints of one run and "
            "does not represent cross-seed variance."
        ),
        "training": training,
        "vbench_final_official": vbench_final,
        "gan_ablation_vbench7_step200_paired": {
            "conditions": vbench_gan,
            "delta_main_minus_dmd": gan_delta,
        },
        "stability_vbench5_steps": stability,
        "ffe_comparison": {
            "note": "FFE on/off both use MAIN600 (checkpoint_model_000600), official manifest.",
            "ffe_on": vbench_final["main_step600_ffe"]["official"],
            "ffe_off_all1": vbench_final["main_step600_all1"]["official"],
        },
        "fourstep_comparison": {
            "note": "Both all4 schedule, shared official manifest/seeds; SF4 is the released gdhe17/Self-Forcing self_forcing_dmd.pt (sha256 verified).",
            "one_forcing_4step_step300": vbench_final["of4_step300_all4"]["official"],
            "self_forcing_4step": vbench_final["sf4_all4"]["official"],
        },
        "diversity_lpips": {
            "conditions": diversity_out,
            "delta_main_minus_dmd": (
                diversity_out["main_step200"]["mean_pairwise_lpips"]
                - diversity_out["dmd_step200"]["mean_pairwise_lpips"]
            ),
            "sf1_condition": (
                "NOT RUN: no 1-step Self-Forcing checkpoint exists on this "
                "machine or in the official gdhe17/Self-Forcing release; "
                "reported as unavailable rather than substituted."
            ),
        },
        "fvd_prdc_256": fvd,
        "latency_h200_21latents": latency,
        "noema_ablation": noema,
        "noema_followup_experiments": followup or None,
        "reviewer_runbook_experiments": reviewer or None,
        "protocol_notes": [
            "EMA export bug fixed before official runs: EMA_FSDP.full_state_dict "
            "dropped all FSDP flat-wrapped block parameters; all official "
            "checkpoints verified to contain complete generator_ema.",
            "All evaluations use --use_ema per README.",
            "opencv-python replaced by opencv-python-headless (no libGL on "
            "cluster nodes); all audited package versions unchanged.",
            "GRiT weights fetched from OpenGVLab/VBench_Used_Models HF mirror "
            "(Azure origin URL returns an error); identical filename and size.",
            "Long-video and curvature causal-intervention experiments were not "
            "run this round, per README.",
        ],
    }

    (out_dir / "final_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    # ---- CSV (flat, one metric per row) ----
    rows = []

    def add(section, condition, metric, value, n=None):
        if isinstance(value, float):
            value = round(value, 6)
        rows.append(
            {"section": section, "condition": condition, "metric": metric,
             "value": value, "sample_size": n, "seed": 0}
        )

    for name, block in vbench_final.items():
        for d, s in sorted(block["per_dimension"].items()):
            add("vbench_final", name, d, s, block["num_videos"])
        if block["official"]:
            for k, v in block["official"].items():
                add("vbench_final", name, f"official_{k}", v, block["num_videos"])
    for name, block in vbench_gan.items():
        for d, s in sorted(block["per_dimension"].items()):
            add("gan_ablation_vbench7", name, d, s, block["num_videos"])
    for d, s in sorted(gan_delta.items()):
        add("gan_ablation_vbench7", "delta_main_minus_dmd", d, s, 1630)
    for step, scores in sorted(stability.items()):
        for d, s in sorted(scores.items()):
            add("stability_vbench5", f"main_ffe_step{step}", d, s, 1165)
    for name, block in diversity_out.items():
        add("diversity_lpips", name, "mean_pairwise_lpips",
            block["mean_pairwise_lpips"], 400)
    add("diversity_lpips", "delta_main_minus_dmd", "mean_pairwise_lpips",
        payload["diversity_lpips"]["delta_main_minus_dmd"], 400)
    for name, block in fvd.items():
        add("fvd_prdc", name, "fvd", block["fvd"], 256)
        for k, v in (block.get("feature_distribution_metrics") or {}).items():
            if isinstance(v, (int, float)):
                add("fvd_prdc", name, k, v, 256)
    for name, block in latency.items():
        for metric, stats in block["summary_ms"].items():
            add("latency_ms", name, f"{metric}_mean", stats["mean"],
                block.get("trials"))
            if "std" in stats:
                add("latency_ms", name, f"{metric}_std", stats["std"],
                    block.get("trials"))
    if noema:
        for name, block in noema["vbench_final"].items():
            for d, s in sorted(block["per_dimension"].items()):
                add("noema_vbench_final", name, d, s, block["num_videos"])
            if block["official"]:
                for k, v in block["official"].items():
                    add("noema_vbench_final", name, f"official_{k}", v,
                        block["num_videos"])
        for name, scores in noema["gan_ablation_vbench7_step200_paired"]["conditions"].items():
            for d, s in sorted(scores.items()):
                add("noema_gan_ablation_vbench7", name, d, s, 1630)
        for d, s in sorted(
            noema["gan_ablation_vbench7_step200_paired"]["delta_main_minus_dmd"].items()
        ):
            add("noema_gan_ablation_vbench7", "delta_main_minus_dmd", d, s, 1630)
        for step, scores in sorted(noema["stability_vbench5_steps"].items()):
            for d, s in sorted(scores.items()):
                add("noema_stability_vbench5", f"main_ffe_step{step}", d, s, 1165)
        for name, block in noema["diversity_lpips"].items():
            add("noema_diversity_lpips", name, "mean_pairwise_lpips",
                block["mean_pairwise_lpips"], 400)
        for name, block in noema["fvd_prdc_256"].items():
            add("noema_fvd_prdc", name, "fvd", block["fvd"], 256)
            for k, v in (block.get("feature_distribution_metrics") or {}).items():
                if isinstance(v, (int, float)):
                    add("noema_fvd_prdc", name, k, v, 256)
        for name, block in noema["latency_h200_21latents"].items():
            for metric, stats in block["summary_ms"].items():
                add("noema_latency_ms", name, f"{metric}_mean", stats["mean"],
                    block.get("trials"))
    for exp_name, exp in (followup or {}).items():
        if not isinstance(exp, dict):
            continue
        for did, block in (exp.get("difference_in_differences") or {}).items():
            for k, v in (block.get("normalized_aggregates") or {}).items():
                add("noema_followup_" + exp_name, did, k, v, 944)
        for run, aggs in (exp.get("runs") or {}).items():
            if isinstance(aggs, dict) and "normalized_aggregates" in aggs:
                aggs = aggs["normalized_aggregates"]
            for k, v in (aggs or {}).items():
                if isinstance(v, (int, float)):
                    add("noema_followup_" + exp_name, run, k, v, 944)
        for comp, block in (exp.get("comparisons") or {}).items():
            aggs = block.get("normalized_aggregates") if isinstance(block, dict) and "normalized_aggregates" in block else block
            for k, v in (aggs or {}).items():
                if isinstance(v, (int, float)):
                    add("noema_followup_" + exp_name, comp, k, v, 944)
    for exp_name, exp in (reviewer or {}).items():
        if not isinstance(exp, dict):
            continue
        for run, block in (exp.get("runs") or {}).items():
            aggs = block.get("normalized_aggregates") if isinstance(block, dict) else None
            for k, v in (aggs or {}).items():
                if isinstance(v, (int, float)):
                    add("reviewer_" + exp_name, run, k, v, 944)
        for comp, block in (exp.get("comparisons") or {}).items():
            aggs = block.get("normalized_aggregates") if isinstance(block, dict) and "normalized_aggregates" in block else block
            for k, v in (aggs or {}).items():
                if isinstance(v, (int, float)):
                    add("reviewer_" + exp_name, comp, k, v, 944)
        for did, block in (exp.get("difference_in_differences") or {}).items():
            for k, v in (block.get("normalized_aggregates") or {}).items():
                if isinstance(v, (int, float)):
                    add("reviewer_" + exp_name, did, k, v, 944)
    for name, block in training.items():
        add("training", name, "final_step", block["final_step"])
        add("training", name, "wallclock_hours", block["wallclock_hours_first_to_done"])
        add("training", name, "mean_seconds_per_logged_iteration",
            block["mean_seconds_per_logged_iteration"])

    with open(out_dir / "final_metrics.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["section", "condition", "metric", "value",
                            "sample_size", "seed"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_dir/'final_metrics.json'} and final_metrics.csv "
          f"({len(rows)} csv rows). RUN_REPORT.md must be written separately.")


if __name__ == "__main__":
    main()
