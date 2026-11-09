"""Summarize only step 200: frame/chunk paired at training seeds 0 and 1."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from experiments.iclr.protocol import (RUN_SPECS, LEGACY_PROFILE, WINDOW_PROFILE,
    SEED2026_PROFILE, PROFILE_RUN_SPECS, atomic_json)
from experiments.rebuttal.consolidate_results import load_vbench_results, official_totals

EXPECTED_CELLS = {
    (name, "framewise" if block == 1 else "chunkwise", seed)
    for name, block, seed in RUN_SPECS
}
EXPECTED_SEEDS = {
    method: {seed for _, m, seed in EXPECTED_CELLS if m == method}
    for method in ("framewise", "chunkwise")
}


def collect_rows(root, *, profile=LEGACY_PROFILE):
    expected_cells = {(name, "framewise" if block == 1 else "chunkwise", seed)
                      for name, block, seed in PROFILE_RUN_SPECS[profile]}
    rows = []
    protocol = None
    for path in sorted(Path(root).rglob("iclr_cell.json")):
        cell = json.loads(path.read_text())
        if cell["step"] != 200 or cell.get("reference_step") != 200 or cell["milestones"] != [200]:
            raise ValueError("endpoint summary requires only explicitly selected step-200 checkpoints; use a separate evaluation directory")
        if (cell.get("mode") != "formal" or cell.get("schedule") != "main_ffe"
                or cell.get("profile") != profile
                or cell.get("stop_condition") != "fixed_steps"
                or cell.get("max_steps") != 200
                or cell.get("conditioning") != "pinned_qwen_rewrite"
                or cell.get("samples_per_prompt") != 5
                or cell.get("use_ema") is not True):
            raise ValueError(f"matched summaries require main-aligned EMA/main_ffe/Qwen cells: {path}")
        if cell["r1_weight"] != 0 or cell["r2_weight"] != 0:
            raise ValueError("matched summaries require R1 and R2 both off")
        if (cell["experiment_id"], cell["method"], cell["seed"]) not in expected_cells:
            raise ValueError(f"unexpected run in selected profile summary: {path}")
        if profile == WINDOW_PROFILE and (cell.get("gradient_num_blocks") != 0
                or cell.get("gradient_num_frames") != 18 or cell.get("effective_gradient_frames") != 18
                or cell.get("generator_updates") != 40 or cell.get("critic_updates") != 200):
            raise ValueError("window-18 summary requires matching actual gradient frames and update counts")
        if profile == SEED2026_PROFILE and (cell.get("gradient_num_blocks") != 17
                or cell.get("gradient_num_frames") != 0
                or cell.get("effective_gradient_frames") != (17 if cell["method"] == "framewise" else 21)
                or cell.get("generator_updates") != 40 or cell.get("critic_updates") != 200):
            raise ValueError("seed-2026 summary requires the original gradient windows and update counts")
        pairing = (cell["manifest_sha256"], cell["prompt_sha256"], cell["full_info_sha256"], cell["rewrite_sha256"],
                   cell["samples_per_prompt"], cell["max_steps"])
        if protocol is None:
            protocol = pairing
        if protocol != pairing:
            raise ValueError("endpoints do not share prompts/noise seeds, sample count or step budget")
        scores = load_vbench_results(path.parent / "vbench" / f"{cell['name']}_eval_results.json")
        totals = official_totals(scores)
        if totals is None:
            raise ValueError("all 16 dimensions are required for matched summary totals")
        scores.update(totals)
        for metric, score in scores.items():
            rows.append({"experiment_id": cell["experiment_id"], "method": cell["method"],
                "regularized": cell["r1_weight"] > 0, "seed": cell["seed"], "step": cell["step"],
                "training_seconds": cell["training_seconds"], "max_steps": cell["max_steps"],
                "generator_updates": cell["generator_updates"], "critic_updates": cell["critic_updates"],
                "metric": metric, "score": score})
            rows[-1]["reference_step"] = cell.get("reference_step")
    if not rows:
        raise ValueError("no completed step-200 evaluation cells")
    return rows


def group_by_step(rows):
    if {(r["method"], r["regularized"]) for r in rows} != {
            (m, False) for m in EXPECTED_SEEDS}:
        raise ValueError("formal summary requires framewise and chunkwise with regularization off")
    groups = {}
    for row in rows:
        key = (row["method"], row["regularized"], row["step"], row["metric"])
        seed_rows = groups.setdefault(key, {})
        if row["seed"] in seed_rows:
            raise ValueError(f"duplicate training seed for step {key}")
        seed_rows[row["seed"]] = row
    milestones = {step for _, _, step, _ in groups}
    if milestones != {200}:
        raise ValueError("formal summary requires only the final step 200")
    metrics = {row["metric"] for row in rows}
    expected = {(m, False, step, metric) for m in EXPECTED_SEEDS
                for step in milestones for metric in metrics}
    if set(groups) != expected:
        raise ValueError("arms have incomplete or different step/metric coverage")
    summaries = []
    for (method, regularized, step, metric), seed_rows in sorted(groups.items()):
        if set(seed_rows) != EXPECTED_SEEDS[method]:
            raise ValueError(f"{method} steps require training seeds {sorted(EXPECTED_SEEDS[method])}")
        values = list(seed_rows.values())
        summaries.append({"method": method, "regularized": regularized,
            "step": step, "metric": metric, "num_training_seeds": len(values),
            "mean": statistics.mean(row["score"] for row in values),
            "sample_std": statistics.stdev(row["score"] for row in values),
            "actual_training_seconds_mean": statistics.mean(row["training_seconds"] for row in values),
            "actual_training_seconds_min": min(row["training_seconds"] for row in values),
            "actual_training_seconds_max": max(row["training_seconds"] for row in values)})
    return summaries


def same_seed_comparison(rows):
    """Compare frame/chunk at identical steps within each training seed."""
    group_by_step(rows)
    pairs = {}
    for row in rows:
        pairs.setdefault((row["seed"], row["step"], row["metric"]), {})[row["method"]] = row
    comparisons = []
    for (seed, step, metric), pair in sorted(pairs.items()):
        frame, chunk = pair["framewise"], pair["chunkwise"]
        comparisons.append({"seed": seed, "step": step, "metric": metric,
            "frame_score": frame["score"], "chunk_score": chunk["score"],
            "frame_minus_chunk": frame["score"] - chunk["score"],
            "frame_training_seconds": frame["training_seconds"],
            "chunk_training_seconds": chunk["training_seconds"]})
    return comparisons


def paired_step_summary(comparisons):
    groups = {}
    for row in comparisons:
        key = (row["step"], row["metric"])
        pairs = groups.setdefault(key, {})
        if row["seed"] in pairs:
            raise ValueError("duplicate seed in paired summary")
        pairs[row["seed"]] = row
    summaries = []
    for (step, metric), pairs in sorted(groups.items()):
        if set(pairs) != EXPECTED_SEEDS["framewise"]:
            raise ValueError("paired summary requires both training seeds 0,1")
        values = [row["frame_minus_chunk"] for row in pairs.values()]
        summaries.append({"step": step, "metric": metric,
            "num_training_pairs": len(values), "mean_frame_minus_chunk": statistics.mean(values),
            "sample_std": statistics.stdev(values)})
    return summaries


def reference_step_summary(rows, step=200):
    if any(row["step"] != step or row["reference_step"] != step for row in rows):
        raise ValueError("reference table requires only explicitly selected step-200 checkpoints")
    groups = {}
    for row in rows:
        key = (row["method"], row["metric"])
        seeds = groups.setdefault(key, {})
        if row["seed"] in seeds:
            raise ValueError("duplicate reference seed")
        seeds[row["seed"]] = row
    metrics = {row["metric"] for row in rows}
    if set(groups) != {(method, metric) for method in EXPECTED_SEEDS for metric in metrics}:
        raise ValueError("reference table requires both granularities")
    result = []
    for (method, metric), seeds in sorted(groups.items()):
        if set(seeds) != EXPECTED_SEEDS[method]:
            raise ValueError("reference table requires seeds 0,1 for each method")
        values = [seeds[seed]["score"] for seed in (0, 1)]
        result.append({"method": method, "step": step, "metric": metric, "num_training_seeds": 2,
            "seed0": values[0], "seed1": values[1], "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "training_seconds_seed0": seeds[0]["training_seconds"],
            "training_seconds_seed1": seeds[1]["training_seconds"]})
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, help="root of step-200 exports, NOT CD or intermediate cells")
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/summary")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reference-step", type=int, choices=(200,), default=200)
    args = parser.parse_args()
    if args.dry_run:
        print("DRY RUN: step200 only; points.csv, step200_seed_summary.csv, same_seed_comparison.csv, paired_step_summary.csv; frame/chunk seeds 0,1; per-method and paired-difference sample SD; no VBench curves")
        return
    rows = collect_rows(args.evaluations)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summary = reference_step_summary(rows, args.reference_step)
    comparisons = same_seed_comparison(rows)
    write_csv(root / "points.csv", rows)
    write_csv(root / "step200_seed_summary.csv", summary)
    write_csv(root / "same_seed_comparison.csv", comparisons)
    write_csv(root / "paired_step_summary.csv", paired_step_summary(comparisons))
    atomic_json(root / "protocol.json", {"aggregation_unit": "training seed",
        "evaluation_scope": "final_checkpoint_only", "step": args.reference_step,
        "weight_source": "generator_ema", "schedule": "main_ffe", "conditioning": "pinned_qwen_rewrite",
        "samples_per_prompt": 5, "historical_score_reproduction_guaranteed": False,
        "summary": "identical final training step 200; actual elapsed costs reported separately",
        "primary_comparison": "framewise minus chunkwise within seeds 0 and 1 at step 200; summarize paired differences",
        "training_seeds": {method: sorted(seeds) for method, seeds in EXPECTED_SEEDS.items()},
        "uncertainty": "n=2 training seeds per method; sample SD of scores and of paired differences, not generation samples",
        "limitation": "endpoint-only two-seed repeatability check; no quality learning curves; equal iterations do not imply matched wall-clock compute; no statistical significance claim"})
    print(f"Wrote {root}")


if __name__ == "__main__":
    main()
