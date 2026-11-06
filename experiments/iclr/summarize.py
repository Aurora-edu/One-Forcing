"""Summarize 4 fixed-200-step runs: frame/chunk paired at training seeds 0 and 1."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from experiments.iclr.protocol import RUN_SPECS, atomic_json
from experiments.rebuttal.consolidate_results import load_vbench_results, official_totals

EXPECTED_CELLS = {
    (name, "framewise" if block == 1 else "chunkwise", seed)
    for name, block, seed in RUN_SPECS
}
EXPECTED_SEEDS = {
    method: {seed for _, m, seed in EXPECTED_CELLS if m == method}
    for method in ("framewise", "chunkwise")
}


def collect_rows(root):
    rows = []
    protocol = None
    for path in sorted(Path(root).rglob("iclr_cell.json")):
        cell = json.loads(path.read_text())
        initial = cell.get("initial_raw_reference") is True and cell["step"] == 0 and cell["milestones"] == [0]
        if (cell.get("mode") != "formal" or cell.get("schedule") != "main_ffe"
                or cell.get("profile") != "main_block_ablation_200_v2"
                or cell.get("stop_condition") != "fixed_steps"
                or cell.get("max_steps") != 200
                or cell.get("conditioning") != "pinned_qwen_rewrite"
                or cell.get("samples_per_prompt") != 5
                or cell.get("use_ema") is not (not initial)):
            raise ValueError(f"matched summaries require main-aligned EMA/main_ffe/Qwen cells (raw initial reference only at step 0): {path}")
        if cell["r1_weight"] != 0 or cell["r2_weight"] != 0:
            raise ValueError("matched summaries require R1 and R2 both off")
        if (cell["experiment_id"], cell["method"], cell["seed"]) not in EXPECTED_CELLS:
            raise ValueError(f"unexpected run in four-run summary: {path}")
        if cell["step"] not in (0, 50, 100, 150, 200) or cell["milestones"] != [cell["step"]]:
            raise ValueError("formal evaluation requires saved steps 0,50,100,150,200")
        pairing = (cell["manifest_sha256"], cell["prompt_sha256"], cell["full_info_sha256"], cell["rewrite_sha256"],
                   cell["samples_per_prompt"], cell["max_steps"])
        if protocol is None:
            protocol = pairing
        if protocol != pairing:
            raise ValueError("curves do not share prompts/noise seeds, sample count or step budget")
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
        raise ValueError("no completed curve evaluation cells")
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
    if milestones != {0, 50, 100, 150, 200}:
        raise ValueError("formal summary requires initial and final steps plus 50,100,150")
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
    parser.add_argument("--evaluations", required=True, help="root of curve exports, NOT CD cells")
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/summary")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reference-step", type=int, choices=(200,))
    args = parser.parse_args()
    if args.dry_run:
        print("DRY RUN: points.csv, matched_step_summary.csv, endpoints.csv, same_seed_comparison.csv, paired_step_summary.csv, curves.pdf/png; frame/chunk seeds 0,1; per-method and paired-difference sample SD")
        return
    rows = collect_rows(args.evaluations)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.reference_step:
        summary = reference_step_summary(rows, args.reference_step)
        write_csv(root / "step200_seed_summary.csv", summary)
        atomic_json(root / "protocol.json", {"axis": "training iterations", "step": args.reference_step,
            "weight_source": "generator_ema", "schedule": "main_ffe", "conditioning": "pinned_qwen_rewrite",
            "samples_per_prompt": 5, "historical_score_reproduction_guaranteed": False})
        return
    if any(row["reference_step"] is not None for row in rows):
        raise ValueError("do not mix the step-200 reference table with fixed-step curves")
    summary = group_by_step(rows)
    write_csv(root / "points.csv", rows)
    write_csv(root / "matched_step_summary.csv", summary)
    write_csv(root / "endpoints.csv", [row for row in summary if row["step"] == 200])
    write_csv(root / "step200_seed_summary.csv", reference_step_summary(
        [{**row, "reference_step": 200} for row in rows if row["step"] == 200]))
    comparisons = same_seed_comparison(rows)
    write_csv(root / "same_seed_comparison.csv", comparisons)
    write_csv(root / "paired_step_summary.csv", paired_step_summary(comparisons))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["total_score", "quality_score", "semantic_score", "dynamic_degree", "imaging_quality", "subject_consistency"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    arm_keys = sorted({(row["method"], row["regularized"]) for row in rows})
    for ax, metric in zip(axes.flat, metrics):
        for color, (method, regularized) in enumerate(arm_keys):
            for seed in sorted(EXPECTED_SEEDS[method]):
                selected = sorted([row for row in rows if row["metric"] == metric
                    and row["method"] == method and row["regularized"] == regularized and row["seed"] == seed],
                    key=lambda row: row["step"])
                ax.plot([row["step"] for row in selected], [row["score"] for row in selected],
                    marker=".", color=f"C{color}", alpha=.65, linestyle=("-", "--", ":")[seed],
                    label=f"{method}/R{'on' if regularized else 'off'}/s{seed}")
        ax.set(title=metric, xlabel="Training iterations", ylabel="Score")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(root / "curves.pdf")
    fig.savefig(root / "curves.png", dpi=160)
    plt.close(fig)
    atomic_json(root / "protocol.json", {"aggregation_unit": "training seed",
        "summary": "identical training steps 0,50,100,150,200; actual elapsed costs reported separately",
        "plot": "individual seed curves against training iterations",
        "step_zero": "explicit raw initialization reference before EMA exists; all later points use EMA",
        "primary_comparison": "framewise minus chunkwise within seeds 0 and 1 at identical training steps; summarize paired differences",
        "training_seeds": {method: sorted(seeds) for method, seeds in EXPECTED_SEEDS.items()},
        "uncertainty": "n=2 training seeds per method; sample SD of scores and of paired differences, not generation samples",
        "limitation": "two-seed repeatability check; equal iterations do not imply matched wall-clock compute; no statistical significance claim"})
    print(f"Wrote {root}")


if __name__ == "__main__":
    main()
