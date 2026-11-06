"""Summarize matched TIME curves, with actual time coordinates and 3-seed SD."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from experiments.iclr.protocol import atomic_json
from experiments.rebuttal.consolidate_results import load_vbench_results, official_totals


def collect_rows(root):
    rows = []
    protocol = None
    for path in sorted(Path(root).rglob("iclr_cell.json")):
        cell = json.loads(path.read_text())
        if cell.get("mode") != "formal" or cell.get("schedule") != "all1" or cell.get("use_ema") is not False:
            raise ValueError(f"matched summaries require formal/raw/all1 curve cells: {path}")
        if 1. in cell["fractions"] and cell["step"] < 10:
            raise ValueError("formal endpoint has fewer than 10 iterations; budget was too short")
        pairing = (cell["manifest_sha256"], cell["prompt_sha256"], cell["full_info_sha256"],
                   cell["samples_per_prompt"], cell["budget_seconds"])
        if protocol is None:
            protocol = pairing
        if protocol != pairing:
            raise ValueError("curves do not share prompts/noise seeds, sample count or time budget")
        scores = load_vbench_results(path.parent / "vbench" / f"{cell['name']}_eval_results.json")
        totals = official_totals(scores)
        if totals is None:
            raise ValueError("all 16 dimensions are required for matched summary totals")
        scores.update(totals)
        for metric, score in scores.items():
            rows.append({"experiment_id": cell["experiment_id"], "method": cell["method"],
                "regularized": cell["r1_weight"] > 0, "seed": cell["seed"], "step": cell["step"],
                "training_seconds": cell["training_seconds"], "budget_seconds": cell["budget_seconds"],
                "generator_updates": cell["generator_updates"], "critic_updates": cell["critic_updates"],
                "metric": metric, "score": score, "fractions": cell["fractions"]})
    if not rows:
        raise ValueError("no completed curve evaluation cells")
    return rows


def group_by_timepoint(rows):
    if {(r["method"], r["regularized"]) for r in rows} != {
            (m, r) for m in ("framewise", "chunkwise") for r in (False, True)}:
        raise ValueError("formal summary requires all four experimental arms")
    groups = {}
    for row in rows:
        for fraction in row["fractions"]:
            key = (row["method"], row["regularized"], fraction, row["metric"])
            seed_rows = groups.setdefault(key, {})
            if row["seed"] in seed_rows:
                raise ValueError(f"duplicate training seed for timepoint {key}")
            seed_rows[row["seed"]] = row
    milestones = {fraction for _, _, fraction, _ in groups}
    if not {0., 1.} <= milestones:
        raise ValueError("formal summary requires initial and final timepoints")
    metrics = {row["metric"] for row in rows}
    expected = {(m, r, f, metric) for m in ("framewise", "chunkwise")
                for r in (False, True) for f in milestones for metric in metrics}
    if set(groups) != expected:
        raise ValueError("arms have incomplete or different timepoint/metric coverage")
    summaries = []
    for (method, regularized, fraction, metric), seed_rows in sorted(groups.items()):
        if set(seed_rows) != {0, 1, 2}:
            raise ValueError("each matched timepoint requires exactly training seeds 0,1,2")
        values = list(seed_rows.values())
        target = values[0]["budget_seconds"] * fraction
        summaries.append({"method": method, "regularized": regularized,
            "fraction": fraction, "metric": metric, "num_training_seeds": 3,
            "mean": statistics.mean(row["score"] for row in values),
            "sample_std": statistics.stdev(row["score"] for row in values),
            "target_training_seconds": target,
            "actual_training_seconds_mean": statistics.mean(row["training_seconds"] for row in values),
            "actual_training_seconds_min": min(row["training_seconds"] for row in values),
            "actual_training_seconds_max": max(row["training_seconds"] for row in values),
            "step_min": min(row["step"] for row in values), "step_max": max(row["step"] for row in values)})
    return summaries


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
    args = parser.parse_args()
    if args.dry_run:
        print("DRY RUN: points.csv, matched_time_summary.csv, endpoints.csv, curves.pdf/png; actual training_seconds, 3 training seeds")
        return
    rows = collect_rows(args.evaluations)
    summary = group_by_timepoint(rows)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "points.csv", rows)
    write_csv(root / "matched_time_summary.csv", summary)
    write_csv(root / "endpoints.csv", [row for row in summary if row["fraction"] == 1.])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["total_score", "quality_score", "semantic_score", "dynamic_degree", "imaging_quality", "subject_consistency"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    arm_keys = sorted({(row["method"], row["regularized"]) for row in rows})
    for ax, metric in zip(axes.flat, metrics):
        for color, (method, regularized) in enumerate(arm_keys):
            for seed in (0,1,2):
                selected = sorted([row for row in rows if row["metric"] == metric
                    and row["method"] == method and row["regularized"] == regularized and row["seed"] == seed],
                    key=lambda row: row["training_seconds"])
                ax.plot([row["training_seconds"] / 3600 for row in selected], [row["score"] for row in selected],
                    marker=".", color=f"C{color}", alpha=.65, linestyle=("-", "--", ":")[seed],
                    label=f"{method}/R{'on' if regularized else 'off'}/s{seed}")
        ax.set(title=metric, xlabel="Actual synchronized training time (hours)", ylabel="Score")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(root / "curves.pdf")
    fig.savefig(root / "curves.png", dpi=160)
    plt.close(fig)
    atomic_json(root / "protocol.json", {"aggregation_unit": "training seed",
        "summary": "first iteration boundary reaching each target time; actual overshoot reported",
        "plot": "individual seed curves at actual time; no averaging or interpolation by training step",
        "step_zero": "same raw tensors, potentially different outputs due to attention granularity",
        "uncertainty": "sample SD across 3 seeds, not generation samples"})
    print(f"Wrote {root}")


if __name__ == "__main__":
    main()
