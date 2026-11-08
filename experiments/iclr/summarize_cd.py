"""Paired original/rectified CD score table; no claims about training-seed variance."""
import argparse
import json
from pathlib import Path

from experiments.iclr.protocol import atomic_json
from experiments.iclr.summarize import write_csv
from experiments.rebuttal.consolidate_results import load_vbench_results, official_totals


def summarize(root):
    records = []
    by_cell = {}
    pairing = None
    first = Path(root) / "cd_original_all1" / "iclr_cell.json"
    first_cell = json.loads(first.read_text())
    local = (first_cell.get("cd_pair_controls") or {}).get("iclr_cd_profile") == "local_progressive_cd_all1_v1"
    schedules = ("all1",) if local else ("all1", "all4")
    if local and {p.name for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("cd_")} != {
            "cd_original_all1", "cd_rectified_all1"}:
        raise ValueError("local progressive CD summaries require exactly two all1 cells; no four-step outputs")
    for arm in ("original", "rectified"):
        for schedule in schedules:
            name = f"cd_{arm}_{schedule}"
            directory = Path(root) / name
            cell = json.loads((directory / "iclr_cell.json").read_text())
            if (cell["name"] != name or cell["schedule"] != schedule
                    or cell["method"] != "framewise" or cell["use_ema"] is not False
                    or cell["samples_per_prompt"] != 1 or cell["num_prompts"] != 944):
                raise ValueError(f"invalid paired CD protocol: {name}")
            identity = tuple(cell[k] for k in ("manifest_sha256", "prompt_sha256", "full_info_sha256"))
            identity += (json.dumps(cell.get("cd_pair_controls"), sort_keys=True), cell.get("cd_checkpoint_step"))
            if pairing is None:
                pairing = identity
            if identity != pairing:
                raise ValueError("CD cells use different training controls, prompts, seeds or VBench metadata")
            scores = load_vbench_results(directory / "vbench" / f"{name}_eval_results.json")
            totals = official_totals(scores)
            if totals is None:
                raise ValueError("all 16 dimensions are required")
            scores.update(totals)
            by_cell[arm, schedule] = scores
            records.append({"arm": arm, "schedule": schedule, "checkpoint": cell["checkpoint"],
                "cd_profile": (cell.get("cd_pair_controls") or {}).get("iclr_cd_profile", "legacy_unspecified"), **scores})
    for arm in ("original", "rectified"):
        paths = {r["checkpoint"] for r in records if r["arm"] == arm}
        if len(paths) != 1:
            raise ValueError(f"{arm}: all1/all4 must use the same checkpoint")
    deltas = []
    for schedule in schedules:
        for metric, value in by_cell["rectified", schedule].items():
            deltas.append({"schedule": schedule, "metric": metric,
                           "rectified_minus_original": value - by_cell["original", schedule][metric]})
    return records, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, help="one step/seed's four CD evaluation cells")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("DRY RUN: four-cell CD table and paired aggregate gains; no GPU access")
        return
    records, deltas = summarize(args.evaluations)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "cd_scores.csv", records)
    write_csv(output / "cd_gains.csv", deltas)
    comparison = {}
    if {row["cd_profile"] for row in records} == {"local_progressive_cd_all1_v1"}:
        comparison = {"evaluation_schedule": "all1", "evaluation_cell_count": 2,
                      "treatment": "high-noise state local rectification, alpha=min(step/100,1)",
                      "claim": "new experimental treatment; improvement is not guaranteed"}
    if {row["cd_profile"] for row in records} == {"original_adjacent_cd_9f_v1"}:
        from experiments.iclr.cd_original_protocol import historical_comparison, REFERENCE_SHA256
        write_csv(output / "historical_comparison.csv", historical_comparison(records))
        comparison = {"historical_comparison": "historical_comparison.csv", "training_recipe": "original 9-frame pure CD",
                      "reference_config_sha256": REFERENCE_SHA256,
                      "historical_initialization_match": "unverified", "historical_score_reproduction_guaranteed": False}
    atomic_json(output / "summary.json", {"scores": records, "gains": deltas,
        **comparison,
        "primary_schedule": "all1", "ffe": False, "score_scale": "normalized totals in 0-1 units; multiply by 100 for paper tables",
        "uncertainty": "aggregate point estimates only; generation samples are not training-seed repeats",
        "interpretation": "rectification gains do not alone establish a cause of one-step failure"})
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
