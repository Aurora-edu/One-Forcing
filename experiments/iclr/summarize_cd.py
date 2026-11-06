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
    for arm in ("original", "rectified"):
        for schedule in ("all1", "all4"):
            name = f"cd_{arm}_{schedule}"
            directory = Path(root) / name
            cell = json.loads((directory / "iclr_cell.json").read_text())
            if (cell["name"] != name or cell["schedule"] != schedule
                    or cell["method"] != "framewise" or cell["use_ema"] is not False
                    or cell["samples_per_prompt"] != 1 or cell["num_prompts"] != 944):
                raise ValueError(f"invalid paired CD protocol: {name}")
            identity = tuple(cell[k] for k in ("manifest_sha256", "prompt_sha256", "full_info_sha256"))
            if pairing is None:
                pairing = identity
            if identity != pairing:
                raise ValueError("CD cells use different prompts, generation seeds or VBench metadata")
            scores = load_vbench_results(directory / "vbench" / f"{name}_eval_results.json")
            totals = official_totals(scores)
            if totals is None:
                raise ValueError("all 16 dimensions are required")
            scores.update(totals)
            by_cell[arm, schedule] = scores
            records.append({"arm": arm, "schedule": schedule, "checkpoint": cell["checkpoint"], **scores})
    for arm in ("original", "rectified"):
        paths = {r["checkpoint"] for r in records if r["arm"] == arm}
        if len(paths) != 1:
            raise ValueError(f"{arm}: all1/all4 must use the same checkpoint")
    deltas = []
    for schedule in ("all1", "all4"):
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
    atomic_json(output / "summary.json", {"scores": records, "gains": deltas,
        "primary_schedule": "all1", "ffe": False, "score_scale": "normalized totals in 0-1 units; multiply by 100 for paper tables",
        "uncertainty": "aggregate point estimates only; generation samples are not training-seed repeats",
        "interpretation": "rectification gains do not alone establish a cause of one-step failure"})
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
