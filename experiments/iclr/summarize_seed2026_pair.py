"""Report the additional seed-2026 pair with the original seeds-0/1 recipe."""
import argparse
import json
import math
from pathlib import Path

from experiments.iclr.protocol import SEED2026_PROFILE, SEED2026_RUN_SPECS, atomic_json
from experiments.iclr.summarize import collect_rows, write_csv
from experiments.iclr.summarize_window_pair import audit_exports


def paired_rows(rows):
    expected = {(name, "framewise" if block == 1 else "chunkwise", seed)
                for name, block, seed in SEED2026_RUN_SPECS}
    groups = {}
    for row in rows:
        if ((row["experiment_id"], row["method"], row["seed"]) not in expected
                or row["step"] != 200 or row["reference_step"] != 200 or row["regularized"]
                or not math.isfinite(row["score"])):
            raise ValueError("expected only the two formal original-window/seed-2026 endpoints")
        pair = groups.setdefault(row["metric"], {})
        if row["method"] in pair:
            raise ValueError("duplicate endpoint/metric in seed-2026 pair")
        pair[row["method"]] = row
    if not groups or any(set(pair) != {"framewise", "chunkwise"} for pair in groups.values()):
        raise ValueError("both methods must have identical, complete metric coverage")
    result = []
    for metric, pair in sorted(groups.items()):
        frame, chunk = pair["framewise"], pair["chunkwise"]
        result.append({"seed": 2026, "step": 200, "metric": metric,
            "frame_score": frame["score"], "chunk_score": chunk["score"],
            "frame_minus_chunk": frame["score"] - chunk["score"],
            "frame_training_seconds": frame["training_seconds"],
            "chunk_training_seconds": chunk["training_seconds"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("DRY RUN: 2 runs, seed=2026, original windows 17/21, step=200, EMA/main_ffe/Qwen 944x5")
        return
    rows = collect_rows(args.evaluations, profile=SEED2026_PROFILE)
    audit_exports(args.evaluations)
    comparisons = paired_rows(rows)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "points.csv", rows)
    write_csv(root / "paired_comparison.csv", comparisons)
    atomic_json(root / "protocol.json", {
        "profile": SEED2026_PROFILE, "runs": [name for name, _, _ in SEED2026_RUN_SPECS],
        "training_seed": 2026, "num_training_pairs": 1, "gradient_num_blocks": 17,
        "effective_gradient_frames": {"framewise": 17, "chunkwise": 21},
        "step": 200, "weight_source": "generator_ema", "schedule": "main_ffe",
        "conditioning": "pinned_qwen_rewrite", "samples_per_prompt": 5,
        "evaluation_scope": "final_checkpoint_only", "score_scale": "0-1",
        "comparison": "additional training seed for the original seeds-0/1 recipe",
        "uncertainty": "one additional pair; no standalone seed variance estimate",
        "limitation": "original unequal gradient windows retained; FFE enhances the first native block (1 vs 3 latent frames)"})
    print(json.dumps({"output": str(root), "training_pairs": 1, "metrics": len(comparisons)}))


if __name__ == "__main__":
    main()
