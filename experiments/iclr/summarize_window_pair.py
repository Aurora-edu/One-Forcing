"""Report the seed-42/window-18 endpoint pair; one training seed, no variance estimate."""
import argparse
import json
import math
from pathlib import Path

from experiments.iclr.protocol import WINDOW_PROFILE, WINDOW_RUN_SPECS, atomic_json
from experiments.iclr.summarize import collect_rows, write_csv
from experiments.rebuttal.consolidate_results import load_vbench_results


def audit_exports(root):
    """Check actual export/scoring records rather than relying on cell labels."""
    for path in Path(root).rglob("iclr_cell.json"):
        cell = json.loads(path.read_text())
        videos = path.parent / "videos"
        intent = json.loads((videos / "export.intent.json").read_text())
        done = json.loads((videos / "export.done").read_text())
        expected = {"method": cell["method"], "schedule": "main_ffe", "use_ema": True,
            "checkpoint_path": str(Path(cell["checkpoint"]).resolve()),
            "config_path": str(Path(cell["config"]).resolve()),
            "manifest_sha256": cell["manifest_sha256"], "prompt_sha256": cell["prompt_sha256"],
            "extended_prompt_sha256": cell["rewrite_sha256"], "selected_manifest_records": 944 * 5,
            "num_output_frames": 21, "fps": 16}
        required = {"status": "complete", "num_videos": 944 * 5, "weight_source": "generator_ema",
            "use_ema": True, "checkpoint_path": expected["checkpoint_path"],
            "manifest_sha256": cell["manifest_sha256"], "latent_frames_per_video": 21,
            "rgb_frames_per_video": 81, "fps": 16}
        if (any(intent.get(k) != v for k, v in expected.items())
                or any(done.get(k) != v for k, v in required.items())):
            raise ValueError(f"actual EMA/FFE export differs from window-18 cell: {path}")
        vbench = path.parent / "vbench"
        scoring = json.loads((vbench / f"{cell['name']}_protocol.json").read_text())
        scores = load_vbench_results(vbench / f"{cell['name']}_eval_results.json")
        if (scoring.get("mode") != "vbench_standard" or scoring.get("samples_per_prompt") != 5
                or scoring.get("official_five_sample_protocol") is not True
                or len(scoring.get("dimensions", [])) != 16
                or {k.replace(" ", "_") for k in scoring["dimensions"]} != set(scores)):
            raise ValueError(f"incomplete or different VBench scoring protocol: {path}")


def paired_rows(rows):
    expected = {(name, "framewise" if block == 1 else "chunkwise", seed)
                for name, block, seed in WINDOW_RUN_SPECS}
    groups = {}
    for row in rows:
        if ((row["experiment_id"], row["method"], row["seed"]) not in expected
                or row["step"] != 200 or row["reference_step"] != 200 or row["regularized"]
                or not math.isfinite(row["score"])):
            raise ValueError("expected only the two formal window-18/seed-42 endpoints")
        methods = groups.setdefault(row["metric"], {})
        if row["method"] in methods:
            raise ValueError("duplicate endpoint/metric in window-18 pair")
        methods[row["method"]] = row
    if not groups or any(set(pair) != {"framewise", "chunkwise"} for pair in groups.values()):
        raise ValueError("both methods must have identical, complete metric coverage")
    result = []
    for metric, pair in sorted(groups.items()):
        frame, chunk = pair["framewise"], pair["chunkwise"]
        result.append({"seed": 42, "step": 200, "effective_gradient_frames": 18,
            "metric": metric, "frame_score": frame["score"], "chunk_score": chunk["score"],
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
        print("DRY RUN: 2 runs, seed=42, window=18, step=200, EMA/main_ffe/Qwen 944x5; no SD or extra evaluations")
        return
    rows = collect_rows(args.evaluations, profile=WINDOW_PROFILE)
    audit_exports(args.evaluations)
    comparisons = paired_rows(rows)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "points.csv", rows)
    write_csv(root / "paired_comparison.csv", comparisons)
    atomic_json(root / "protocol.json", {
        "profile": WINDOW_PROFILE, "runs": [name for name, _, _ in WINDOW_RUN_SPECS],
        "training_seed": 42, "num_training_pairs": 1, "effective_gradient_frames": 18,
        "gradient_window_position": "tail", "step": 200, "weight_source": "generator_ema",
        "schedule": "main_ffe", "conditioning": "pinned_qwen_rewrite", "samples_per_prompt": 5,
        "evaluation_scope": "final_checkpoint_only", "third_run": "cancelled by user",
        "uncertainty": "one training pair; no run-to-run variance or significance estimate",
        "limitation": "equal steps and gradient frames; actual costs reported separately; main_ffe enhances the first native block (1 vs 3 latent frames)"})
    print(json.dumps({"output": str(root), "training_pairs": 1, "metrics": len(comparisons)}))


if __name__ == "__main__":
    main()
