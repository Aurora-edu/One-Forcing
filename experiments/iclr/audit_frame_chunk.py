"""Read-only audit of existing frame/chunk runs; optional visual export plan."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

from omegaconf import OmegaConf

from experiments.iclr.protocol import REPO, RUN_SPECS, atomic_json
from experiments.rebuttal.consolidate_results import (
    DIM_WEIGHT, NORMALIZE_DIC, QUALITY_LIST, SEMANTIC_LIST, official_totals, load_vbench_results,
)
from utils.config import load_config


def score_contributions(points):
    """Decompose the existing official Total, without constructing a new metric."""
    groups = {}
    for row in points:
        groups.setdefault((row["method"], int(row["seed"])), {})[row["metric"]] = float(row["score"])
    output = []
    for seed in (0, 1):
        if any((method, seed) not in groups for method in ("framewise", "chunkwise")):
            continue
        frame, chunk = (groups[method, seed] for method in ("framewise", "chunkwise"))
        for metric, bounds in NORMALIZE_DIC.items():
            key = metric.replace(" ", "_")
            family = QUALITY_LIST if metric in QUALITY_LIST else SEMANTIC_LIST
            family_weight = .8 if metric in QUALITY_LIST else .2
            contribution = ((frame[key] - chunk[key]) / (bounds["Max"] - bounds["Min"])
                            * DIM_WEIGHT[metric] / sum(DIM_WEIGHT[m] for m in family) * family_weight)
            output.append({"seed": seed, "metric": key, "frame": frame[key], "chunk": chunk[key],
                           "total_contribution_percentage_points": 100 * contribution})
    return output


def sampled_weights(state):
    """Small deterministic CPU fingerprint; equal samples do not prove equality."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        flat = tensor.detach().reshape(-1)
        if flat.numel():
            indices = torch.linspace(0, flat.numel() - 1, min(16, flat.numel())).long()
            values = flat[indices].double()
            if not torch.isfinite(values).all():
                raise ValueError(f"nonfinite sampled checkpoint tensor: {name}")
            digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def audit_run(root, name, block, seed, *, check_weights=False):
    method = "framewise" if block == 1 else "chunkwise"
    run = root / "runs" / name
    checkpoint = run / "checkpoint_model_000200" / "model.pt"
    cell_root = root / "eval_step200" / name / f"step000200_{method}"
    report = {"run": name, "missing": [], "errors": [], "warnings": []}

    def read(path):
        if not path.is_file():
            report["missing"].append(str(path)); return None
        return json.loads(path.read_text())

    def expect(actual, expected, label):
        if actual != expected:
            report["errors"].append(f"{label}: expected {expected!r}, got {actual!r}")

    cfg_path = run / "resolved_config.yaml"
    cfg = OmegaConf.to_container(load_config(str(cfg_path)), resolve=True) if cfg_path.is_file() else None
    if cfg is None:
        report["missing"].append(str(cfg_path))
    else:
        for key, expected in {"experiment_id": name, "num_frame_per_block": block, "seed": seed,
            "max_steps": 200, "r1_weight": 0., "r2_weight": 0., "gradient_num_blocks": 17,
            "ema_weight": .99, "ema_start_step": 50, "gan_feature_layers": [21, 29],
            "gan_num_registers": 2, "gan_block_ffn_dim": 2048}.items():
            expect(cfg.get(key), expected, f"training config {key}")
        report["config"] = cfg
        report["effective_gradient_frames"] = min(21, cfg["gradient_num_blocks"] * block)
        report["first_block_enhanced_latent_frames"] = block
    complete = read(checkpoint.parent / "complete.json")
    if complete:
        expect(complete.get("schema_version"), 3, "checkpoint schema")
        for key, expected in {"step": 200, "generator_updates": 40, "critic_updates": 200}.items():
            expect(complete.get("clock", {}).get(key), expected, f"checkpoint clock {key}")
        if cfg:
            for key in ("experiment_id", "seed", "num_frame_per_block", "r1_weight", "r2_weight", "ema_weight"):
                expect(complete["config"].get(key), cfg.get(key), f"checkpoint config {key}")
        if checkpoint.exists():
            expect(checkpoint.stat().st_size, complete["files"].get("model.pt"), "checkpoint file size")
    report["initialization"] = read(run / "initialization.json")
    metrics = run / "metrics.jsonl"
    if metrics.is_file():
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        expect([r["step"] for r in rows], list(range(1, 201)), "actual training steps")
        expect(sum("generator_loss" in r for r in rows), 40, "actual generator updates")
        if cfg:
            ema_updates = sum("generator_loss" in r and r["step"] > cfg["ema_start_step"] for r in rows)
            report["ema_expected_from_logged_schedule"] = {
                "generator_updates_after_ema_start": ema_updates,
                "decay": cfg["ema_weight"],
                "initial_shadow_coefficient": cfg["ema_weight"] ** ema_updates,
                "note": "nominal parameter weighting; does not establish that the actual EMA state was updated correctly"}
        selected = {key: [float(r[key]) for r in rows if key in r] for key in
                    ("generator_grad_norm", "critic_grad_norm", "gan_fake_logit", "gan_real_logit")}
        finite = {key: [x for x in values if math.isfinite(x)] for key, values in selected.items()}
        report["training_ranges"] = {key: {"min": min(v), "max": max(v)} for key, v in finite.items() if v}
        if any(not math.isfinite(v) for values in selected.values() for v in values):
            report["errors"].append("nonfinite training logs")
    else:
        report["missing"].append(str(metrics))
    cell = read(cell_root / "iclr_cell.json")
    if cell:
        for key, expected in {"experiment_id": name, "seed": seed, "method": method, "step": 200,
            "use_ema": True, "schedule": "main_ffe", "samples_per_prompt": 5}.items():
            expect(cell.get(key), expected, f"evaluation label {key}")
        expect(Path(cell["checkpoint"]).resolve(), checkpoint.resolve(), "evaluation checkpoint path")
    intent = read(cell_root / "videos" / "export.intent.json")
    if intent:
        for key, expected in {"method": method, "schedule": "main_ffe", "use_ema": True,
                              "selected_manifest_records": 4720}.items():
            expect(intent.get(key), expected, f"export intent {key}")
        expect(Path(intent["checkpoint_path"]).resolve(), checkpoint.resolve(), "actual export checkpoint path")
        for path_key, hash_key in (("manifest_path", "manifest_sha256"), ("prompt_path", "prompt_sha256"),
                                   ("extended_prompt_path", "extended_prompt_sha256")):
            path = Path(intent[path_key]) if intent.get(path_key) else None
            if path and path.is_file():
                expect(hashlib.sha256(path.read_bytes()).hexdigest(), intent[hash_key], f"actual export {hash_key}")
        report["export_identity"] = {k: intent.get(k) for k in
            ("manifest_sha256", "prompt_sha256", "extended_prompt_sha256", "checkpoint_path")}
    exports = sorted((cell_root / "videos").glob("export.shard_*.done"))
    if not exports and (cell_root / "videos/export.done").is_file():
        exports = [cell_root / "videos/export.done"]
    if not exports:
        report["missing"].append(str(cell_root / "videos/export*.done"))
    else:
        counts = []
        for path in exports:
            exported = json.loads(path.read_text())
            for key, expected in {"use_ema": True, "weight_source": "generator_ema", "num_total_videos": 4720,
                                  "latent_frames_per_video": 21, "fps": 16}.items():
                expect(exported.get(key), expected, f"actual shard {path.name}:{key}")
            expect(Path(exported["checkpoint_path"]).resolve(), checkpoint.resolve(), f"actual shard {path.name}:checkpoint")
            counts.append(exported["num_videos"])
        expect(sum(counts), 4720, "actual shard exported record count")
        report["exported_records"] = sum(counts)
    mp4_count = sum(1 for _ in (cell_root / "videos").glob("*.mp4"))
    report["mp4_files_present"] = mp4_count
    if mp4_count != 4720:
        report["missing"].append(f"full evaluation MP4s: expected 4720, currently found {mp4_count}")
    infer_configs = sorted((cell_root / "videos").glob("infer_config_*_shard*.yaml"))
    if not infer_configs:
        report["missing"].append(str(cell_root / "videos" / "infer_config_*_shard*.yaml"))
    for path in infer_configs:
        actual = load_config(str(path))
        for key, expected in {"num_frame_per_block": block, "denoising_step_list": [1000],
            "first_frame_denoising_step_list": [1000, 750, 500, 250], "rollout_schedule": "fixed"}.items():
            expect(OmegaConf.to_container(actual[key]) if OmegaConf.is_config(actual.get(key)) else actual.get(key),
                   expected, f"actual inference {path.name}:{key}")
        if cfg:
            expect(OmegaConf.to_container(actual.model_kwargs, resolve=True), cfg["model_kwargs"],
                   f"actual inference {path.name}:model_kwargs")
    scores = cell_root / "vbench" / f"step000200_{method}_eval_results.json"
    if scores.is_file():
        report["scores"] = load_vbench_results(scores)
        totals = official_totals(report["scores"])
        if totals is None:
            report["errors"].append("incomplete VBench dimensions")
        else:
            report["scores"].update(totals)
    else:
        report["missing"].append(str(scores))
    if check_weights:
        if not checkpoint.is_file():
            report["missing"].append(str(checkpoint))
        else:
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            expect(payload.get("step"), 200, "actual model.pt step")
            if "generator_ema" not in payload:
                report["errors"].append("model.pt has no EMA weights")
            else:
                expect({k: tuple(v.shape) for k, v in payload["generator"].items()},
                       {k: tuple(v.shape) for k, v in payload["generator_ema"].items()}, "raw/EMA keys and shapes")
                report["sampled_raw_sha256"] = sampled_weights(payload["generator"])
                report["sampled_ema_sha256"] = sampled_weights(payload["generator_ema"])
                if report["sampled_raw_sha256"] == report["sampled_ema_sha256"]:
                    report["warnings"].append("raw/EMA samples identical; inspect the full weights and EMA update history")
            del payload
    return report


def prepare_visuals(root, output, python):
    """Plan 64 final-checkpoint clips; hold attention settings fixed across FFE."""
    prompts = output / "visual_prompts.txt"
    prompts.write_text("\n".join((REPO / "prompts/demos.txt").read_text().splitlines()[:4]) + "\n")
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(str(REPO))]
    for name, block, _ in RUN_SPECS:
        for schedule in ("all1", "main_ffe"):
            cfg = load_config(str(root / "runs" / name / "resolved_config.yaml"))
            cfg.num_frame_per_block = block
            cfg.rollout_schedule = "fixed"
            cfg.denoising_step_list = [1000]
            cfg.warp_denoising_step = True
            cfg.pop("first_frame_denoising_step_list", None)
            if schedule == "main_ffe":
                cfg.first_frame_denoising_step_list = [1000, 750, 500, 250]
            path = output / f"{name}_{schedule}.yaml"
            OmegaConf.save(cfg, path)
            for weight in ("raw", "ema"):
                command = [python, str(REPO / "scripts/export_videos.py"), "--config_path", str(path),
                    "--checkpoint_path", str(root / "runs" / name / "checkpoint_model_000200/model.pt"),
                    "--prompt_path", str(prompts), "--limit", "4", "--num_samples_per_prompt", "1", "--seed", "0",
                    "--output_folder", str(output / "videos" / name / f"{weight}_{schedule}")]
                if weight == "ema":
                    command.append("--use_ema")
                commands += ['fc_audit_pids="$(nvidia-smi -i "${FC_AUDIT_GPU:-0}" --query-compute-apps=pid --format=csv,noheader,nounits)"',
                             'if [[ -n "$fc_audit_pids" ]]; then echo "GPU busy; wait for existing jobs" >&2; exit 1; fi',
                             'CUDA_VISIBLE_DEVICES="${FC_AUDIT_GPU:-0}" ' + shlex.join(command)]
    path = output / "export_visuals.sh"
    path.write_text("\n".join(commands) + "\n")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="existing ICLR_ROOT, containing runs/ and eval_step200/")
    parser.add_argument("--output", required=True, help="new audit output directory")
    parser.add_argument("--check-weights", action="store_true", help="CPU mmap inspection and small raw/EMA tensor samples")
    parser.add_argument("--prepare-visuals", action="store_true", help="write a 64-clip inference plan; do not launch GPU jobs")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("choose a fresh audit output directory")
    reports = []
    for name, block, seed in RUN_SPECS:
        try:
            reports.append(audit_run(root, name, block, seed, check_weights=args.check_weights))
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            reports.append({"run": name, "missing": [], "warnings": [],
                            "errors": [f"cannot audit malformed/inconsistent artifacts: {error}"]})
    for seed in (0, 1):
        pair = [r for r in reports if r["run"].endswith(f"s{seed}")]
        for key in ("generator_checkpoint_sha256", "generator_tensor_sha256", "critic_tensor_sha256"):
            values = [(r.get("initialization") or {}).get(key) for r in pair]
            if all(values) and values[0] != values[1]:
                for r in pair:
                    r["errors"].append(f"same-seed initialization differs: {key}")
    for key in ("generator_checkpoint_sha256", "generator_tensor_sha256"):
        values = {(r.get("initialization") or {}).get(key) for r in reports} - {None}
        if len(values) > 1:
            for r in reports:
                r["errors"].append(f"generator initialization differs across seeds: {key}")
    ignored = {"num_frame_per_block", "seed", "experiment_id", "logdir", "resume_ckpt", "wandb_save_dir"}
    configs = [(r, {k: v for k, v in r["config"].items() if k not in ignored}) for r in reports if "config" in r]
    if configs:
        for report, config in configs[1:]:
            changed = [k for k in config.keys() | configs[0][1].keys() if config.get(k) != configs[0][1].get(k)]
            if changed:
                report["errors"].append(f"unexpected cross-run config changes: {sorted(changed)}")
    fingerprints = {}
    for report in reports:
        if "sampled_ema_sha256" in report:
            fingerprints.setdefault(report["sampled_ema_sha256"], []).append(report["run"])
    for names in fingerprints.values():
        if len(names) > 1:
            for report in reports:
                if report["run"] in names:
                    report["warnings"].append(f"sampled EMA weights repeated across {names}; verify full weights")
    identities = [tuple(r["export_identity"].get(k) for k in ("manifest_sha256", "prompt_sha256", "extended_prompt_sha256"))
                  for r in reports if "export_identity" in r]
    if len(set(identities)) > 1:
        for r in reports:
            r["errors"].append("evaluation prompt/rewrite/noise manifests differ across runs")
    points = [{"method": "framewise" if r["run"].startswith("frame") else "chunkwise",
               "seed": r["run"][-1], "metric": key, "score": value}
              for r in reports for key, value in r.get("scores", {}).items()]
    summary_path = root / "summary_step200" / "points.csv"
    if summary_path.is_file():
        summary_points = list(csv.DictReader(summary_path.open()))
        lookup = {(row["experiment_id"], row["metric"]): float(row["score"]) for row in summary_points}
        for report in reports:
            for metric, value in report.get("scores", {}).items():
                if not math.isclose(value, lookup.get((report["run"], metric), float("nan")), abs_tol=1e-7):
                    report["errors"].append(f"summary label/value differs from raw VBench JSON: {metric}")
        if not points:
            points = summary_points  # Still decompose archived scores, while reporting missing raw evidence.
    output.mkdir(parents=True)
    contributions = score_contributions(points)
    if contributions:
        with (output / "score_contributions.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contributions[0])); writer.writeheader(); writer.writerows(contributions)
    result = {"status": "issues" if any(r["errors"] for r in reports) else
                        "incomplete" if any(r["missing"] for r in reports) else "metadata_checked",
              "runs": reports, "weight_inspection_requested": args.check_weights,
              "num_weight_files_inspected": sum("sampled_raw_sha256" in r for r in reports),
              "limitation": "metadata and sampled tensors cannot prove full weight identity or certify video dynamics; no new training or VBench run"}
    if args.prepare_visuals:
        try:
            result["visual_plan"] = prepare_visuals(root, output, sys.executable)
        except (FileNotFoundError, ValueError) as error:
            result["visual_plan_error"] = str(error)
    atomic_json(output / "audit.json", result)
    print(json.dumps({"status": result["status"], "report": str(output / "audit.json"),
                      "errors": sum(len(r["errors"]) for r in reports), "missing": sum(len(r["missing"]) for r in reports)}))


if __name__ == "__main__":
    main()
