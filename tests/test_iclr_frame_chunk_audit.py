import csv
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch

from experiments.iclr.audit_frame_chunk import audit_run, prepare_visuals, score_contributions
from experiments.iclr.protocol import REPO, RUN_SPECS
from experiments.rebuttal.consolidate_results import NORMALIZE_DIC, official_totals


def test_metric_decomposition_exactly_recovers_total_difference():
    points = []
    expected = {}
    for seed in (0, 1):
        scores = {}
        for method in ("framewise", "chunkwise"):
            value = .4 + .03 * seed + (.02 if method == "framewise" else 0.)
            s = {k.replace(" ", "_"): v["Min"] + value * (v["Max"] - v["Min"])
                 for k, v in NORMALIZE_DIC.items()}
            s["dynamic_degree"] = .2 if method == "framewise" else .9
            scores[method] = official_totals(s)["total_score"]
            points.extend({"seed": seed, "method": method, "metric": k, "score": v} for k, v in s.items())
        expected[seed] = 100 * (scores["framewise"] - scores["chunkwise"])
    rows = score_contributions(points)
    for seed in (0, 1):
        assert sum(r["total_contribution_percentage_points"] for r in rows if r["seed"] == seed) == pytest.approx(expected[seed])


def config(name="frame_roff_s0", block=1):
    return {"experiment_id": name, "seed": 0, "num_frame_per_block": block, "max_steps": 200,
            "r1_weight": 0., "r2_weight": 0., "gradient_num_blocks": 17, "ema_weight": .99,
            "ema_start_step": 50, "gan_feature_layers": [21, 29], "gan_num_registers": 2,
            "gan_block_ffn_dim": 2048, "model_kwargs": {"timestep_shift": 5., "local_attn_size": -1, "sink_size": 2}}


def test_visual_plan_holds_attention_fixed_and_never_starts_jobs(tmp_path):
    for name, block, _ in RUN_SPECS:
        run = tmp_path / "runs" / name; run.mkdir(parents=True)
        OmegaConf.save(OmegaConf.create(config(name, block)), run / "resolved_config.yaml")
    out = tmp_path / "audit"; out.mkdir()
    script = Path(prepare_visuals(tmp_path, out, sys.executable))
    text = script.read_text()
    assert text.count("scripts/export_videos.py") == 16  # 4 runs x raw/EMA x all1/FFE; 4 clips each.
    assert text.count("--use_ema") == 8
    subprocess.run(["bash", "-n", str(script)], check=True)
    for name, block, _ in RUN_SPECS:
        all1 = OmegaConf.load(out / f"{name}_all1.yaml")
        ffe = OmegaConf.load(out / f"{name}_main_ffe.yaml")
        assert all1.model_kwargs == ffe.model_kwargs == config()["model_kwargs"]
        assert "first_frame_denoising_step_list" not in all1
        assert list(ffe.first_frame_denoising_step_list) == [1000, 750, 500, 250]
        assert all1.num_frame_per_block == ffe.num_frame_per_block == block
    assert not (out / "videos").exists()


def test_audit_detects_actual_checkpoint_step_and_export_method_mismatches(tmp_path):
    name = "frame_roff_s0"; run = tmp_path / "runs" / name
    checkpoint = run / "checkpoint_model_000200" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"step": 150, "generator": {"w": torch.ones(8)}, "generator_ema": {"w": torch.zeros(8)}}, checkpoint)
    cfg = config()
    OmegaConf.save(OmegaConf.create(cfg), run / "resolved_config.yaml")
    cell = tmp_path / "eval_step200" / name / "step000200_framewise"
    (cell / "videos").mkdir(parents=True)
    (cell / "videos/export.intent.json").write_text(json.dumps({"method": "chunkwise", "schedule": "main_ffe",
        "use_ema": True, "selected_manifest_records": 4720, "checkpoint_path": str(checkpoint)}))
    actual = {**cfg, "denoising_step_list": [1000], "first_frame_denoising_step_list": [1000, 750, 500, 250],
              "rollout_schedule": "fixed"}
    OmegaConf.save(OmegaConf.create(actual), cell / "videos/infer_config_framewise_main_ffe_shard0.yaml")
    for shard in range(8):
        (cell / f"videos/export.shard_{shard:02d}_of_08.done").write_text(json.dumps({
            "use_ema": True, "weight_source": "generator_ema", "num_total_videos": 4720,
            "num_videos": 590, "checkpoint_path": str(checkpoint), "latent_frames_per_video": 21, "fps": 16}))
    report = audit_run(tmp_path, name, 1, 0, check_weights=True)
    assert any("actual model.pt step" in error for error in report["errors"])
    assert any("export intent method" in error for error in report["errors"])
    assert report["sampled_raw_sha256"] != report["sampled_ema_sha256"]
    assert report["exported_records"] == 4720
    assert report["missing"]  # Missing original evidence never gets silently labelled PASS.


def test_cli_reports_missing_evidence_without_gpu_queries(tmp_path):
    out = tmp_path / "audit"
    subprocess.run([sys.executable, "-m", "experiments.iclr.audit_frame_chunk", "--root", str(tmp_path),
        "--output", str(out), "--check-weights"], cwd=REPO, check=True, capture_output=True)
    report = json.loads((out / "audit.json").read_text())
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 4 and all(r["missing"] for r in report["runs"])
