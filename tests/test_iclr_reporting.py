import csv
import json
import subprocess
import sys

import pytest
import torch

from experiments.iclr.audit_initialization import audit
from experiments.iclr.collect_teacher import make_schedule, euler_update
from experiments.iclr.evaluate import record_cell, validate_manifest
from experiments.iclr.protocol import REPO, RUN_SPECS
from experiments.iclr.summarize import group_by_step, collect_rows, same_seed_comparison, paired_step_summary, reference_step_summary
from experiments.iclr.summarize_cd import summarize as summarize_cd
from experiments.rebuttal.consolidate_results import NORMALIZE_DIC, official_totals


@pytest.mark.parametrize("steps", [50, 200])
def test_dense_teacher_schedule_and_constant_velocity_integrate_to_endpoint(steps):
    scheduler, times = make_schedule(steps, 8.)
    assert len(times) == steps + 1 and times[0] == 1000 and times[-1] == 0
    assert all(a > b for a, b in zip(times, times[1:]))
    initial = torch.arange(12).reshape(1, 3, 1, 2, 2).float()
    x = initial.clone()
    velocity = torch.full_like(x, 2.)
    for t in times[:-1]:
        x = euler_update(scheduler, x, velocity, torch.full((1, 3), t))
    torch.testing.assert_close(x, initial - velocity, atol=1e-5, rtol=1e-5)


def test_evaluation_provenance_cannot_be_overwritten(tmp_path):
    p = tmp_path / "cell" / "iclr_cell.json"
    record_cell(p, {"seed": 1})
    record_cell(p, {"seed": 1})
    with pytest.raises(ValueError, match="another cell"):
        record_cell(p, {"seed": 2})
    assert json.loads(p.read_text()) == {"seed": 1}
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "old_results.json").write_text("{}")
    with pytest.raises(ValueError, match="no ICLR provenance"):
        record_cell(unknown / "iclr_cell.json", {})


def test_original_manifest_rejects_rewrites_and_extra_samples(tmp_path):
    prompts = tmp_path / "prompts.txt"
    manifest = tmp_path / "manifest.jsonl"
    prompts.write_text("A bird\nA dog\n")
    records = [{"prompt_index": i, "sample_index": 0, "prompt": p,
                "output_name": f"{p}-0.mp4", "seed": i}
               for i, p in enumerate(["A bird", "A dog"])]
    manifest.write_text("\n".join(json.dumps(r) for r in records))
    assert validate_manifest(manifest, prompts, expected_prompts=2)["samples_per_prompt"] == 1
    records[0]["extended_prompt"] = "rewritten"
    manifest.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="rewritten"):
        validate_manifest(manifest, prompts, expected_prompts=2)


def synthetic_evaluations(root):
    """All scores below are test fixtures, never paper measurements."""
    for experiment_id, block, seed in RUN_SPECS:
        method = "framewise" if block == 1 else "chunkwise"
        for step in (200,):
            name = f"step{step:06d}_{method}"
            directory = root / experiment_id / name
            (directory / "vbench").mkdir(parents=True)
            cell = {"name": name, "method": method, "use_ema": step > 0,
                "mode": "formal", "schedule": "main_ffe", "seed": seed,
                "profile": "main_block_ablation_200_v2", "conditioning": "pinned_qwen_rewrite",
                "rewrite_sha256": "fixture_rewrites", "reference_step": 200,
                "initial_raw_reference": step == 0,
                "manifest_sha256": "fixture_manifest", "prompt_sha256": "fixture_prompts",
                "full_info_sha256": "fixture_info", "samples_per_prompt": 5,
                "max_steps": 200, "stop_condition": "fixed_steps",
                "training_seconds": step * (.5 if block == 1 else 1.) + (seed if step else 0),
                "step": step, "generator_updates": step // 5,
                "critic_updates": step, "milestones": [step],
                "r1_weight": 0., "r2_weight": 0., "experiment_id": experiment_id}
            (directory / "iclr_cell.json").write_text(json.dumps(cell))
            score = .4 + .01*seed + (.02 + .01*seed if method == "framewise" else 0)
            scores = {k.replace(" ", "_"): v["Min"] + score*(v["Max"]-v["Min"])
                      for k, v in NORMALIZE_DIC.items()}
            (directory / "vbench" / f"{name}_eval_results.json").write_text(json.dumps(scores))


def test_step200_summary_end_to_end(tmp_path):
    root = tmp_path / "synthetic"
    synthetic_evaluations(root)
    rows = collect_rows(root)
    summary = group_by_step(rows)
    final = next(r for r in summary if r["method"] == "chunkwise"
                 and not r["regularized"] and r["step"] == 200 and r["metric"] == "total_score")
    assert final["mean"] == pytest.approx(.405)
    assert final["sample_std"] == pytest.approx(.01 / 2**.5)
    assert final["num_training_seeds"] == 2
    assert final["actual_training_seconds_min"] == 200
    assert final["actual_training_seconds_max"] == 201
    frame = next(r for r in summary if r["method"] == "framewise"
                 and r["step"] == 200 and r["metric"] == "total_score")
    assert frame["num_training_seeds"] == 2
    assert frame["mean"] == pytest.approx(.43)
    assert frame["sample_std"] == pytest.approx(.02 / 2**.5)
    assert frame["actual_training_seconds_max"] == 101
    comparisons = same_seed_comparison(rows)
    paired = [r for r in comparisons if r["step"] == 200 and r["metric"] == "total_score"]
    assert [r["seed"] for r in paired] == [0, 1]
    assert [r["frame_minus_chunk"] for r in paired] == pytest.approx([.02, .03])
    paired_summary = next(r for r in paired_step_summary(comparisons)
                          if r["step"] == 200 and r["metric"] == "total_score")
    assert paired_summary["num_training_pairs"] == 2
    assert paired_summary["mean_frame_minus_chunk"] == pytest.approx(.025)
    assert paired_summary["mean_frame_minus_chunk"] == pytest.approx(frame["mean"] - final["mean"])
    assert paired_summary["sample_std"] == pytest.approx(.01 / 2**.5)
    with pytest.raises(ValueError, match="both training seeds"):
        paired_step_summary([r for r in comparisons if r["seed"] == 0])
    with pytest.raises(ValueError, match="duplicate seed"):
        paired_step_summary(comparisons + [comparisons[0]])
    with pytest.raises(ValueError, match="framewise and chunkwise"):
        group_by_step([r for r in rows if r["method"] == "framewise"])
    with pytest.raises(ValueError, match="seeds"):
        group_by_step([r for r in rows if r["seed"] != 1])
    # The old three-run plan is insufficient for the new two-pair summary.
    with pytest.raises(ValueError, match="seeds"):
        group_by_step([r for r in rows if r["experiment_id"] != "chunk_roff_s1"])
    extra_chunk = {**next(r for r in rows if r["method"] == "chunkwise"), "seed": 2}
    with pytest.raises(ValueError, match="seeds"):
        group_by_step(rows + [extra_chunk])
    with pytest.raises(ValueError, match="only the final step 200"):
        group_by_step([{**r, "step": 150} for r in rows])
    out = tmp_path / "report"
    subprocess.run([sys.executable, "-m", "experiments.iclr.summarize", "--evaluations",
                    str(root), "--output", str(out)], cwd=REPO, check=True, capture_output=True)
    for filename in ("points.csv", "same_seed_comparison.csv", "paired_step_summary.csv", "step200_seed_summary.csv"):
        assert (out / filename).stat().st_size > 0
    assert not (out / "curves.pdf").exists() and not (out / "curves.png").exists()
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["evaluation_scope"] == "final_checkpoint_only" and protocol["step"] == 200
    with (out / "step200_seed_summary.csv").open() as stream:
        endpoints = list(csv.DictReader(stream))
    assert all(row["step"] == "200" and row["sample_std"] != "" and row["num_training_seeds"] == "2" for row in endpoints)
    with (out / "same_seed_comparison.csv").open() as stream:
        paired_rows = list(csv.DictReader(stream))
    assert {row["seed"] for row in paired_rows} == {"0", "1"}
    assert all(float(row["frame_minus_chunk"]) == pytest.approx(.02 + .01 * int(row["seed"]))
               for row in paired_rows if row["metric"] == "total_score")
    with (out / "paired_step_summary.csv").open() as stream:
        paired_summary_rows = list(csv.DictReader(stream))
    assert all(float(row["mean_frame_minus_chunk"]) == pytest.approx(.025)
               and float(row["sample_std"]) == pytest.approx(.01 / 2**.5)
               and row["num_training_pairs"] == "2"
               for row in paired_summary_rows if row["metric"] == "total_score")
    path = next(root.rglob("iclr_cell.json"))
    cell = json.loads(path.read_text())
    cell["manifest_sha256"] = "another_noise_schedule"
    path.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="do not share"):
        collect_rows(root)
    cell["manifest_sha256"] = "fixture_manifest"
    cell["r2_weight"] = .001
    path.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="both off"):
        collect_rows(root)


def test_initialization_audit_detects_actual_weight_changes(tmp_path):
    payload = {"generator_checkpoint_sha256": "a", "generator_tensor_sha256": "b",
        "generator_source_key": "generator", "strict_load": True, "use_ema": False}
    for name, _, seed in RUN_SPECS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "initialization.json").write_text(json.dumps({**payload,
            "experiment_id": name, "initialization_seed": seed, "critic_tensor_sha256": f"critic_seed{seed}"}))
    assert audit(tmp_path)["actual_tensor_identity"] == "matched"
    changed = tmp_path / "frame_roff_s0" / "initialization.json"
    payload = json.loads(changed.read_text())
    payload["critic_tensor_sha256"] = "different"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="critic_tensor"):
        audit(tmp_path)


def test_step200_summary_and_legacy_protocol_rejection(tmp_path):
    synthetic_evaluations(tmp_path)
    rows = collect_rows(tmp_path)
    reference = [{**r, "reference_step": 200} for r in rows if r["step"] == 200]
    summary = reference_step_summary(reference)
    frame = next(r for r in summary if r["method"] == "framewise" and r["metric"] == "total_score")
    assert frame["num_training_seeds"] == 2 and frame["mean"] == pytest.approx(.43)
    with pytest.raises(ValueError, match="explicitly selected"):
        reference_step_summary([{**r, "reference_step": None} for r in rows])
    with pytest.raises(ValueError, match="seeds 0,1"):
        reference_step_summary([r for r in reference if r["experiment_id"] != "chunk_roff_s1"])
    path = next(tmp_path.rglob("iclr_cell.json"))
    cell = json.loads(path.read_text())
    cell.update(step=150, milestones=[150])
    path.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="only explicitly selected step-200"):
        collect_rows(tmp_path)
    cell.update(step=200, milestones=[200])
    cell.update(schedule="all1", use_ema=False)
    path.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="main-aligned"):
        collect_rows(tmp_path)


@pytest.mark.parametrize("module", ["collect_teacher", "summarize", "summarize_cd", "audit_initialization"])
def test_additional_cli_help(module):
    subprocess.run([sys.executable, "-m", f"experiments.iclr.{module}", "--help"],
                   check=True, capture_output=True, cwd=REPO)


def test_cd_table_pairs_raw_all1_all4_and_same_checkpoints(tmp_path):
    for arm in ("original", "rectified"):
        for schedule in ("all1", "all4"):
            name = f"cd_{arm}_{schedule}"
            directory = tmp_path / name
            (directory / "vbench").mkdir(parents=True)
            cell = {"name": name, "schedule": schedule, "method": "framewise", "use_ema": False,
                    "num_prompts": 944, "samples_per_prompt": 1, "checkpoint": f"/{arm}.pt",
                    "manifest_sha256": "m", "prompt_sha256": "p", "full_info_sha256": "i"}
            (directory / "iclr_cell.json").write_text(json.dumps(cell))
            score = .4 + (.1 if arm == "rectified" else 0)
            scores = {k.replace(" ", "_"): v["Min"] + score*(v["Max"]-v["Min"])
                      for k, v in NORMALIZE_DIC.items()}
            (directory / "vbench" / f"{name}_eval_results.json").write_text(json.dumps(scores))
    table, gains = summarize_cd(tmp_path)
    assert len(table) == 4
    assert {r["schedule"] for r in gains} == {"all1", "all4"}
    assert all(r["rectified_minus_original"] == pytest.approx(.1)
               for r in gains if r["metric"] == "total_score")
    path = tmp_path / "cd_original_all4" / "iclr_cell.json"
    cell = json.loads(path.read_text())
    cell["checkpoint"] = "/different.pt"
    path.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="same checkpoint"):
        summarize_cd(tmp_path)


def test_synthetic_trajectory_cli_cannot_masquerade_as_teacher(tmp_path):
    directory = tmp_path / "synthetic"
    directory.mkdir()
    time = torch.linspace(1, 0, 51)
    points = time[:, None, None, None, None].repeat(1, 2, 1, 1, 1)
    torch.save({"schema_version": 2, "trajectory_kind": "synthetic_test",
                "trajectory": points, "selected_timesteps": (1000*time).tolist()}, directory/"line.pt")
    command = [sys.executable, "-m", "experiments.iclr.trajectory", "--trajectories", str(directory),
               "--output", str(tmp_path / "analysis")]
    failed = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    assert failed.returncode != 0 and "synthetic fixtures" in failed.stderr
    subprocess.run(command + ["--synthetic-check"], capture_output=True, check=True, cwd=REPO)
    summary = json.loads((tmp_path / "analysis" / "summary.json").read_text())
    assert summary["analysis_kind"] == "synthetic_validation"
    assert summary["synthetic_is_paper_result"] is False
