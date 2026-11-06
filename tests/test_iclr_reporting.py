import json
import subprocess
import sys

import pytest
import torch

from experiments.iclr.audit_initialization import audit
from experiments.iclr.collect_teacher import make_schedule, euler_update
from experiments.iclr.evaluate import record_cell, validate_manifest
from experiments.iclr.protocol import REPO
from experiments.iclr.summarize import group_by_timepoint, collect_rows
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
    for method in ("framewise", "chunkwise"):
        for regularized in (False, True):
            for seed in (0, 1, 2):
                for fraction in (0., .25, .5, .75, 1.):
                    name = f"step{int(fraction*100):06d}_{method}"
                    directory = root / f"{method}_{regularized}_{seed}" / name
                    (directory / "vbench").mkdir(parents=True)
                    cell = {"name": name, "method": method, "use_ema": False,
                        "mode": "formal", "schedule": "all1", "seed": seed,
                        "manifest_sha256": "fixture_manifest", "prompt_sha256": "fixture_prompts",
                        "full_info_sha256": "fixture_info", "samples_per_prompt": 1,
                        "budget_seconds": 100., "training_seconds": 100*fraction + (seed if fraction else 0),
                        "step": int(fraction*100), "generator_updates": int(fraction*20),
                        "critic_updates": int(fraction*100), "fractions": [fraction],
                        "r1_weight": float(regularized), "r2_weight": .001*regularized,
                        "experiment_id": f"{method}_{regularized}_{seed}"}
                    (directory / "iclr_cell.json").write_text(json.dumps(cell))
                    scores = {k.replace(" ", "_"): v["Min"] + (.4 + .01*seed)*(v["Max"]-v["Min"])
                              for k, v in NORMALIZE_DIC.items()}
                    (directory / "vbench" / f"{name}_eval_results.json").write_text(json.dumps(scores))


def test_time_summary_and_plot_end_to_end(tmp_path):
    root = tmp_path / "synthetic"
    synthetic_evaluations(root)
    rows = collect_rows(root)
    summary = group_by_timepoint(rows)
    final = next(r for r in summary if r["method"] == "chunkwise"
                 and not r["regularized"] and r["fraction"] == 1. and r["metric"] == "total_score")
    assert final["mean"] == pytest.approx(.41)
    assert final["sample_std"] == pytest.approx(.01)
    assert final["actual_training_seconds_min"] == 100
    assert final["actual_training_seconds_max"] == 102
    with pytest.raises(ValueError, match="four experimental arms"):
        group_by_timepoint([r for r in rows if r["method"] == "framewise"])
    with pytest.raises(ValueError, match="seeds"):
        group_by_timepoint([r for r in rows if r["seed"] != 2])
    with pytest.raises(ValueError, match="initial and final"):
        group_by_timepoint([r for r in rows if 1. not in r["fractions"]])
    out = tmp_path / "report"
    subprocess.run([sys.executable, "-m", "experiments.iclr.summarize", "--evaluations",
                    str(root), "--output", str(out)], cwd=REPO, check=True, capture_output=True)
    for filename in ("points.csv", "endpoints.csv", "matched_time_summary.csv", "curves.pdf", "curves.png"):
        assert (out / filename).stat().st_size > 0
    path = next(root.rglob("iclr_cell.json"))
    cell = json.loads(path.read_text())
    cell["manifest_sha256"] = "another_noise_schedule"
    path.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="do not share"):
        collect_rows(root)


def test_initialization_audit_detects_actual_weight_changes(tmp_path):
    payload = {"generator_checkpoint_sha256": "a", "generator_tensor_sha256": "b",
        "critic_tensor_sha256": "c", "initialization_seed": 12345,
        "generator_source_key": "generator", "strict_load": True, "use_ema": False}
    for i in range(12):
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / "initialization.json").write_text(json.dumps(payload))
    assert audit(tmp_path)["actual_tensor_identity"] == "matched"
    payload["critic_tensor_sha256"] = "different"
    (tmp_path / "0" / "initialization.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="critic_tensor"):
        audit(tmp_path)


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
