import copy
import csv
import json
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from experiments.iclr import cd_local_protocol as local
from experiments.iclr.cd_original import completed_pair, preflight
from experiments.iclr.cd_local_validation import choose_rows
from experiments.iclr.evaluate import audit_cd_checkpoints
from experiments.iclr.protocol import REPO, sha256_file
from experiments.rebuttal.consolidate_results import NORMALIZE_DIC
from experiments.rebuttal.curvature_intervention import CurvatureTrajectoryDataset
from experiments.rebuttal.train_curvature_cd import validate_config as trainer_validate
from utils.config import load_config
from test_iclr_cd_original import write_runs


def config():
    return load_config(str(REPO / "experiments/iclr/configs/cd_local_all1.yaml"))


@pytest.mark.parametrize("step,strength", [(1, .01), (50, .5), (100, 1.), (300, 1.)])
def test_local_interpolation_changes_only_one_state_in_nine_frame_window(step, strength):
    data = torch.tensor([10., 8., 6., 4., 1., 3.]).reshape(1, 6, 1, 1, 1, 1).repeat(1, 1, 21, 1, 1, 1)
    times = local.original.TIMESTEPS + [None]
    changed, actual, mse = local.transform(data, times, step=step, enabled=True)
    assert actual == strength and mse > 0
    mask = torch.zeros_like(data, dtype=torch.bool); mask[:, 1, :9] = True
    assert torch.equal(changed[~mask], data[~mask]) and torch.equal(data[:, 1], torch.full_like(data[:, 1], 8.))
    weight = (times[1] - times[2]) / (times[0] - times[2])
    expected = (1 - strength) * 8 + strength * (weight * 10 + (1 - weight) * 6)
    torch.testing.assert_close(changed[:, 1, :9], torch.full_like(changed[:, 1, :9], expected))
    original, a, m = local.transform(data, times, step=step, enabled=False)
    assert torch.equal(original, data) and a == 0 and m == 0


def test_ten_local_cd_optimizer_updates_preserve_the_pure_objective():
    from test_iclr_cd_original import Toy
    from experiments.rebuttal.train_curvature_cd import backward_cd_objective, adjacent_pair_index
    data = torch.tensor([10., 8., 6., 4., 1., 3.]).reshape(1, 6, 1, 1, 1, 1).repeat(1, 1, 21, 1, 1, 1)
    online, reference, target = Toy(), Toy(), Toy()
    opt = torch.optim.AdamW(online.parameters(), lr=2e-6, betas=(.9, .999))
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=2e-6, betas=(.9, .999))
    for step in range(1, 11):
        treated, a, _ = local.transform(data, local.original.TIMESTEPS + [None], step=step, enabled=True)
        expected = data.clone(); expected[:, 1, :9] = 8 + a * .5
        pair = adjacent_pair_index(step, 4)
        opt.zero_grad(); ref_opt.zero_grad()
        value = backward_cd_objective(online, target, treated, local.original.TIMESTEPS + [None], {}, pair, 9, 0.)
        y = expected[:, pair + 1, :9] * (target.weight.detach() if pair < 3 else 1.)
        loss = (reference.weight * expected[:, pair, :9] - y).square().mean()
        loss.backward()
        torch.testing.assert_close(online.weight.grad, reference.weight.grad)
        opt.step(); ref_opt.step()
        torch.testing.assert_close(online.weight, reference.weight)
        with torch.no_grad(): target.weight.mul_(.95).add_(online.weight, alpha=.05)
        assert value["endpoint_loss"] == 0 and value["loss"] == value["cd_loss"]
    assert online.frames == [9] * 10 and target.weight.grad is None


@pytest.mark.parametrize("key,value", [("training_num_frames", 21), ("endpoint_loss_weight", 1.),
    ("local_rectification_ramp_steps", 200), ("local_rectification_point", 2), ("iclr_cd_eval_schedule", "all4")])
def test_shared_trainer_dispatch_rejects_unapproved_local_settings(key, value):
    c = config(); trainer_validate(c)
    c[key] = value
    with pytest.raises(ValueError): trainer_validate(c)


def test_local_launcher_and_evaluator_plan_exactly_two_all1_cells(tmp_path):
    run = subprocess.run([sys.executable, "-m", "experiments.iclr.cd_local", "--data", "/missing",
        "--generator-ckpt", "/missing.pt", "--output", str(tmp_path / "train"), "--dry-run"],
        cwd=REPO, check=True, text=True, capture_output=True)
    assert run.stdout.count("--intervention") == 2 and "all4" not in run.stdout
    command = [sys.executable, "-m", "experiments.iclr.evaluate", "cd", "--training-profile", local.PROFILE,
        "--original", "/missing/o.pt", "--rectified", "/missing/r.pt", "--prompts", "/missing/p.txt",
        "--manifest", "/missing/m.jsonl", "--full-info", "/missing/i.json", "--output", str(tmp_path / "eval"), "--dry-run"]
    run = subprocess.run(command, cwd=REPO, check=True, text=True, capture_output=True)
    assert run.stdout.count("run_vbench_condition.sh") == 2
    assert run.stdout.count("--schedule all1") == 2 and "all4" not in run.stdout
    assert not (tmp_path / "eval").exists() and not (tmp_path / "train").exists()


def local_runs(root):
    paths, runs = write_runs(root)
    for path, run in zip(paths, runs):
        arm = run["intervention"]
        run.update(iclr_cd_profile=local.PROFILE, config=OmegaConf.to_container(config()),
                   rectification_treatment="none" if arm == "curved" else "high_noise_local_progressive")
        directory = path.parent.parent
        (directory / "training.done").write_text(json.dumps(run))
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        for row in metrics:
            a = local.alpha(row["step"], enabled=arm == "rectified")
            row.update(rectification_alpha=a, rectification_mse=.01 * a * a)
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics))
        torch.save({"step": 300, "generator": {"test": torch.ones(1)},
                    "experiment_metadata": {**run, "checkpoint_step": 300}}, path)
    return paths, runs


def test_local_completed_pair_audits_ramp_and_actual_raw_weights(tmp_path):
    paths, runs = local_runs(tmp_path)
    audit = completed_pair(tmp_path, profile=local.PROFILE)
    assert audit["actual_updates"]["rectified"]["updates_per_adjacent_pair"] == [75] * 4
    assert audit["actual_updates"]["rectified"]["final_rectification_alpha"] == 1
    eval_config = REPO / "experiments/rebuttal/configs/eval_all1.yaml"
    assert audit_cd_checkpoints(*paths, eval_config, training_profile=local.PROFILE)["cd_checkpoint_step"] == 300
    changed = copy.deepcopy(runs[1]); changed["seed"] = changed["config"]["seed"] = 2027
    with pytest.raises(ValueError, match="seed|config"):
        local.validate_done_pair(runs[0], changed)
    metrics = tmp_path / "rectified/metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[0]["rectification_alpha"] = 1.
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="ramp"):
        audit_cd_checkpoints(*paths, eval_config, training_profile=local.PROFILE)


def test_local_preflight_reuses_data_and_validation_rows_exclude_training_prompts(tmp_path):
    checkpoint = tmp_path / "source.pt"; torch.save({"test": torch.ones(1)}, checkpoint)
    inputs = tmp_path / "inputs"; inputs.mkdir()
    data = torch.tensor([10., 8., 6., 4., 1., 3.]).reshape(6, 1, 1, 1, 1).repeat(1, 21, 1, 1, 1)
    for index in range(12):
        torch.save({"schema_version": 2, "prompt": f"synthetic {0 if index == 2 else index}", "trajectory": data,
            "selected_timesteps": local.original.TIMESTEPS + [None], "noise_seed": index,
            "guidance_scale": 6., "generator_ckpt": str(checkpoint), "use_ema": False,
            "generator_checkpoint_sha256": sha256_file(checkpoint)}, inputs / f"{index:05d}.pt")
    output = tmp_path / "data"
    subprocess.run([sys.executable, "-m", "experiments.rebuttal.build_curvature_dataset", "--trajectory_dir",
        str(inputs), "--output_lmdb", str(output)], cwd=REPO, check=True, capture_output=True)
    audit = preflight(output, checkpoint, config())
    assert set(audit["resolved_eval_timesteps"]) == {"all1"}
    assert audit["iclr_cd_profile"] == local.PROFILE
    assert audit["mean_intervention_mse"][1] > 0 and sum(audit["mean_intervention_mse"][2:]) == 0
    dataset = CurvatureTrajectoryDataset(str(output), "curved")
    try:
        assert choose_rows(dataset, {0, 1}, 3) == [3, 4, 5]
        with pytest.raises(ValueError, match="insufficient"):
            choose_rows(dataset, set(range(12)), 1)
    finally: dataset.env.close()


def test_local_summary_needs_only_all1_and_rejects_four_step_outputs(tmp_path):
    from experiments.iclr.summarize_cd import summarize
    for arm in ("original", "rectified"):
        name = f"cd_{arm}_all1"; directory = tmp_path / "eval" / name
        (directory / "vbench").mkdir(parents=True)
        cell = {"name": name, "schedule": "all1", "method": "framewise", "use_ema": False,
                "samples_per_prompt": 1, "num_prompts": 944, "checkpoint": f"/{arm}.pt",
                "manifest_sha256": "m", "prompt_sha256": "p", "full_info_sha256": "f",
                "cd_pair_controls": {"iclr_cd_profile": local.PROFILE}, "cd_checkpoint_step": 300}
        (directory / "iclr_cell.json").write_text(json.dumps(cell))
        value = .4 if arm == "original" else .3
        scores = {k.replace(" ", "_"): v["Min"] + value * (v["Max"] - v["Min"]) for k, v in NORMALIZE_DIC.items()}
        (directory / "vbench" / f"{name}_eval_results.json").write_text(json.dumps(scores))
    records, gains = summarize(tmp_path / "eval")
    assert len(records) == 2 and {r["schedule"] for r in gains} == {"all1"}
    assert next(r["rectified_minus_original"] for r in gains if r["metric"] == "total_score") == pytest.approx(-.1)
    subprocess.run([sys.executable, "-m", "experiments.iclr.summarize_cd", "--evaluations", str(tmp_path / "eval"),
        "--output", str(tmp_path / "summary")], cwd=REPO, check=True, capture_output=True)
    assert json.loads((tmp_path / "summary/summary.json").read_text())["evaluation_cell_count"] == 2
    (tmp_path / "eval/cd_original_all4").mkdir()
    with pytest.raises(ValueError, match="two all1"):
        summarize(tmp_path / "eval")
