"""CPU regression checks for the historical CD recipe and its result boundary."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from experiments.iclr import cd_original_protocol as protocol
from experiments.iclr.cd_original import CONFIG, completed_pair, preflight
from experiments.iclr.evaluate import audit_cd_checkpoints
from experiments.iclr.protocol import REPO, sha256_file
from experiments.rebuttal.consolidate_results import NORMALIZE_DIC
from experiments.rebuttal.train_curvature_cd import backward_cd_objective, adjacent_pair_index, validate_config
from utils.config import load_config


def config():
    return load_config(str(CONFIG))


@pytest.mark.parametrize("field,value", [("training_num_frames", 21), ("endpoint_loss_weight", 1.),
    ("holdout_rows", 64), ("seed", 0), ("lr", 1e-5), ("target_ema_decay", .99)])
def test_original_recipe_cannot_drift_into_the_endpoint_variant(field, value):
    c = config()
    validate_config(c)  # Actual shared trainer dispatch, not just the helper.
    c[field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(c)


def test_default_and_smoke_launcher_are_cpu_only_and_restore_seed(tmp_path):
    command = [sys.executable, "-m", "experiments.iclr.cd_original", "--data", "/missing/data",
        "--generator-ckpt", "/missing/source.pt", "--output", str(tmp_path / "new"), "--dry-run"]
    for extra, steps in (([], 300), (["--smoke"], 10)):
        run = subprocess.run(command + extra, cwd=REPO, check=True, text=True, capture_output=True)
        assert run.stdout.count("--intervention") == 2
        assert run.stdout.count("--seed 2026") == 2
        assert f"{steps} updates/arm" in run.stdout and "endpoint=0; holdout=0" in run.stdout
    assert not (tmp_path / "new").exists()


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))
        self.frames = []

    def forward(self, noisy_image_or_video, **kwargs):
        self.frames.append(noisy_image_or_video.shape[1])
        return None, noisy_image_or_video * self.weight


def test_ten_actual_optimizer_updates_match_pure_cd_and_ignore_tail_frames():
    # Synthetic trajectory only. NaNs beyond frame 9 must never enter training.
    data = torch.tensor([10., 8., 6., 4., 1., 3.]).reshape(1, 6, 1, 1, 1, 1).repeat(1, 1, 21, 1, 1, 1)
    data[:, :, 9:] = float("nan")
    online, reference, target = Toy(), Toy(), Toy()
    target.weight.requires_grad_(False)
    opt = torch.optim.AdamW(online.parameters(), lr=2e-6, betas=(.9, .999))
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=2e-6, betas=(.9, .999))
    for step in range(1, 11):
        pair = adjacent_pair_index(step, 4)
        opt.zero_grad(); ref_opt.zero_grad()
        value = backward_cd_objective(online, target, data, protocol.TIMESTEPS + [None], {}, pair, 9, 0.)
        expected_target = data[:, pair + 1, :9].clone()
        if pair < 3:
            expected_target *= target.weight.detach()
        loss = (reference.weight * data[:, pair, :9] - expected_target).square().mean()
        loss.backward()
        torch.testing.assert_close(online.weight.grad, reference.weight.grad)
        assert value["endpoint_loss"] == 0 and value["loss"] == value["cd_loss"]
        opt.step(); ref_opt.step()
        torch.testing.assert_close(online.weight, reference.weight)
        with torch.no_grad():
            target.weight.mul_(.95).add_(online.weight, alpha=.05)
    assert online.frames == [9] * 10  # No extra endpoint forward.
    assert target.weight.grad is None


def test_existing_21_frame_lmdb_is_reused_on_cpu_only_with_matching_source(tmp_path):
    source = tmp_path / "source.pt"
    torch.save({"generator": {"test": torch.ones(1)}}, source)
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    data = torch.tensor([10., 8., 6., 4., 1., 3.]).reshape(6, 1, 1, 1, 1).repeat(1, 21, 1, 1, 1)
    for index in range(8):
        torch.save({"schema_version": 2, "prompt": "synthetic test", "trajectory": data,
            "selected_timesteps": protocol.TIMESTEPS + [None], "noise_seed": index,
            "guidance_scale": 6., "generator_ckpt": str(source), "use_ema": False,
            "generator_checkpoint_sha256": sha256_file(source)}, trajectories / f"{index:05d}.pt")
    lmdb = tmp_path / "lmdb"
    subprocess.run([sys.executable, "-m", "experiments.rebuttal.build_curvature_dataset",
        "--trajectory_dir", str(trajectories), "--output_lmdb", str(lmdb)], cwd=REPO, check=True, capture_output=True)
    manifest = lmdb / "curvature_dataset_manifest.json"
    before = sha256_file(manifest)
    audit = preflight(lmdb, source, config())
    assert audit["trajectories_reused"] and audit["num_training_rows"] == 8
    assert audit["training_frames"] == 9 and audit["historical_score_reproduction_guaranteed"] is False
    assert sha256_file(manifest) == before
    command = [sys.executable, "-m", "experiments.iclr.cd_original", "--data", str(lmdb),
        "--generator-ckpt", str(source), "--output", str(tmp_path / "preflight"), "--preflight"]
    subprocess.run(command, cwd=REPO, check=True, capture_output=True)
    assert json.loads((tmp_path / "preflight/preflight.json").read_text())["status"] == "pass"
    torch.save({"different": torch.zeros(1)}, source)
    with pytest.raises(ValueError, match="source hash"):
        preflight(lmdb, source, config())


def write_runs(root):
    paths, runs = [], []
    for arm in ("curved", "rectified"):
        directory = root / arm
        directory.mkdir(parents=True)
        rows = [{"step": step, "pair_index": adjacent_pair_index(step, 4), "endpoint_loss_weight": 0.,
            "endpoint_updates": 0, "endpoint_loss": 0., "loss": .1, "cd_loss": .1,
            "grad_norm": .2, "seconds": .01, "intervention": arm,
            "global_row_indices": list(range((step - 1) * 8, step * 8))} for step in range(1, 301)]
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        digest = hashlib.sha256()
        for r in rows:
            digest.update(f"{r['step']}:".encode() + ",".join(str(i) for i in r["global_row_indices"]).encode() + b"\n")
        run = {"iclr_cd_profile": protocol.PROFILE, "initial_generator_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64, "seed": 2026, "world_size": 8,
            "training_num_frames": 9, "training_timesteps": protocol.TIMESTEPS,
            "adjacent_pairs": [list(p) for p in zip(protocol.TIMESTEPS[:-1], protocol.TIMESTEPS[1:])],
            "pair_schedule": "deterministic_low_to_high_sweep", "training_objective": "adjacent_state_consistency_l2",
            "target_network": "in_training_ema_stop_gradient", "target_ema_decay": .95,
            "endpoint_loss_weight": 0., "holdout_row_indices": [], "global_sample_order_sha256": digest.hexdigest(),
            "history_conditioning": "teacher_forcing_clean", "evaluation_use_ema": False,
            "final_step": 300, "config": OmegaConf.to_container(config()), "intervention": arm,
            "status": "complete", "use_ema": False}
        (directory / "training.done").write_text(json.dumps(run))
        checkpoint = directory / "checkpoint_model_000300/model.pt"
        checkpoint.parent.mkdir()
        torch.save({"step": 300, "generator": {"test": torch.ones(1)},
            "experiment_metadata": {**run, "checkpoint_step": 300}}, checkpoint)
        paths.append(checkpoint); runs.append(run)
    return paths, runs


def test_actual_update_audit_and_raw_checkpoint_preflight_reject_mixed_runs(tmp_path):
    paths, runs = write_runs(tmp_path)
    result = completed_pair(tmp_path)
    assert result["actual_updates"]["curved"]["updates_per_adjacent_pair"] == [75] * 4
    eval_config = REPO / "experiments/rebuttal/configs/eval_all1.yaml"
    audit = audit_cd_checkpoints(*paths, eval_config, training_profile=protocol.PROFILE)
    assert audit["cd_pair_controls"]["endpoint_loss_weight"] == 0
    with pytest.raises(ValueError, match="one-step CD"):
        audit_cd_checkpoints(*paths, eval_config)  # Endpoint profile cannot accept these runs.
    changed = copy.deepcopy(runs[1]); changed["seed"] = 0
    with pytest.raises(ValueError, match="seed"):
        protocol.validate_done_pair(runs[0], changed)
    metrics = tmp_path / "curved/metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[0]["endpoint_updates"] = 1
    metrics.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="endpoint supervision"):
        audit_cd_checkpoints(*paths, eval_config, training_profile=protocol.PROFILE)


def test_four_cell_summary_keeps_negative_gains_and_labels_historical_reference(tmp_path):
    from experiments.iclr.summarize_cd import summarize
    for arm in ("original", "rectified"):
        for schedule in ("all1", "all4"):
            name = f"cd_{arm}_{schedule}"; directory = tmp_path / "eval" / name
            (directory / "vbench").mkdir(parents=True)
            cell = {"name": name, "schedule": schedule, "method": "framewise", "use_ema": False,
                    "samples_per_prompt": 1, "num_prompts": 944, "checkpoint": f"/{arm}.pt",
                    "manifest_sha256": "m", "prompt_sha256": "p", "full_info_sha256": "f",
                    "cd_pair_controls": {"iclr_cd_profile": protocol.PROFILE}, "cd_checkpoint_step": 300}
            (directory / "iclr_cell.json").write_text(json.dumps(cell))
            value = .4 if arm == "original" else .3  # Synthetic negative result must survive.
            scores = {k.replace(" ", "_"): v["Min"] + value * (v["Max"] - v["Min"])
                      for k, v in NORMALIZE_DIC.items()}
            (directory / "vbench" / f"{name}_eval_results.json").write_text(json.dumps(scores))
    records, gains = summarize(tmp_path / "eval")
    assert all(r["rectified_minus_original"] == pytest.approx(-.1) for r in gains if r["metric"] == "total_score")
    output = tmp_path / "summary"
    subprocess.run([sys.executable, "-m", "experiments.iclr.summarize_cd", "--evaluations",
        str(tmp_path / "eval"), "--output", str(output)], cwd=REPO, check=True, capture_output=True)
    history = list(csv.DictReader((output / "historical_comparison.csv").open()))
    assert len(history) == 12
    row = next(r for r in history if r["arm"] == "original" and r["schedule"] == "all4" and r["metric"] == "total_score")
    assert float(row["historical_score"]) == pytest.approx(.6115644845113526)
    assert json.loads((output / "summary.json").read_text())["historical_score_reproduction_guaranteed"] is False
