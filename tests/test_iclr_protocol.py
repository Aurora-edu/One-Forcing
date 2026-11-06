"""CPU-only regression tests for the controlled ICLR experiment protocol."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from experiments.iclr.evaluate import cd_cells, curve_cells, command_for_cell, validate_manifest
from experiments.iclr.make_configs import make_matrix
from experiments.iclr.protocol import REPO, tensor_state_sha256, validate_matrix
from experiments.iclr.state import RestartableStream, TrainingClock
from experiments.iclr.trajectory import geometry, load_physical_trajectory, velocity_proxy
from utils.config import load_config


def matrix(smoke=False, budget=20):
    return make_matrix(load_config(str(REPO / "experiments/iclr/configs/base.yaml")),
        generator_ckpt="/missing/verified_ode.pt", teacher="/missing/teacher",
        data="/missing/data", prompt_cache="", budget_seconds=budget, smoke=smoke)


def test_strict_12_run_matrix_detects_hidden_confounders():
    configs = matrix()
    assert len(configs) == 12
    assert {(c.num_frame_per_block, c.r1_weight > 0, c.seed) for c in configs} == {
        (b, r, s) for b in (1, 3) for r in (False, True) for s in (0, 1, 2)}
    assert all(c.gradient_num_blocks == 0 and c.gradient_num_frames == 21 for c in configs)
    assert all(c.ema_weight == 0 and c.r1_loss_type == "apt" and c.r1_sigma == .1 for c in configs)
    configs[-1].gan_num_registers = 1
    with pytest.raises(ValueError, match="uncontrolled"):
        validate_matrix(configs)
    with pytest.raises(ValueError, match="12 runs"):
        validate_matrix(configs[:4])


def test_clock_thresholds_overrun_counters_and_resume():
    clock = TrainingClock(100)
    assert clock.due() == [0]
    clock.saved_fractions += clock.due()
    clock.advance(31, generator=True)
    assert clock.due() == [.25]
    clock.saved_fractions += clock.due()
    clock = TrainingClock.from_state_dict(clock.state_dict())
    clock.advance(71, generator=False)
    assert clock.due() == [.5, .75, 1]
    assert clock.exhausted and clock.training_seconds == 102
    assert (clock.generator_updates, clock.critic_updates) == (1, 2)
    assert not clock.should_continue("formal", 0)


def test_smoke_small_time_budget_still_covers_10_steps():
    clock = TrainingClock(.0001)
    while clock.should_continue("smoke", 10):
        clock.advance(1, generator=clock.step % 5 == 0)
    assert (clock.step, clock.critic_updates, clock.generator_updates) == (10, 10, 2)
    assert clock.exhausted


@pytest.mark.parametrize("rank", [0, 1])
def test_data_stream_resume_across_epoch_and_no_model_rng_consumption(rank):
    args = dict(batch_size=2, seed=13, rank=rank, world_size=2)
    stream = RestartableStream(list(range(19)), **args)
    torch.manual_seed(7)
    rng_before = torch.get_rng_state().clone()
    consumed = [next(stream).tolist() for _ in range(6)]
    saved = stream.state_dict()
    future = [next(stream).tolist() for _ in range(12)]
    restored = RestartableStream(list(range(19)), **args)
    restored.load_state_dict(saved)
    assert [next(restored).tolist() for _ in range(12)] == future
    assert torch.equal(rng_before, torch.get_rng_state())
    assert consumed[:4] != consumed[4:] + future[:2]
    changed = RestartableStream(list(range(20)), **args)
    with pytest.raises(ValueError, match="dataset_size"):
        changed.load_state_dict(saved)


def test_tensor_hash_tracks_actual_values_and_dtype():
    a = {"x": torch.arange(5, dtype=torch.float32)}
    assert tensor_state_sha256(a) == tensor_state_sha256({"x": a["x"].clone()})
    assert tensor_state_sha256(a) != tensor_state_sha256({"x": a["x"].half()})


def test_variable_speed_straight_line_has_no_geometry_but_nonzero_proxy():
    t = np.array([0., .1, .5, 1.])
    points = torch.tensor(np.stack((t ** 2, 2 * t ** 2), axis=1))
    g, _ = geometry(points, t)
    assert g["path_to_chord_ratio"] == pytest.approx(1.)
    assert g["max_distance_to_chord"] < 1e-12
    assert g["total_turn_radians"] == 0
    assert velocity_proxy(points, t)["velocity_deviation_integral_du"] > .01
    assert velocity_proxy(points, t, parameterization="arclength")["velocity_deviation_integral_du"] < 1e-25


def test_uniform_straight_line_and_nonlinear_time_change_proxy_only():
    t = np.array([0., .2, .75, 1.])
    points = torch.tensor(np.stack((t, 3*t), axis=1))
    assert velocity_proxy(points, t)["velocity_deviation_integral_du"] < 1e-25
    assert velocity_proxy(points, t, parameterization="power")["velocity_deviation_integral_du"] > 0
    assert geometry(points, t)[0]["total_turn_radians"] == 0


def test_high_noise_boundary_is_clipped_in_coordinate_not_midpoint_binned():
    times = [0., .8, 1.]
    points = torch.tensor([[0., 0.], [1., 0.], [1., 1.]], dtype=torch.float64)
    flow = velocity_proxy(points, times, high_noise=.9)
    segment = flow["segments"][-1]
    assert segment["high_noise_du"] == pytest.approx(.1)
    assert flow["high_noise_integral_du"] == pytest.approx(segment["proxy"] * .1)
    power = velocity_proxy(points, times, high_noise=.9, parameterization="power")
    assert power["segments"][-1]["high_noise_du"] == pytest.approx(1 - .9**2)
    assert power["high_noise_integral_du"] == pytest.approx(power["segments"][-1]["proxy"] * (1-.9**2))


def test_nonuniform_integration_uses_widths():
    t = [0., .01, 1.]
    p = torch.tensor([[0.], [1.], [2.]], dtype=torch.float64)
    result = velocity_proxy(p, t)
    expected = (100-2)**2 * .01 + (1/.99-2)**2 * .99
    assert result["velocity_deviation_integral_du"] == pytest.approx(expected)


def test_stationary_duplicate_points_closed_path_and_half_precision():
    t = [0., .2, .7, 1.]
    p = torch.zeros(4, 2)
    assert geometry(p, t)[0]["total_turn_radians"] == 0
    assert geometry(p, t)[0]["path_to_chord_ratio"] is None
    assert not velocity_proxy(p, t, parameterization="arclength")["defined"]
    assert velocity_proxy(p, t)["velocity_deviation_integral_du"] == 0
    repeated = torch.tensor([[0.,0.], [0.,0.], [1.,0.], [1.,1.]])
    assert geometry(repeated, t)[0]["total_turn_radians"] == pytest.approx(np.pi/2)
    assert velocity_proxy(repeated, t, parameterization="arclength")["zero_width_segments"] == 1
    closed = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,0.]])
    assert geometry(closed, t)[0]["path_to_chord_ratio"] is None
    half = torch.tensor([[.01,.02],[.013,.04],[.9,1.1],[2.2,3.3]], dtype=torch.float16)
    assert geometry(half, t)[0] == geometry(half.float(), t)[0]
    assert velocity_proxy(half, t) == velocity_proxy(half.float(), t)


def test_trailing_null_dataset_clean_is_never_ode_endpoint(tmp_path):
    path = tmp_path / "trajectory.pt"
    torch.save({"schema_version": 2, "selected_timesteps": [1000.,500.,0.,None],
        "trajectory": torch.tensor([[10.], [5.], [0.], [999.]]).half()}, path)
    points, times, _ = load_physical_trajectory(path)
    assert points.dtype == torch.float32
    assert points[:, 0].tolist() == [0, 5, 10]
    assert times.tolist() == [0, .5, 1]
    assert geometry(points, times)[0]["path_to_chord_ratio"] == 1


def test_cd_cells_and_chunk_curve_use_actual_method(tmp_path):
    cells = cd_cells("missing_original.pt", "missing_rectified.pt", "config.yaml")
    assert {(c["name"], c["schedule"]) for c in cells} == {
        (f"cd_{arm}_{schedule}", schedule) for arm in ("original", "rectified") for schedule in ("all1","all4")}
    row = {"step": 7, "checkpoint": "/missing/chunk.pt", "method": "chunkwise",
           "weight_source": "generator", "schedule": "all1", "training_seconds": 12.3,
           "generator_updates": 2, "critic_updates": 7, "fractions": [.25],
           "seed": 1, "r1_weight": 0., "r2_weight": 0., "mode": "formal",
           "experiment_id": "chunk_roff_s1", "budget_seconds": 40}
    index = tmp_path / "checkpoints.jsonl"
    index.write_text(json.dumps(row)+"\n")
    cell = curve_cells(index, "missing.yaml")[0]
    assert cell["method"] == "chunkwise" and cell["training_seconds"] == 12.3
    from types import SimpleNamespace
    args = SimpleNamespace(prompts="prompts.txt", manifest="m.jsonl", full_info="info.json",
        output="eval", gpus="0,1,2,3,4,5,6,7", vbench_python="vbpython", python=sys.executable)
    command = command_for_cell(cell, args)
    assert command[command.index("--method") + 1] == "chunkwise"
    assert "--require_no_ema" in command and "--use_ema" not in command


@pytest.mark.parametrize("module", ["make_configs", "train", "launch", "evaluate", "cd", "trajectory"])
def test_new_cli_help_without_gpu_or_weight_access(module):
    result = subprocess.run([sys.executable,"-m",f"experiments.iclr.{module}","--help"],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert "usage:" in result.stdout


def test_no_weight_dryrun_command_chain(tmp_path):
    root = tmp_path / "configs"
    base = [sys.executable, "-m", "experiments.iclr.make_configs", "--generator-ckpt", "/missing/ode.pt",
            "--budget-seconds", "20", "--output", str(root)]
    subprocess.run(base + ["--dry-run"], check=True, capture_output=True)
    assert not root.exists()
    subprocess.run(base, check=True, capture_output=True)
    for command in (
        ["launch", "--matrix", str(root / "matrix.json"), "--dry-run"],
        ["train", "--config", str(root/"frame_roff_s0.yaml"), "--output", str(tmp_path/"run"), "--dry-run"],
        ["cd", "--data", "/missing/trajectory_lmdb", "--generator-ckpt", "/missing/ode.pt", "--dry-run"],
        ["trajectory", "--trajectories", "/missing/trajectories", "--dry-run"],
        ["evaluate", "cd", "--original", "/missing/a.pt", "--rectified", "/missing/b.pt",
         "--prompts", "/missing/prompts.txt", "--manifest", "/missing/manifest.jsonl",
         "--full-info", "/missing/info.json", "--output", str(tmp_path/"eval"), "--dry-run"],
    ):
        subprocess.run([sys.executable,"-m",f"experiments.iclr.{command[0]}",*command[1:]],
                       check=True, capture_output=True, text=True)
    assert not (tmp_path/"run").exists()
    assert not (tmp_path/"eval").exists()
