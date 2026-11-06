"""CPU-only regression tests for the controlled ICLR experiment protocol."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from experiments.iclr.evaluate import cd_cells, curve_cells, command_for_cell, validate_manifest
from experiments.iclr.make_configs import make_matrix
from experiments.iclr.protocol import REPO, MAIN_REFERENCE, REFERENCE_RUNTIME_FIELDS, effective_gradient_frames, tensor_state_sha256, validate_config, validate_matrix
from experiments.iclr.state import RestartableStream, TrainingClock
from experiments.iclr.trajectory import geometry, load_physical_trajectory, velocity_proxy
from utils.config import load_config


def matrix(smoke=False):
    return make_matrix(load_config(str(REPO / "experiments/iclr/configs/base.yaml")),
        generator_ckpt="/missing/verified_ode.pt", teacher="/missing/teacher",
        data="/missing/data", prompt_cache="", smoke=smoke)


def test_strict_four_run_matrix_detects_hidden_confounders():
    configs = matrix()
    assert [(c.experiment_id, c.num_frame_per_block, c.seed) for c in configs] == [
        ("frame_roff_s0", 1, 0), ("chunk_roff_s0", 3, 0),
        ("frame_roff_s1", 1, 1), ("chunk_roff_s1", 3, 1)]
    assert all(c.r1_weight == 0 and c.r2_weight == 0 for c in configs)
    assert all(c.gradient_num_blocks == 17 and "gradient_num_frames" not in c for c in configs)
    assert [effective_gradient_frames(c) for c in configs] == [17, 21, 17, 21]
    assert all(c.ema_weight == .99 and c.ema_start_step == 50 and c.max_steps == 200 for c in configs)
    configs[-1].prompt_embedding_cache_path = "/different_cache"
    with pytest.raises(ValueError, match="uncontrolled"):
        validate_matrix(configs)
    with pytest.raises(ValueError, match="4 runs"):
        validate_matrix(configs[:3])
    with pytest.raises(ValueError, match="4 runs"):
        validate_matrix(matrix() * 3)


@pytest.mark.parametrize("field,value", [("r1_weight", 1.), ("r2_weight", .001),
                                        ("seed", 2), ("experiment_id", "chunk_roff_s1")])
def test_four_run_protocol_rejects_unauthorized_arms(field, value):
    configs = matrix()
    configs[1][field] = value
    with pytest.raises(ValueError, match="main reference|both be off|allowed runs"):
        validate_matrix(configs)


def test_four_run_protocol_rejects_duplicates_and_wrong_order():
    configs = matrix()
    with pytest.raises(ValueError, match="order/cell mismatch"):
        validate_matrix([configs[1], configs[0], configs[2], configs[3]])
    with pytest.raises(ValueError, match="order/cell mismatch"):
        validate_matrix([configs[0], configs[0], configs[2], configs[3]])


def test_all_scientific_settings_match_pinned_main_except_block_size():
    reference = OmegaConf.load(MAIN_REFERENCE)
    for config in matrix():
        for key in reference:
            if key not in REFERENCE_RUNTIME_FIELDS:
                assert config[key] == reference[key], key
    changed = matrix()[0]
    changed.ema_weight = 0
    with pytest.raises(ValueError, match="main reference"):
        validate_config(changed)
    changed = matrix()[0]
    changed.gradient_num_blocks = 0
    changed.gradient_num_frames = 21
    with pytest.raises(ValueError, match="main reference"):
        validate_config(changed)


@pytest.mark.parametrize("seconds", [.001, 1., 10000.])
def test_formal_clock_stops_at_exactly_200_regardless_of_elapsed_time(seconds):
    clock = TrainingClock()
    for step in range(200):
        assert clock.should_continue()
        clock.advance(seconds, generator=step % 5 == 0)
    assert not clock.should_continue()
    assert (clock.step, clock.generator_updates, clock.critic_updates) == (200, 40, 200)


def test_protocol_rejects_old_time_budget_and_non200_formal_steps():
    config = matrix()[0]
    config.max_steps = 750
    with pytest.raises(ValueError, match="200"):
        validate_config(config)
    config = matrix()[0]
    config.iclr_budget_seconds = 7200
    with pytest.raises(ValueError, match="time-budget"):
        validate_config(config)


@pytest.mark.parametrize("method,block", [("framewise", 1), ("chunkwise", 3)])
def test_main_ffe_shell_config_and_pipeline_keep_fixed_blocks(tmp_path, method, block):
    # Run the actual config-producing shell path, but replace GPU export with a no-op.
    stub = tmp_path / "python_without_export"
    stub.write_text("#!/bin/sh\nif [ \"$1\" = \"-\" ]; then\n  exec "
                    + shlex.quote(sys.executable) + " \"$@\"\nfi\nexit 0\n")
    stub.chmod(0o755)
    checkpoint = tmp_path / "unused.pt"
    checkpoint.touch()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A bird flying\n")
    out = tmp_path / "export"
    subprocess.run(["bash", "scripts/infer.sh", "--method", method, "--schedule", "main_ffe",
                    "--config_path", str(MAIN_REFERENCE), "--checkpoint_path", str(checkpoint),
                    "--prompt_path", str(prompts), "--output_folder", str(out)],
                   cwd=REPO, env={**os.environ, "PYTHON_BIN": str(stub)}, check=True, capture_output=True)
    cfg = load_config(str(out / f"infer_config_{method}_main_ffe_shard0.yaml"))
    assert cfg.model_kwargs == OmegaConf.load(MAIN_REFERENCE).model_kwargs
    assert cfg.num_frame_per_block == block and cfg.rollout_schedule == "fixed"
    from types import SimpleNamespace
    from pipeline.causal_inference import CausalInferencePipeline
    dummy = SimpleNamespace(model=SimpleNamespace(local_attn_size=-1),
                            get_scheduler=lambda: SimpleNamespace(timesteps=torch.linspace(1000, 1, 1000)))
    pipeline = CausalInferencePipeline(cfg, device=torch.device("cpu"), generator=dummy,
                                       text_encoder=object(), vae=object())
    assert pipeline._build_rollout_frame_counts(21, None) == [block] * (21 // block)
    assert len(pipeline._block_denoising_step_list(0)) == 4
    assert len(pipeline._block_denoising_step_list(1)) == 1


def test_clock_step_milestones_counters_and_resume():
    clock = TrainingClock(10, (0, 5, 10))
    assert clock.due() == [0]
    clock.saved_steps += clock.due()
    for step in range(5):
        clock.advance(31, generator=step % 5 == 0)
    assert clock.due() == [5]
    clock.saved_steps += clock.due()
    clock = TrainingClock.from_state_dict(clock.state_dict())
    assert clock.due() == []
    for step in range(5, 10):
        clock.advance(71, generator=step % 5 == 0)
    assert clock.due() == [10]
    assert clock.exhausted and clock.training_seconds == 510
    assert (clock.generator_updates, clock.critic_updates) == (2, 10)
    assert not clock.should_continue()


def test_smoke_always_covers_10_steps():
    clock = TrainingClock(10, (0, 5, 10))
    while clock.should_continue():
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


def test_main_data_order_repeats_the_original_sampler_epoch_and_resumes():
    stream = RestartableStream(list(range(12)), batch_size=1, seed=0,
                               rank=0, world_size=2, fixed_sampler_epoch=True)
    first = [next(stream).item() for _ in range(6)]
    assert [next(stream).item() for _ in range(6)] == first
    next(stream)
    saved = stream.state_dict()
    resumed = RestartableStream(list(range(12)), batch_size=1, seed=0,
                                rank=0, world_size=2, fixed_sampler_epoch=True)
    resumed.load_state_dict(saved)
    assert [next(resumed).item() for _ in range(10)] == [next(stream).item() for _ in range(10)]


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
    row = {"step": 50, "checkpoint": "/missing/chunk.pt", "method": "chunkwise",
           "weight_source": "generator", "schedule": "all1", "training_seconds": 12.3,
           "generator_updates": 10, "critic_updates": 50, "milestones": [50],
           "seed": 0, "r1_weight": 0., "r2_weight": 0., "mode": "formal",
           "experiment_id": "chunk_roff_s0", "max_steps": 200, "stop_condition": "fixed_steps",
           "profile": "main_block_ablation_200_v2", "available_weights": ["generator", "generator_ema"]}
    index = tmp_path / "checkpoints.jsonl"
    index.write_text(json.dumps(row)+"\n")
    cell = curve_cells(index, "missing.yaml")[0]
    assert cell["method"] == "chunkwise" and cell["training_seconds"] == 12.3
    from types import SimpleNamespace
    args = SimpleNamespace(prompts="prompts.txt", manifest="m.jsonl", full_info="info.json",
        output="eval", gpus="0,1,2,3,4,5,6,7", vbench_python="vbpython", python=sys.executable)
    command = command_for_cell(cell, args)
    assert command[command.index("--method") + 1] == "chunkwise"
    assert "--use_ema" in command and "--require_no_ema" not in command
    assert cell["schedule"] == "main_ffe"


@pytest.mark.parametrize("module", ["make_configs", "train", "launch", "evaluate", "cd", "trajectory"])
def test_new_cli_help_without_gpu_or_weight_access(module):
    result = subprocess.run([sys.executable,"-m",f"experiments.iclr.{module}","--help"],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert "usage:" in result.stdout


def test_no_weight_dryrun_command_chain(tmp_path):
    root = tmp_path / "configs"
    base = [sys.executable, "-m", "experiments.iclr.make_configs", "--generator-ckpt", "/missing/ode.pt",
            "--output", str(root)]
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
        result = subprocess.run([sys.executable,"-m",f"experiments.iclr.{command[0]}",*command[1:]],
                                check=True, capture_output=True, text=True)
        if command[0] == "launch":
            lines = result.stdout.splitlines()
            assert len(lines) == 5 and "DRY RUN: 4 runs" in lines[-1]
            assert all(f"{name}.yaml" in line for name, line in zip(
                ("frame_roff_s0", "chunk_roff_s0", "frame_roff_s1", "chunk_roff_s1"), lines))
        elif command[0] == "cd":
            lines = result.stdout.splitlines()
            assert len(lines) == 3
            assert "--intervention curved --seed 0" in lines[0]
            assert "--intervention rectified --seed 0" in lines[1]
            assert "300 updates per arm" in lines[2] and "NO resume" in lines[2]
    assert not (tmp_path/"run").exists()
    assert not (tmp_path/"eval").exists()
    manifest_path = root / "matrix.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"][0]["name"] = "frame_ron_s0"
    manifest_path.write_text(json.dumps(manifest))
    rejected = subprocess.run([sys.executable, "-m", "experiments.iclr.launch", "--matrix",
                               str(manifest_path), "--dry-run"], capture_output=True, text=True)
    assert rejected.returncode != 0 and "run names" in rejected.stderr
