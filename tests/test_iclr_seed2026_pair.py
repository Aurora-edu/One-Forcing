"""Check that the new seed cannot silently change the historical ablation."""
import json
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from experiments.iclr.evaluate import audit_window_checkpoint, curve_cells
from experiments.iclr.make_configs import make_matrix
from experiments.iclr.protocol import (REPO, SEED2026_PROFILE, SEED2026_RUN_SPECS,
    config_dict, effective_gradient_frames, resume_signature, validate_matrix)
from experiments.iclr.summarize_seed2026_pair import paired_rows
from utils.config import load_config


def matrix(profile=None):
    kwargs = {"profile": profile} if profile else {}
    return make_matrix(load_config(str(REPO / "experiments/iclr/configs/base.yaml")),
        generator_ckpt="/missing/ode.pt", teacher="/missing/teacher", data="/missing/data",
        prompt_cache="", **kwargs)


def test_only_seed_and_identity_differ_from_historical_runs():
    old, new = matrix(), matrix(SEED2026_PROFILE)
    assert [(c.experiment_id, c.num_frame_per_block, c.seed) for c in new] == list(SEED2026_RUN_SPECS)
    assert [effective_gradient_frames(c) for c in new] == [17, 21]
    for previous, current in zip(old, new):
        a, b = config_dict(previous), config_dict(current)
        assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {
            "seed", "experiment_id", "iclr_profile"}
    for key, value in (("gradient_num_blocks", 0), ("gradient_num_frames", 18),
                       ("seed", 42), ("max_steps", 750), ("iclr_eval_conditioning", "original")):
        changed = matrix(SEED2026_PROFILE)
        changed[0][key] = value
        with pytest.raises(ValueError):
            validate_matrix(changed)


@pytest.mark.parametrize("arm", [0, 1])
def test_final_checkpoint_keeps_original_window_and_requires_actual_ema(tmp_path, arm):
    cfg = matrix(SEED2026_PROFILE)[arm]
    config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config)
    checkpoint = tmp_path / "model.pt"
    row = {"experiment_id": cfg.experiment_id, "seed": 2026, "profile": SEED2026_PROFILE,
        "schedule": "all1", "stop_condition": "fixed_steps", "mode": "formal", "max_steps": 200,
        "r1_weight": 0., "r2_weight": 0., "method": "framewise" if arm == 0 else "chunkwise",
        "gradient_num_blocks": 17, "gradient_num_frames": 0,
        "effective_gradient_frames": 17 if arm == 0 else 21,
        "training_seconds": 123., "step": 200, "generator_updates": 40, "critic_updates": 200,
        "available_weights": ["generator", "generator_ema"], "milestones": [200],
        "checkpoint": str(checkpoint)}
    index = tmp_path / "checkpoints.jsonl"
    index.write_text("\n".join(json.dumps({**row, "step": step, "milestones": [step]})
                               for step in (0, 50, 100, 150, 200)))
    cells = curve_cells(index, str(config))
    assert len(cells) == 1 and cells[0]["use_ema"] and cells[0]["schedule"] == "main_ffe"
    payload = {"step": 200, "generator_ema": {"weight": torch.ones(1)}, "experiment_metadata": {
        "profile": SEED2026_PROFILE, "mode": "formal", "contains_generator_ema": True}}
    torch.save(payload, checkpoint)
    complete = {"schema_version": 3, "world_size": 8, "config": resume_signature(cfg),
        "clock": {"step": 200, "generator_updates": 40, "critic_updates": 200},
        "files": {"model.pt": checkpoint.stat().st_size}}
    (tmp_path / "complete.json").write_text(json.dumps(complete))
    audit_window_checkpoint(cells[0])
    for key, value in (("seed", 0), ("effective_gradient_frames", 18), ("generator_updates", 39)):
        index.write_text(json.dumps({**row, key: value}))
        with pytest.raises(ValueError, match="metadata mismatch"):
            curve_cells(index, str(config))
    payload["generator_ema"] = {}
    torch.save(payload, checkpoint)
    complete["files"]["model.pt"] = checkpoint.stat().st_size
    (tmp_path / "complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="EMA"):
        audit_window_checkpoint(cells[0])


def test_cli_plans_two_new_runs_and_summary_requires_a_pair(tmp_path):
    configs = tmp_path / "configs"
    subprocess.run([sys.executable, "-m", "experiments.iclr.make_configs", "--profile", SEED2026_PROFILE,
        "--generator-ckpt", "/missing/ode.pt", "--output", str(configs)], check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.launch",
        "--matrix", str(configs / "matrix.json"), "--dry-run"], check=True, capture_output=True, text=True)
    assert "DRY RUN: 2 runs" in result.stdout and result.stdout.count("--nproc_per_node=8") == 2
    rows = []
    for name, block, seed in SEED2026_RUN_SPECS:
        result = subprocess.run([sys.executable, "-m", "experiments.iclr.train",
            "--config", str(configs / f"{name}.yaml"), "--output", str(tmp_path / name), "--dry-run"],
            check=True, capture_output=True, text=True)
        assert json.loads(result.stdout)["effective_gradient_frames"] == (17 if block == 1 else 21)
        assert not (tmp_path / name).exists()
        rows.append({"experiment_id": name, "method": "framewise" if block == 1 else "chunkwise",
            "seed": seed, "step": 200, "reference_step": 200, "regularized": False,
            "metric": "total_score", "score": .5, "training_seconds": 1.})
    assert paired_rows(rows)[0]["frame_minus_chunk"] == 0.
    for invalid in (rows[:1], rows + rows[:1], [{**r, "seed": 0} for r in rows]):
        with pytest.raises(ValueError):
            paired_rows(invalid)
