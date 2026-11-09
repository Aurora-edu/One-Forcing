"""Keep the reused GAN arm and new DMD-only training/evaluation comparable."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from experiments.iclr.audit_gan_pair import audit, PAIR_DIFFERENCES
from experiments.iclr.evaluate import curve_cells
from experiments.iclr.make_configs import make_matrix
from experiments.iclr.protocol import (REPO, SEED2026_PROFILE, DMD_ONLY_PROFILE,
    config_dict, resume_signature, validate_config)
from utils.config import load_config


def pair_configs():
    base = load_config(str(REPO / "experiments/iclr/configs/base.yaml"))
    return [make_matrix(base, generator_ckpt="/shared/ode.pt", teacher="/shared/teacher",
        data="/shared/data", prompt_cache="", profile=profile)[0]
        for profile in (SEED2026_PROFILE, DMD_ONLY_PROFILE)]


def test_only_objective_and_identity_or_evaluation_metadata_change():
    gan, dmd = map(config_dict, pair_configs())
    assert {k for k in gan.keys() | dmd.keys() if gan.get(k) != dmd.get(k)} == PAIR_DIFFERENCES
    assert dmd["gradient_num_blocks"] == 17 and dmd["num_frame_per_block"] == 1
    assert dmd["seed"] == 2026 and dmd["max_steps"] == 200
    for key, value in (("gan_g_weight", .03), ("gan_d_weight", .03), ("seed", 0),
                       ("num_frame_per_block", 3), ("gradient_num_frames", 18),
                       ("iclr_eval_weight_source", "generator_ema"), ("iclr_eval_schedule", "main_ffe")):
        changed = pair_configs()[1]
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_cli_plans_one_companion_and_cannot_enter_frame_chunk_evaluator(tmp_path):
    output = tmp_path / "configs"
    subprocess.run([sys.executable, "-m", "experiments.iclr.make_configs", "--profile", DMD_ONLY_PROFILE,
        "--generator-ckpt", "/missing/ode.pt", "--output", str(output)], check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.launch", "--matrix",
        str(output / "matrix.json"), "--dry-run"], check=True, capture_output=True, text=True)
    assert "DRY RUN: 1 runs" in result.stdout and result.stdout.count("--nproc_per_node=8") == 1
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.train", "--config",
        str(output / "dmd_only_s2026.yaml"), "--output", str(tmp_path / "run"), "--dry-run"],
        check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["weight_source"] == "generator"
    assert not (tmp_path / "run").exists()
    index = tmp_path / "checkpoints.jsonl"
    index.write_text(json.dumps({"profile": DMD_ONLY_PROFILE}))
    with pytest.raises(ValueError, match="matched curve"):
        curve_cells(index, "unused.yaml")


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def completed_pair(tmp_path):
    runs = []
    for cfg, gan in zip(pair_configs(), (True, False)):
        run = tmp_path / cfg.experiment_id
        run.mkdir()
        runs.append(run)
        OmegaConf.save(cfg, run / "resolved_config.yaml")
        init = {"initialization_seed": 2026, "experiment_id": cfg.experiment_id,
            "strict_load": True, "use_ema": False, "use_gan_branch": gan,
            "generator_checkpoint_sha256": "shared_file", "generator_tensor_sha256": "shared_generator",
            "generator_source_key": "generator", "teacher_model_path": "/shared/teacher",
            "critic_backbone_tensor_sha256": "shared_backbone",
            "critic_tensor_sha256": "with_head" if gan else "without_head"}
        write_json(run / "initialization.json", init)
        write_json(run / "source.json", {"git_commit": "same_commit", "git_status": []})
        clock = {"step": 200, "max_steps": 200, "generator_updates": 40, "critic_updates": 200,
                 "training_seconds": 100.}
        write_json(run / "training.done", {**clock, "formal_result": True, "step_budget_reached": True})
        ckpt = run / "checkpoint_model_000200"
        ckpt.mkdir()
        torch.save({"step": 200, "generator": {"weight": torch.ones(1)},
            "generator_ema": {"weight": torch.zeros(1)},
            "experiment_metadata": {"profile": cfg.iclr_profile, "mode": "formal"}}, ckpt / "model.pt")
        write_json(ckpt / "complete.json", {"schema_version": 3, "world_size": 8,
            "config": resume_signature(cfg), "initialization": init, "clock": clock,
            "files": {"model.pt": (ckpt / "model.pt").stat().st_size}})
        rows = []
        for step in range(1, 201):
            row = {"step": step, "critic_loss": .1, "critic_grad_norm": .1,
                "gan_d_loss": .1 if gan else 0., "r1_loss": 0., "r2_loss": 0.}
            if (step - 1) % 5 == 0:
                row.update(generator_loss=.1, generator_grad_norm=.1, dmd_loss=.1,
                           gan_g_loss=.1 if gan else 0.)
            rows.append(row)
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return runs


def test_audit_accepts_different_heads_but_same_backbone_and_selects_raw(completed_pair):
    result = audit(*completed_pair)
    assert result["status"] == "pair_verified" and len(result["endpoints"]) == 2
    assert result["evaluation"] == {"weight_source": "generator", "schedule": "ffe",
                                     "conditioning": "pinned_qwen", "samples_per_prompt": 1}
    assert all(r["selected_weight_source"] == "generator" for r in result["endpoints"].values())
    assert audit(*completed_pair, initialization_only=True)["endpoints"] == {}


@pytest.mark.parametrize("change", ["backbone", "data", "budget", "gan_loss", "nan", "raw", "profile"])
def test_audit_rejects_mismatched_training_or_weights(completed_pair, change):
    run = completed_pair[1]
    if change == "backbone":
        path = run / "initialization.json"
        data = json.loads(path.read_text())
        data["critic_backbone_tensor_sha256"] = "different"
        write_json(path, data)
    elif change == "data":
        path = run / "resolved_config.yaml"
        cfg = OmegaConf.load(path)
        cfg.data_path = "/different/data"
        OmegaConf.save(cfg, path)
    elif change == "budget":
        path = run / "training.done"
        data = json.loads(path.read_text())
        data["generator_updates"] = 39
        write_json(path, data)
    elif change in ("gan_loss", "nan"):
        path = run / "metrics.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]["gan_g_loss"] = .1 if change == "gan_loss" else float("nan")
        path.write_text("\n".join(json.dumps(r) for r in rows))
    else:
        path = run / "checkpoint_model_000200/model.pt"
        data = torch.load(path, weights_only=True)
        if change == "raw":
            del data["generator"]
        else:
            data["experiment_metadata"]["profile"] = SEED2026_PROFILE
        torch.save(data, path)
        manifest = path.parent / "complete.json"
        complete = json.loads(manifest.read_text())
        complete["files"]["model.pt"] = path.stat().st_size
        write_json(manifest, complete)
    with pytest.raises((ValueError, KeyError)):
        audit(*completed_pair)


def test_trainer_records_common_backbone_without_changing_rng(monkeypatch):
    from experiments.iclr import trainer
    cfg = pair_configs()[0]
    common = torch.nn.Linear(2, 2)
    records = []
    for gan in (True, False):
        wrapper = torch.nn.Module()
        wrapper.model = common
        if gan:
            wrapper.head = torch.nn.Linear(2, 1)
        model = SimpleNamespace(fake_score=wrapper, use_gan_branch=gan)
        monkeypatch.setattr(trainer.Trainer, "_build_model", lambda self, config: model)
        instance = trainer.ICLRTrainer.__new__(trainer.ICLRTrainer)
        instance.is_main_process, instance.restart = True, None
        before = torch.get_rng_state().clone()
        instance._build_model(cfg)
        assert torch.equal(before, torch.get_rng_state())
        records.append(instance.initialization)
    assert records[0]["critic_backbone_tensor_sha256"] == records[1]["critic_backbone_tensor_sha256"]
    assert records[0]["critic_tensor_sha256"] != records[1]["critic_tensor_sha256"]
