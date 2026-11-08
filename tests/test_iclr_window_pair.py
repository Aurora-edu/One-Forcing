"""CPU checks for actual gradient flow and isolated seed-42 endpoint reporting."""
import json
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from experiments.iclr.audit_initialization import audit
from experiments.iclr.evaluate import audit_window_checkpoint, curve_cells
from experiments.iclr.make_configs import make_matrix
from experiments.iclr.protocol import (REPO, WINDOW_PROFILE, WINDOW_RUN_SPECS,
    effective_gradient_frames, resume_signature, validate_config, validate_matrix)
from experiments.iclr.summarize import collect_rows
from experiments.iclr.summarize_window_pair import audit_exports, paired_rows
from experiments.rebuttal.consolidate_results import NORMALIZE_DIC
from utils.config import load_config


def configs():
    return make_matrix(load_config(str(REPO / "experiments/iclr/configs/base.yaml")),
        generator_ckpt="/missing/ode.pt", teacher="/missing/teacher", data="/missing/data",
        prompt_cache="", profile=WINDOW_PROFILE)


def test_pair_changes_only_granularity_and_both_windows_are_18():
    pair = configs()
    validate_matrix(pair)
    a, b = [OmegaConf.to_container(c, resolve=True) for c in pair]
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {
        "experiment_id", "num_frame_per_block"}
    assert [(c.experiment_id, c.num_frame_per_block, c.seed) for c in pair] == list(WINDOW_RUN_SPECS)
    assert all(effective_gradient_frames(c) == 18 and c.gradient_num_blocks == 0
               and c.gradient_num_frames == 18 and c.randomize_seed is False for c in pair)
    old = make_matrix(load_config(str(REPO / "experiments/iclr/configs/base.yaml")),
        generator_ckpt="/missing/ode.pt", teacher="/missing/teacher", data="/missing/data", prompt_cache="")
    assert resume_signature(pair[0]) != resume_signature(old[0])


@pytest.mark.parametrize("field,value", [("gradient_num_blocks", 17), ("gradient_num_frames", 17),
    ("gradient_num_frames", 21), ("seed", 0), ("r1_weight", 1.), ("ema_weight", .9),
    ("gradient_window_position", "first"), ("iclr_eval_samples_per_prompt", 1), ("max_steps", 750)])
def test_pair_rejects_hidden_changes(field, value):
    pair = configs()
    pair[1][field] = value
    with pytest.raises(ValueError):
        validate_matrix(pair)


def test_pair_rejects_extra_missing_and_duplicate_runs():
    pair = configs()
    for changed in (pair[:1], pair * 2, [pair[0], pair[0]], list(reversed(pair))):
        with pytest.raises(ValueError):
            validate_matrix(changed)


@pytest.mark.parametrize("block", [1, 3])
def test_real_training_rollout_and_loss_mask_backpropagate_exactly_tail_18(monkeypatch, block):
    # Exercise the real temporal loop, replacing only Wan and its cache allocation.
    # Each latent frame has an independent parameter, revealing any extra gradient.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    from model.base import SelfForcingModel
    from pipeline.self_forcing_training import SelfForcingTrainingPipeline
    cfg = configs()[0 if block == 1 else 1]
    model = SimpleNamespace(args=cfg, num_frame_per_block=block)
    resolved = SelfForcingModel._resolve_gradient_num_frames(model, 21)
    assert resolved == effective_gradient_frames(cfg) == 18
    model._resolve_gradient_num_frames = lambda n: SelfForcingModel._resolve_gradient_num_frames(model, n)
    model._get_gradient_window_position = lambda: "tail"
    mask = SelfForcingModel._build_gradient_mask(model, torch.zeros(1, 21, 1, 1, 1), 21, 21)
    assert mask.flatten().tolist() == [False] * 3 + [True] * 18

    class ToyGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frames = torch.nn.Parameter(torch.ones(21))
            self.gradient_calls = []

        def forward(self, noisy_image_or_video, timestep, current_start, **kwargs):
            start = current_start // 1560
            length = noisy_image_or_video.shape[1]
            if int(timestep[0, 0]) > 0 and torch.is_grad_enabled():
                self.gradient_calls.extend(range(start, start + length))
            pred = self.frames[start:start + length].reshape(1, length, 1, 1, 1).expand_as(noisy_image_or_video)
            # A real operation inside no_grad ensures the result is detached there.
            return None, pred * 1

    generator = ToyGenerator()
    scheduler = SimpleNamespace(timesteps=torch.linspace(1000, 1, 1000),
                                add_noise=lambda clean, noise, t: clean)
    pipeline = SelfForcingTrainingPipeline(torch.tensor([1000]), scheduler, generator,
        num_frame_per_block=block, same_step_across_blocks=True, gradient_num_frames=resolved)
    monkeypatch.setattr(pipeline, "_initialize_kv_cache", lambda **kw: None)
    monkeypatch.setattr(pipeline, "_initialize_crossattn_cache", lambda **kw: None)
    prediction, _, _ = pipeline.inference_with_trajectory(torch.zeros(1, 21, 1, 1, 1))
    prediction.sum().backward()
    assert generator.gradient_calls == list(range(3, 21))
    torch.testing.assert_close(generator.frames.grad, torch.tensor([0.] * 3 + [1.] * 18))


def make_index(tmp_path, cfg):
    config = tmp_path / f"{cfg.experiment_id}.yaml"
    OmegaConf.save(cfg, config, resolve=True)
    index = tmp_path / "checkpoints.jsonl"
    row = {"experiment_id": cfg.experiment_id, "seed": 42, "profile": WINDOW_PROFILE,
        "schedule": "all1", "stop_condition": "fixed_steps", "mode": "formal", "max_steps": 200,
        "r1_weight": 0., "r2_weight": 0., "method": "framewise" if cfg.num_frame_per_block == 1 else "chunkwise",
        "gradient_num_blocks": 0, "gradient_num_frames": 18, "effective_gradient_frames": 18,
        "training_seconds": 123., "step": 200, "generator_updates": 40, "critic_updates": 200,
        "available_weights": ["generator", "generator_ema"], "milestones": [200],
        "checkpoint": str(tmp_path / "model.pt")}
    index.write_text("\n".join(json.dumps({**row, "step": step, "milestones": [step]})
                               for step in (0, 50, 100, 150, 200)))
    return config, index, row


def test_only_step200_ema_and_actual_saved_protocol_can_be_evaluated(tmp_path):
    cfg = configs()[1]
    config, index, row = make_index(tmp_path, cfg)
    cells = curve_cells(index, str(config))
    assert len(cells) == 1 and cells[0]["use_ema"] is True and cells[0]["schedule"] == "main_ffe"
    assert cells[0]["effective_gradient_frames"] == 18
    payload = {"step": 200, "generator_ema": {"w": torch.ones(1)}, "experiment_metadata": {
        "profile": WINDOW_PROFILE, "mode": "formal", "contains_generator_ema": True}}
    torch.save(payload, tmp_path / "model.pt")
    complete = {"schema_version": 3, "world_size": 8, "config": resume_signature(cfg),
        "clock": {"step": 200, "generator_updates": 40, "critic_updates": 200},
        "files": {"model.pt": (tmp_path / "model.pt").stat().st_size}}
    (tmp_path / "complete.json").write_text(json.dumps(complete))
    audit_window_checkpoint(cells[0])
    complete["config"]["gradient_num_frames"] = 21
    (tmp_path / "complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="saved training protocol"):
        audit_window_checkpoint(cells[0])
    for field, value in (("effective_gradient_frames", 21), ("generator_updates", 39),
                         ("seed", 0), ("experiment_id", "chunk_roff_s0")):
        index.write_text(json.dumps({**row, field: value}))
        with pytest.raises(ValueError, match="metadata mismatch"):
            curve_cells(index, str(config))
    index.write_text(json.dumps({**row, "available_weights": ["generator"]}))
    with pytest.raises(ValueError, match="EMA"):
        curve_cells(index, str(config))


def synthetic_pair(root):
    for name, block, seed in WINDOW_RUN_SPECS:
        method = "framewise" if block == 1 else "chunkwise"
        cellname = f"step000200_{method}"
        directory = root / name / cellname
        (directory / "vbench").mkdir(parents=True)
        cell = {"name": cellname, "method": method, "seed": seed, "experiment_id": name,
            "profile": WINDOW_PROFILE, "use_ema": True, "mode": "formal", "schedule": "main_ffe",
            "conditioning": "pinned_qwen_rewrite", "samples_per_prompt": 5,
            "reference_step": 200, "step": 200, "max_steps": 200, "milestones": [200],
            "stop_condition": "fixed_steps", "r1_weight": 0, "r2_weight": 0,
            "manifest_sha256": "fixture_manifest", "prompt_sha256": "fixture_prompt",
            "full_info_sha256": "fixture_info", "rewrite_sha256": "fixture_rewrites",
            "gradient_num_blocks": 0, "gradient_num_frames": 18, "effective_gradient_frames": 18,
            "config": str(root / name / "resolved_config.yaml"), "checkpoint": str(root / name / "model.pt"),
            "training_seconds": block * 100, "generator_updates": 40, "critic_updates": 200}
        (directory / "iclr_cell.json").write_text(json.dumps(cell))
        score = .6 if block == 1 else .5  # Synthetic numbers, never measured results.
        scores = {k.replace(" ", "_"): v["Min"] + score * (v["Max"] - v["Min"])
                  for k, v in NORMALIZE_DIC.items()}
        (directory / "vbench" / f"{cellname}_eval_results.json").write_text(json.dumps(scores))
        (directory / "vbench" / f"{cellname}_protocol.json").write_text(json.dumps({
            "mode": "vbench_standard", "samples_per_prompt": 5, "official_five_sample_protocol": True,
            "dimensions": list(scores)}))
        videos = directory / "videos"
        videos.mkdir()
        (videos / "export.intent.json").write_text(json.dumps({"method": method, "schedule": "main_ffe",
            "use_ema": True, "checkpoint_path": cell["checkpoint"], "config_path": cell["config"],
            "manifest_sha256": "fixture_manifest", "prompt_sha256": "fixture_prompt",
            "extended_prompt_sha256": "fixture_rewrites", "selected_manifest_records": 4720,
            "num_output_frames": 21, "fps": 16}))
        (videos / "export.done").write_text(json.dumps({"status": "complete", "num_videos": 4720,
            "weight_source": "generator_ema", "use_ema": True, "checkpoint_path": cell["checkpoint"],
            "manifest_sha256": "fixture_manifest", "latent_frames_per_video": 21,
            "rgb_frames_per_video": 81, "fps": 16}))


def test_single_pair_summary_has_no_seed_variance_and_rejects_unpaired_inputs(tmp_path):
    root = tmp_path / "evaluations"
    synthetic_pair(root)
    rows = collect_rows(root, profile=WINDOW_PROFILE)
    paired = paired_rows(rows)
    assert len(paired) == 19
    total = next(r for r in paired if r["metric"] == "total_score")
    assert total["frame_score"] == pytest.approx(.6)
    assert total["chunk_score"] == pytest.approx(.5)
    assert total["frame_minus_chunk"] == pytest.approx(.1)
    assert all("sample_std" not in row for row in paired)
    for changed in (rows + rows[:1], [r for r in rows if r["method"] == "framewise"]):
        with pytest.raises(ValueError):
            paired_rows(changed)
    with pytest.raises(ValueError):
        collect_rows(root)  # Old four-run summaries must not accept this pair.
    out = tmp_path / "summary"
    subprocess.run([sys.executable, "-m", "experiments.iclr.summarize_window_pair",
        "--evaluations", str(root), "--output", str(out)], check=True, capture_output=True)
    assert json.loads((out / "protocol.json").read_text())["num_training_pairs"] == 1
    exportpath = next(root.rglob("export.done"))
    done = json.loads(exportpath.read_text())
    exportpath.write_text(json.dumps({**done, "weight_source": "generator", "use_ema": False}))
    with pytest.raises(ValueError, match="actual EMA/FFE"):
        audit_exports(root)
    cellpath = next(root.rglob("iclr_cell.json"))
    cell = json.loads(cellpath.read_text())
    cell["manifest_sha256"] = "different_noise"
    cellpath.write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="do not share"):
        collect_rows(root, profile=WINDOW_PROFILE)


def test_pair_initialization_checks_both_raw_and_critic_identities(tmp_path):
    for name, _, seed in WINDOW_RUN_SPECS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "initialization.json").write_text(json.dumps({"experiment_id": name,
            "initialization_seed": seed, "generator_checkpoint_sha256": "same_file",
            "generator_tensor_sha256": "same_raw", "generator_source_key": "generator",
            "critic_tensor_sha256": "same_critic", "strict_load": True, "use_ema": False}))
    assert audit(tmp_path, profile=WINDOW_PROFILE)["num_runs"] == 2
    path = tmp_path / "chunk_w18_s42" / "initialization.json"
    row = json.loads(path.read_text())
    row["critic_tensor_sha256"] = "different"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="critic_tensor_sha256"):
        audit(tmp_path, profile=WINDOW_PROFILE)


def test_pair_cli_plans_exactly_two_runs_without_gpu_or_weight_access(tmp_path):
    directory = tmp_path / "configs"
    subprocess.run([sys.executable, "-m", "experiments.iclr.make_configs",
        "--profile", WINDOW_PROFILE, "--generator-ckpt", "/missing/ode.pt",
        "--output", str(directory)], check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.launch",
        "--matrix", str(directory / "matrix.json"), "--dry-run"], check=True, capture_output=True, text=True)
    assert "DRY RUN: 2 runs" in result.stdout
    assert result.stdout.count("--nproc_per_node=8") == 2
    for name, _, _ in WINDOW_RUN_SPECS:
        result = subprocess.run([sys.executable, "-m", "experiments.iclr.train",
            "--config", str(directory / f"{name}.yaml"), "--output", str(tmp_path / name), "--dry-run"],
            check=True, capture_output=True, text=True)
        assert json.loads(result.stdout)["effective_gradient_frames"] == 18
        assert not (tmp_path / name).exists()
