import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from experiments.iclr import cd_protocol as protocol
from experiments.iclr.cd_diagnostics import frame_errors, summarize_records, report
from experiments.iclr.protocol import REPO
from experiments.rebuttal.curvature_intervention import rectify_trajectory
from experiments.rebuttal.train_curvature_cd import backward_cd_objective
from pipeline.causal_inference import CausalInferencePipeline
from utils.config import load_config


TIMES = [1000., 937.5, 833.333312988, 625., 0., None]


def config():
    return load_config(str(REPO / "experiments/iclr/configs/cd_one_step.yaml"))


class ToyGenerator(nn.Module):
    def __init__(self, value=0.7):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, noisy_image_or_video, **kwargs):
        return None, self.weight * noisy_image_or_video


def trajectory():
    return torch.tensor([10., 8., 6., 4., 1., 3.]).reshape(1, 6, 1, 1, 1, 1).repeat(1, 1, 21, 1, 1, 1)


@pytest.mark.parametrize("pair", [0, 1, 2, 3])
def test_sequential_backward_matches_combined_loss_and_detaches_target(pair):
    online, target, reference = ToyGenerator(), ToyGenerator(.5), ToyGenerator()
    data = trajectory()
    values = backward_cd_objective(online, target, data, TIMES, {}, pair, 21, 1.)
    target_value = data[:, pair + 1] * (.5 if pair < 3 else 1.)
    cd = (reference.weight * data[:, pair] - target_value).square().mean()
    anchor = (reference.weight * data[:, 0] - data[:, -2]).square().mean()
    loss = (cd + anchor) / 2
    loss.backward()
    assert online.weight.grad.item() == pytest.approx(reference.weight.grad.item())
    assert values["loss"].item() == pytest.approx(loss.item())
    assert target.weight.grad is None
    assert values["endpoint_loss"].item() == pytest.approx(anchor.item())


def test_endpoint_term_is_identical_but_cd_intervention_changes_gradients():
    data = trajectory()
    rectified = rectify_trajectory(data[0], TIMES).unsqueeze(0)
    models = [ToyGenerator(), ToyGenerator()]
    values = [backward_cd_objective(model, ToyGenerator(.5), arm, TIMES, {}, 2, 21, 1.)
              for model, arm in zip(models, (data, rectified))]
    assert values[0]["endpoint_loss"] == values[1]["endpoint_loss"]
    assert values[0]["cd_loss"] != values[1]["cd_loss"]
    assert models[0].weight.grad != models[1].weight.grad
    with pytest.raises(FloatingPointError):
        backward_cd_objective(ToyGenerator(float("nan")), ToyGenerator(), data, TIMES, {}, 3, 21, 1.)


def test_fixed_profile_holdout_and_source_identity(tmp_path, monkeypatch):
    c = config()
    protocol.validate_one_step_config(c)
    train, heldout = protocol.split_rows(80, 64)
    assert train == list(range(64, 80)) and heldout == list(range(64))
    assert not set(train) & set(heldout)
    c.training_num_frames = 9
    with pytest.raises(ValueError, match="training_num_frames"):
        protocol.validate_one_step_config(c)
    with pytest.raises(ValueError):
        protocol.split_rows(64, 64)
    monkeypatch.setattr(protocol, "sha256_file", lambda path: protocol.AR_SHA256)
    manifest = {"generator_checkpoint_sha256": protocol.AR_SHA256, "use_ema": False,
                "selected_timesteps": TIMES, "latents_shape": [80, 6, 21, 1, 1, 1]}
    assert protocol.verify_ar_source(tmp_path / "moved.pt", manifest) == protocol.AR_SHA256
    with pytest.raises(ValueError, match="rebuild"):
        protocol.verify_ar_source(tmp_path / "moved.pt", {"use_ema": False})
    manifest["latents_shape"][2] = 9
    with pytest.raises(ValueError, match="21-frame"):
        protocol.verify_ar_source(tmp_path / "moved.pt", manifest)


def done_pair():
    c = OmegaConf.to_container(config())
    base = {"iclr_cd_profile": protocol.PROFILE, "initial_generator_sha256": protocol.AR_SHA256,
        "dataset_manifest_sha256": "test-dataset", "seed": 0, "world_size": 8,
        "training_num_frames": 21, "training_timesteps": TIMES[:-1],
        "adjacent_pairs": list(zip(TIMES[:-2], TIMES[1:-1])), "pair_schedule": "deterministic_low_to_high_sweep",
        "training_objective": "adjacent_cd_plus_endpoint_l2", "target_ema_decay": .95,
        "endpoint_loss_weight": 1., "holdout_row_indices": list(range(64)),
        "global_sample_order_sha256": "test-order", "final_step": 300, "config": c,
        "history_conditioning": "teacher_forcing_clean", "status": "complete", "use_ema": False}
    return [{**copy.deepcopy(base), "intervention": arm} for arm in ("curved", "rectified")]


def test_pair_and_checkpoint_audits_reject_mixed_protocols():
    original, rectified = done_pair()
    protocol.validate_done_pair(original, rectified)
    payload = {"step": 300, "generator": {}, "experiment_metadata": {**original, "checkpoint_step": 300}}
    protocol.validate_checkpoint(payload, original, 300)
    rectified["global_sample_order_sha256"] = "different-order"
    with pytest.raises(ValueError, match="global_sample_order"):
        protocol.validate_done_pair(original, rectified)
    payload["experiment_metadata"]["training_num_frames"] = 9
    with pytest.raises(ValueError, match="training_num_frames"):
        protocol.validate_checkpoint(payload, original, 300)


def test_evaluation_preflight_audits_both_actual_checkpoint_files(tmp_path):
    from experiments.iclr.evaluate import audit_cd_checkpoints
    paths = []
    for done in done_pair():
        root = tmp_path / done["intervention"]
        checkpoint = root / "checkpoint_model_000300" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        (root / "training.done").write_text(json.dumps(done))
        torch.save({"step": 300, "generator": {"test": torch.ones(1)},
                    "experiment_metadata": {**done, "checkpoint_step": 300}}, checkpoint)
        paths.append(checkpoint)
    result = audit_cd_checkpoints(*paths, REPO / "experiments/rebuttal/configs/eval_all1.yaml")
    assert result["cd_checkpoint_step"] == 300
    assert result["cd_pair_controls"]["iclr_cd_profile"] == protocol.PROFILE
    torch.save({"step": 50, "generator": {}}, paths[1])
    with pytest.raises(ValueError, match="raw checkpoint"):
        audit_cd_checkpoints(*paths, REPO / "experiments/rebuttal/configs/eval_all1.yaml")


def test_builder_preserves_source_checkpoint_identity(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    for index in range(2):
        torch.save({"schema_version": 2, "prompt": "synthetic test", "trajectory": trajectory()[0],
            "selected_timesteps": TIMES, "noise_seed": index, "guidance_scale": 6.,
            "generator_ckpt": "/synthetic/ar_diffusion.pt", "use_ema": False,
            "generator_checkpoint_sha256": protocol.AR_SHA256}, sources / f"{index:05d}.pt")
    output = tmp_path / "lmdb"
    subprocess.run([sys.executable, "-m", "experiments.rebuttal.build_curvature_dataset",
        "--trajectory_dir", str(sources), "--output_lmdb", str(output)], check=True, capture_output=True)
    manifest = json.loads((output / "curvature_dataset_manifest.json").read_text())
    assert manifest["generator_checkpoint_sha256"] == protocol.AR_SHA256
    assert manifest["latents_shape"] == [2, 6, 21, 1, 1, 1]


class HistoryGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(local_attn_size=3)

    def get_scheduler(self):
        return SimpleNamespace(timesteps=torch.arange(1000, 0, -1))

    def forward(self, noisy_image_or_video, timestep, kv_cache, current_start, **kwargs):
        history = kv_cache[0]["history"]
        prediction = noisy_image_or_video + history[:, :current_start].sum(1, keepdim=True)
        if timestep.max() == 0:
            history[:, current_start:current_start + 1] = noisy_image_or_video
        return None, prediction


class EmptyEncoder(nn.Module):
    def forward(self, **kwargs):
        return {}


def test_real_inference_path_only_replaces_past_context_and_resets_between_samples():
    args = SimpleNamespace(denoising_step_list=[1000], warp_denoising_step=False,
                           independent_first_frame=False, num_frame_per_block=1, context_noise=0)
    pipeline = CausalInferencePipeline(args, "cpu", generator=HistoryGenerator(),
                                      text_encoder=EmptyEncoder(), vae=nn.Identity())
    pipeline.num_transformer_blocks = pipeline.frame_seq_length = 1
    pipeline._initialize_kv_cache = lambda **kwargs: setattr(
        pipeline, "kv_cache1", [{"history": torch.zeros(1, 3, 1, 1, 1)}])
    pipeline._initialize_crossattn_cache = lambda **kwargs: setattr(pipeline, "crossattn_cache", [{"is_init": False}])
    noise = torch.ones(1, 3, 1, 1, 1)
    context = torch.tensor([2., 3., 99.]).reshape_as(noise)
    def infer(history=None):
        return pipeline.inference(noise, ["synthetic test"], return_video=False,
                                  return_latents=True, context_latents=history)[1]
    free_before = infer()
    forced = infer(context)
    context[:, -1] = -9000
    assert torch.equal(forced, infer(context))  # current/future clean never leaks
    assert forced.flatten().tolist() == [1., 3., 6.]
    assert free_before.flatten().tolist() == [1., 2., 4.]
    assert torch.equal(free_before, infer())
    errors = frame_errors(forced, free_before, forced)
    assert errors[0]["prediction_shift"] == 0
    assert errors[2]["history_gap"] == 4


def synthetic_records():
    return [{"arm": arm, "step": step, "row_index": 0, "frame": frame,
             "prompt": "synthetic test", "noise_seed": 0, "forced_mse": 2.,
             "free_mse": 1., "history_gap": -1., "prediction_shift": 1.}
            for arm in ("curved", "rectified") for step in (0, 300) for frame in range(21)]


def test_diagnostic_reporting_preserves_negative_results_and_exports_curves(tmp_path):
    records = synthetic_records()
    rows, deltas = summarize_records(records, [0, 300], [0])
    assert all(r["history_gap"] == -1 for r in rows)
    assert all(r["history_gap"] == 0 for r in deltas)
    with pytest.raises(ValueError, match="incomplete"):
        summarize_records(records[:-1], [0, 300], [0])
    (tmp_path / "rank_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for arm in ("curved", "rectified"):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "metrics.jsonl").write_text(json.dumps(
            {"step": 1, "cd_loss": 1., "endpoint_loss": 2., "grad_norm": 3.}) + "\n")
    report(tmp_path, tmp_path, [0, 300], [0])
    assert (tmp_path / "learning_curves.pdf").stat().st_size > 100
    assert (tmp_path / "learning_curves.png").stat().st_size > 100


def test_diagnostic_cli_is_a_nonmutating_cpu_dry_run(tmp_path):
    output = tmp_path / "diagnostics"
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.cd_diagnostics",
        "--training-root", "/missing/run", "--data", "/missing/data", "--output", str(output), "--dry-run"],
        capture_output=True, text=True, check=True, cwd=REPO)
    assert "DRY RUN" in result.stdout and not output.exists()
