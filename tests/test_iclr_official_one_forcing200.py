"""Fresh official seed2027 training and registration into the common evaluation."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.iclr import official_frame_chunk as task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/rebuttal"))
import run_official_baselines as evaluation


@pytest.fixture
def run(tmp_path):
    source, output = tmp_path / "official", tmp_path / "one2027"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(task.REPO), str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach", task.OFFICIAL], check=True)
    task.prepare(source, output, profile=task.ONE_PROFILE)
    return source, output


def test_one_run_only_changes_official_seed_and_preserves_old_profiles(run):
    source, output = run
    manifest = task.check(output, required_profile=task.ONE_PROFILE)
    assert manifest["seed"] == 2027 and manifest["steps"] == 200
    assert list(manifest["training_configs"]) == ["one_forcing"]
    original = task.read_yaml(source / "config.yaml")
    assert task.read_yaml(output / "configs/one_forcing.yaml") == dict(original, seed=2027)
    assert task.profile_settings(task.FRAME_PROFILE)["seed"] == 42
    assert task.profile_settings(task.GAN_PROFILE)["seed"] == 42
    with pytest.raises(ValueError, match="This entry requires"):
        task.check(output, required_profile=task.FRAME_PROFILE)
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.official_one_forcing200",
                             "train", "--pair", str(output), "--arm", "one_forcing",
                             "--generator", "/assets/ode.pt", "--teacher", "/assets/teacher",
                             "--data", "/assets/data", "--wan", "/assets/wan", "--dry-run"],
                            cwd=task.REPO, capture_output=True, text=True, check=True)
    assert "train.py" in result.stdout and "DRY RUN" in result.stdout
    assert not (output / "runs").exists()


def test_completed_training_seed_and_hash_are_required_for_evaluation(run, tmp_path, monkeypatch):
    source, output = run
    assets = tmp_path / "assets"
    for name in ("teacher", "data", "wan"):
        (assets / name).mkdir(parents=True)
    for name in ("data/data.mdb", "wan/Wan2.1_VAE.pth", "wan/models_t5_umt5-xxl-enc-bf16.pth"):
        (assets / name).write_bytes(b"CPU fixture")
    state = {"model.weight": torch.ones(2)}
    torch.save({"generator": state}, assets / "generator.pt")
    original_popen = subprocess.Popen

    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs["cwd"] == source and "--nproc_per_node=8" in command
            assert "PYTHONPATH" not in kwargs["env"]
            checkpoint = output / "runs/one_forcing/checkpoint_model_000200/model.pt"
            checkpoint.parent.mkdir()
            torch.save({"step": 200, "generator": state}, checkpoint)
            rows = [{"step": step, "critic_loss": .2, "fake_score_loss": .1,
                     "gan_d_loss": .01, "r1_loss": 0., "r2_loss": 0.,
                     **({"generator_loss": .3, "dmd_loss": .2, "gan_g_loss": .01}
                        if step % 5 == 1 else {})} for step in range(1, 201)]
            self.stdout = iter(repr(row) + "\n" for row in rows)
        def wait(self): return 0
        def poll(self): return 0

    def popen(command, *args, **kwargs):
        return Process(command, **kwargs) if isinstance(command, list) and command[0] == sys.executable and "train.py" in command else original_popen(command, *args, **kwargs)

    monkeypatch.setattr(task.subprocess, "Popen", popen)
    monkeypatch.setattr(task, "require_idle", lambda gpus: None)
    monkeypatch.setenv("PYTHONPATH", "/must/not/leak")
    task.worker_train(SimpleNamespace(pair=str(output), arm="one_forcing", python=sys.executable,
        gpus="0,1,2,3,4,5,6,7", generator=str(assets / "generator.pt"), teacher=str(assets / "teacher"),
        data=str(assets / "data"), wan=str(assets / "wan")))
    manifest = task.check(output)
    done = task.audit_training(output, "one_forcing", manifest)
    assert done["seed"] == 2027 and done["generator_updates"] == 40 and done["critic_updates"] == 200
    assert json.loads((output / "runs/one_forcing/step10_health.json").read_text())["last_step"] == 10
    work = tmp_path / "evaluation"
    args = SimpleNamespace(work_root=work, model="one_forcing", one_forcing_run=output,
                           wan_path=None, download_weights=False, phase="full", schedule="ffe")
    evaluation.prepare_sources(args)
    intent = evaluation.build_intent(args, "one_forcing")
    assert intent["training"]["seed"] == 2027 and intent["training"]["iterations"] == 200
    assert intent["weight_key"] == "generator" and intent["frames_per_block"] == 1
    assert len(intent["records"]) == 4720
    assert intent["training"]["checkpoint_sha256"] == done["checkpoint_sha256"]
    assert evaluation.models(SimpleNamespace(model="both")) == ["self_forcing", "causal_forcing"]
    assert len(evaluation.models(SimpleNamespace(model="all"))) == 3
    # A receipt with another training seed must not be accepted as the new run.
    done_path = output / "runs/one_forcing/training.done"
    done["seed"] = 42
    done_path.write_text(json.dumps(done))
    with pytest.raises(ValueError, match="completion"):
        evaluation.verify_checkpoint(Path(intent["checkpoint"]), "one_forcing", work)
