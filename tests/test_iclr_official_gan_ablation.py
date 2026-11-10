"""Verify the native 300-iteration GAN ablation without running a GPU model."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from experiments.iclr import official_frame_chunk as task

ENTRY = "experiments.iclr.official_gan_ablation"


@pytest.fixture
def pair(tmp_path):
    root, output = tmp_path / "official", tmp_path / "gan300"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(task.REPO), str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--quiet", "--detach", task.OFFICIAL], check=True)
    task.prepare(root, output, profile=task.GAN_PROFILE)
    return root, output


def test_both_arms_are_official_framewise300_with_only_gan_weights_different(pair):
    root, output = pair
    original = task.read_yaml(root / "config.yaml")
    gan, dmd = [task.read_yaml(output / f"configs/{arm}.yaml") for arm in ("dmd_gan", "dmd_only")]
    assert gan == dict(original, seed=42, max_steps=300)
    assert dmd == dict(gan, gan_g_weight=0., gan_d_weight=0.)
    assert {key for key in gan if gan[key] != dmd[key]} == {"gan_g_weight", "gan_d_weight"}
    assert gan["gan_g_weight"] == gan["gan_d_weight"] == .03
    assert gan["r1_weight"] == gan["r2_weight"] == 0
    assert gan["num_frame_per_block"] == 1 and gan["gradient_num_blocks"] == 17
    assert gan["resume_ckpt"] == "" and gan["trainer"] == "one_forcing"
    manifest = task.check(output, required_profile=task.GAN_PROFILE)
    assert manifest["steps"] == 300 and manifest["effective_gradient_frames"] == {"dmd_gan": 17, "dmd_only": 17}
    assert task.git(root, "status", "--porcelain") == ""


@pytest.mark.parametrize("key,value", [("gan_g_weight", .01), ("max_steps", 200),
                                      ("seed", 2026), ("gradient_num_blocks", 18)])
def test_protocol_rejects_scientific_changes_even_if_rehashed(pair, key, value):
    _, output = pair
    path = output / "configs/dmd_gan.yaml"
    cfg = task.read_yaml(path); cfg[key] = value
    OmegaConf.save(OmegaConf.create(cfg), path)
    manifest_path = output / "pair.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_configs"]["dmd_gan"] = task.sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="config differs"):
        task.check(output)


def test_inference_config_matches_actual_official_shell_for_both_arms(pair, tmp_path):
    root, output = pair
    wrapper = tmp_path / "cpu_python"
    wrapper.write_text("#!/bin/sh\nif [ \"$1\" = \"-\" ]; then\n  exec " +
                      shlex.quote(sys.executable) + " \"$@\"\nfi\n" +
                      "[ \"$1\" = \"scripts/export_videos.py\" ] || exit 9\nexit 0\n")
    wrapper.chmod(0o755)
    checkpoint = tmp_path / "unused.pt"; checkpoint.touch()
    for arm in ("dmd_gan", "dmd_only"):
        native = tmp_path / f"native_{arm}"
        subprocess.run(["bash", "scripts/infer.sh", "--method", "framewise",
            "--config_path", str(output / f"configs/{arm}.yaml"),
            "--checkpoint_path", str(checkpoint), "--prompt_path", str(output / "inputs/prompts_0.txt"),
            "--output_folder", str(native), "--num_output_frames", "21"],
            cwd=root, env=dict(os.environ, PYTHON_BIN=str(wrapper), CUDA_VISIBLE_DEVICES=""),
            check=True, capture_output=True)
        cfg = task.read_yaml(native / "infer_config_framewise.yaml")
        assert cfg == task.read_yaml(output / f"configs/{arm}_infer.yaml")
        assert cfg["num_frame_per_block"] == 1 and cfg["denoising_step_list"] == [1000]
        assert cfg["first_frame_denoising_step_list"] == [1000, 750, 500, 250]
        assert cfg.get("rollout_schedule", "fixed") == "fixed"
        commands = task.export_commands(output, arm, sys.executable)
        assert len(commands) == 8
        assert all("--use_ema" not in command and "--schedule" not in command for command in commands)
        assert all("checkpoint_model_000300/model.pt" in " ".join(command) for command in commands)
    task.official_source(root)


def test_cli_dry_runs_and_profile_separation(pair):
    root, output = pair
    for arm in ("dmd_gan", "dmd_only"):
        result = subprocess.run([sys.executable, "-m", ENTRY, "train", "--pair", str(output),
            "--arm", arm, "--generator", "/missing/init.pt", "--teacher", "/missing/teacher",
            "--data", "/missing/data", "--wan", "/missing/wan", "--dry-run"],
            cwd=task.REPO, check=True, capture_output=True, text=True)
        assert "train.py" in result.stdout and str(root) in result.stdout
        assert "experiments.iclr.train" not in result.stdout and "DRY RUN" in result.stdout
    assert not (output / "runs").exists()
    result = subprocess.run([sys.executable, "-m", "experiments.iclr.official_frame_chunk",
                             "check", "--pair", str(output)], cwd=task.REPO, capture_output=True)
    assert result.returncode != 0 and b"This entry requires" in result.stderr


def metrics(arm, count=300):
    return [{"step": step, "critic_loss": .2, "fake_score_loss": .1,
             "gan_d_loss": .01 if arm == "dmd_gan" else 0., "r1_loss": 0., "r2_loss": 0.,
             **({"generator_loss": .3, "dmd_loss": .2,
                 "gan_g_loss": .01 if arm == "dmd_gan" else 0.} if step % 5 == 1 else {})}
            for step in range(1, count + 1)]


def test_native_worker_audits300_steps_and_objectives_then_reports_correct_pair(pair, tmp_path, monkeypatch):
    from experiments.rebuttal.consolidate_results import DIM_WEIGHT
    root, output = pair
    assets = tmp_path / "assets"; assets.mkdir()
    for name in ("teacher", "data", "wan"):
        (assets / name).mkdir()
    for name in ("data/data.mdb", "wan/Wan2.1_VAE.pth", "wan/models_t5_umt5-xxl-enc-bf16.pth"):
        (assets / name).write_bytes(b"CPU fixture")
    state = {"model.weight": torch.ones(2)}
    torch.save({"generator": state}, assets / "generator.pt")
    real_popen = subprocess.Popen
    calls = []

    class TrainingProcess:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            run = Path(command[command.index("--logdir") + 1])
            final = run / "checkpoint_model_000300/model.pt"; final.parent.mkdir()
            torch.save({"step": 300, "generator": state}, final)
            self.stdout = iter(repr(row) + "\n" for row in metrics(run.name))
        def wait(self): return 0
        def poll(self): return 0

    def popen(command, *args, **kwargs):
        if isinstance(command, list) and command[0] == sys.executable and "train.py" in command:
            return TrainingProcess(command, **kwargs)
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(task.subprocess, "Popen", popen)
    monkeypatch.setattr(task, "require_idle", lambda gpus: None)
    monkeypatch.setenv("PYTHONPATH", "/must/not/reach/official")
    manifest = task.check(output)
    for arm in ("dmd_gan", "dmd_only"):
        args = SimpleNamespace(pair=str(output), arm=arm, python=sys.executable, gpus="0,1,2,3,4,5,6,7",
            generator=str(assets / "generator.pt"), teacher=str(assets / "teacher"),
            data=str(assets / "data"), wan=str(assets / "wan"))
        task.worker_train(args)
        done = task.audit_training(output, arm, manifest)
        assert done["last_step"] == 300 and done["generator_updates"] == 60 and done["critic_updates"] == 300
        audit = json.loads((output / f"runs/{arm}/step10_objectives.json").read_text())
        assert audit["generator_loss_records"] == 2 and audit["critic_loss_records"] == 10
        cell = output / "eval" / arm; (cell / "vbench").mkdir(parents=True)
        task.atomic_json(cell / "export.done", {"num_videos": 4720, "weight_source": "generator",
            "use_ema": False, "official_commit": task.OFFICIAL, "checkpoint_sha256": done["checkpoint_sha256"]})
        task.atomic_json(cell / "export_plan.json", {"protocol": manifest,
            "commands": task.export_commands(output, arm, sys.executable)})
        task.atomic_json(cell / f"vbench/{arm}_protocol.json", {"mode": "vbench_standard", "samples_per_prompt": 5})
        scores = {key.replace(" ", "_"): .7 if arm == "dmd_gan" else .8 for key in DIM_WEIGHT}
        task.atomic_json(cell / f"vbench/{arm}_eval_results.json", scores)
    assert len(calls) == 2 and all(kwargs["cwd"] == root and "PYTHONPATH" not in kwargs["env"] for _, kwargs in calls)
    result = task.report(output)
    assert [row["arm"] for row in result["runs"]] == ["dmd_gan", "dmd_only"]
    delta = result["gan_minus_dmd_percentage_points"]
    assert delta["total_score"] < 0  # A negative finding must be preserved, too.
    assert "frame_minus_chunk_percentage_points" not in result
    log = output / "runs/dmd_only/train.log"
    rows = metrics("dmd_only"); rows[0]["gan_g_loss"] = .01
    log.write_text("\n".join(repr(row) for row in rows))
    with pytest.raises(ValueError, match="nonzero GAN"):
        task.audit_training(output, "dmd_only", manifest)


def test_tmux_dispatches_gan_entry_without_touching_existing_sessions(pair, monkeypatch):
    _, output = pair
    original_run = subprocess.run
    tmux_calls = []
    def run(command, *args, **kwargs):
        if command[0] == "tmux":
            tmux_calls.append(command)
            return SimpleNamespace(returncode=1 if command[1] == "has-session" else 0)
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(task.subprocess, "run", run)
    monkeypatch.setattr(task, "require_idle", lambda gpus: None)
    args = SimpleNamespace(pair=str(output), arm="dmd_only", action="train", python=sys.executable,
        gpus="0,1,2,3,4,5,6,7", generator="/missing/init.pt", teacher="/missing/teacher",
        data="/missing/data", wan="/missing/wan", execute=True, session="")
    task.launch(args)
    assert [command[1] for command in tmux_calls] == ["has-session", "new-session"]
    assert ENTRY in tmux_calls[1][-1] and "--worker" in tmux_calls[1][-1]
    assert "official_gan300_train_dmd_only_s42" in tmux_calls[1]


def test_raw_checkpoint_rejects_a_step200_file_renamed_as_step300(tmp_path):
    initial, final = tmp_path / "init.pt", tmp_path / "model.pt"
    state = {"model.weight": torch.ones(1)}
    torch.save({"generator": state}, initial)
    torch.save({"generator": state, "step": 200}, final)
    with pytest.raises(ValueError, match="step does not match 300"):
        task.raw_checkpoint(final, initial, expected_step=300)
