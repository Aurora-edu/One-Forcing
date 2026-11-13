"""CPU checks of historical source/config isolation, continuation and reporting."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from experiments.iclr import historical_dmd400 as task
from experiments.rebuttal.consolidate_results import DIM_WEIGHT, official_totals


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    output = tmp_path_factory.mktemp("historical_dmd400") / "task"
    task.prepare(output)
    yield output
    # Only worktrees created by this fixture; no user checkout or job is touched.
    for name in ("training", "evaluation"):
        subprocess.run(["git", "-C", str(task.REPO), "worktree", "remove",
                        str(output / name)], check=True, capture_output=True)


def args():
    return SimpleNamespace(generator="/missing/ode.pt", teacher="/missing/teacher",
        data="/missing/clean_data", prompt_cache="/missing/cache", python=sys.executable,
        gpus="0,1,2,3,4,5,6,7", session="test_dmd400", vbench_python="/missing/vbench/python")


def test_pinned_recipe_uses_literal_zero_and_only_extends_budget(prepared):
    cfg = task.read_config(prepared / "dmd400.yaml")
    old = task.read_config(prepared / "training/experiments/rebuttal/configs/train_1step_dmd_only.yaml")
    assert cfg == dict(old, max_steps=400)
    assert cfg["seed"] == 0 and cfg["randomize_seed"] is False
    assert cfg["gan_g_weight"] == cfg["gan_d_weight"] == 0
    assert cfg["num_frame_per_block"] == 1 and cfg["gradient_num_blocks"] == 17
    assert cfg["rollout_schedule"] == "fixed" and cfg["denoising_step_list"] == [1000]
    assert cfg["resume_ckpt"] == "" and cfg["dfake_gen_update_ratio"] == 5
    task.check(prepared)


def test_historical_training_and_legacy_inference_are_separate(prepared):
    command = task.train_command(prepared, args())
    assert str(prepared / "training/experiments/rebuttal/launch_train.sh") in command
    assert command[command.index("--max_steps") + 1] == "400"
    assert "--resume_ckpt" not in command and set(task.TRAIN_FLAGS) <= set(command)
    evaluate = task.eval_command(prepared, args())
    assert "--use_ema" not in evaluate and evaluate[evaluate.index("--schedule") + 1] == "ffe"
    assert "checkpoint_model_000400/model.pt" in " ".join(evaluate)
    cfg = task.read_config(prepared / "evaluation/experiments/rebuttal/configs/eval_ffe.yaml")
    assert cfg["first_rollout_num_frames"] == 4 and cfg["rollout_schedule"] == "first4then1"
    preview = task.eval_command(prepared, args(), preview=True)
    assert "--use_ema" not in preview
    records = [json.loads(s) for s in (prepared / "inputs/preview.jsonl").read_text().splitlines()]
    assert len(records) == 8 and all(r["seed"] == r["prompt_index"] * 5 for r in records)


def test_dry_run_does_not_require_assets_or_touch_gpu(prepared):
    command = [sys.executable, "-m", "experiments.iclr.historical_dmd400", "train",
        "--output", str(prepared), "--generator", "/missing/ode.pt", "--teacher", "/missing/teacher",
        "--data", "/missing/data", "--prompt-cache", "/missing/cache", "--wan-root", "/missing/wan"]
    result = subprocess.run(command, cwd=task.REPO, capture_output=True, text=True, check=True)
    assert "--max_steps 400" in result.stdout and "--seed 0" in result.stdout
    assert not (prepared / "launch.json").exists() and not task.run_dir(prepared).exists()


@pytest.mark.parametrize("key,value", [("seed", 42), ("max_steps", 200),
    ("resume_ckpt", "/old200.pt"), ("gan_g_weight", .03)])
def test_rejects_changed_recipe_even_if_checksum_updated(prepared, key, value):
    path = prepared / "dmd400.yaml"
    original = path.read_text()
    try:
        cfg = task.read_config(path); cfg[key] = value
        OmegaConf.save(OmegaConf.create(cfg), path)
        with pytest.raises(ValueError, match="historical recipe"):
            task.check(prepared)
    finally:
        path.write_text(original)


def install_fake_training(output, count):
    run = task.run_dir(output); run.mkdir(parents=True, exist_ok=True)
    command = task.train_command(output, args())
    task.write_json(output / "launch.json", {"command": command})
    cfg = task.read_config(output / "dmd400.yaml")
    for key, flag in (("generator_ckpt", "--generator_ckpt"), ("teacher_model_path", "--teacher_model_path"),
                      ("data_path", "--data_path"), ("prompt_embedding_cache_path", "--prompt_embedding_cache_path")):
        cfg[key] = command[command.index(flag) + 1]
    cfg.update(fake_score_cpu_offload=True, manual_generator_backward=True,
               generator_optimizer_state_cpu_offload=True, rank0_preload_generator_ckpt=True)
    OmegaConf.save(OmegaConf.create(cfg), run / "resolved_config.yaml")
    (run / "runtime_seed.txt").write_text("0\n")
    task.write_json(run / "run_metadata.json", {"git_commit": task.TRAIN_COMMIT, "git_status_porcelain": []})
    records = [{"step": i, "critic_loss": .2, "fake_score_loss": .2,
                "gan_d_loss": 0., "r1_loss": 0., "r2_loss": 0.,
                **({"generator_loss": .1, "dmd_loss": .1, "gan_g_loss": 0.} if (i - 1) % 5 == 0 else {})}
               for i in range(1, count + 1)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if count == 400:
        task.write_json(run / "training.done", {"final_step": 400, "max_steps": 400})
        for step in (100, 200, 300, 400):
            path = run / f"checkpoint_model_{step:06d}/model.pt"; path.parent.mkdir(exist_ok=True)
            torch.save({"step": step, "generator": {"model.weight": torch.ones(2)}}, path)
    return records


def test_health_checks_continuous_steps_zero_gan_and_actual_checkpoint(prepared):
    rows = install_fake_training(prepared, 10)
    assert task.audit_training(prepared, final=False)["generator_updates"] == 2
    rows[3]["gan_d_loss"] = .01
    path = task.run_dir(prepared) / "metrics.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="Nonzero GAN"):
        task.audit_training(prepared, final=False)
    rows = install_fake_training(prepared, 400)
    assert task.audit_training(prepared, final=True)["generator_updates"] == 80
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[200:]))
    with pytest.raises(ValueError, match="continuous"):
        task.audit_training(prepared, final=True)
    install_fake_training(prepared, 400)
    checkpoint = task.run_dir(prepared) / "checkpoint_model_000400/model.pt"
    torch.save({"step": 200, "generator": {"model.weight": torch.ones(2)}}, checkpoint)
    with pytest.raises(ValueError, match="step mismatch"):
        task.audit_training(prepared, final=True)
    install_fake_training(prepared, 400)


def test_result_recomputes_scores_and_retains_negative_gan_gain(prepared):
    install_fake_training(prepared, 400)
    scores = {key.replace(" ", "_"): .99 for key in DIM_WEIGHT}
    summary = {"status": "pass", "protocol": task.protocol.audit_inputs(prepared / "inputs", 5),
               "scores": scores, "normalized_aggregates": official_totals(scores),
               "weight_source": "generator", "method": "framewise", "schedule": "ffe",
               "checkpoint_path": str(task.run_dir(prepared) / "checkpoint_model_000400/model.pt")}
    path = prepared / "eval/dmd400_ffe/dmd400_ffe_paper_protocol_summary.json"
    task.write_json(path, summary)
    result = task.report(prepared)
    assert result["gan_reference"]["total_score"] == 83.93
    assert result["gan_minus_dmd_percentage_points"]["total_score"] < 0
    assert "rounded" in result["reference_precision"]
    summary["weight_source"] = "generator_ema"; task.write_json(path, summary)
    with pytest.raises(ValueError, match="raw framewise"):
        task.report(prepared)
    summary["weight_source"] = "generator"
    summary["protocol"]["manifest_sha256"] = "wrong"; task.write_json(path, summary)
    with pytest.raises(ValueError, match="historical Qwen"):
        task.report(prepared)


def test_rejects_wrong_gpu_batch():
    for ids in ("0", "0,1,2,3", "0,1,2,3,4,5,6,6"):
        with pytest.raises(ValueError, match="eight"):
            task.eight_gpus(ids)


def test_busy_gpu_cannot_start_and_tmux_never_stops_sessions(monkeypatch, tmp_path):
    monkeypatch.setattr(task.subprocess, "check_output", lambda *a, **k: "12345\n")
    with pytest.raises(RuntimeError, match="occupied"):
        task.require_idle("0,1,2,3,4,5,6,7")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(task.subprocess, "run", run)
    with pytest.raises(FileExistsError, match="tmux session exists"):
        task.start_tmux(["python", "task.py"], tmp_path, "existing", tmp_path / "log")
    assert calls == [["tmux", "has-session", "-t", "existing"]]
