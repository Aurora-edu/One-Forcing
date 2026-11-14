"""Real CPU optimizer-loop tests; fixtures below are not measured GPU results."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest
import torch

from experiments.iclr import matched_time as task
from experiments.iclr.protocol import config_dict, effective_gradient_frames, resume_signature, validate_config
from experiments.iclr.state import TrainingTimeClock
# Historical Wan evaluates a CUDA default argument at import. Mock only that
# lookup during CPU-test collection; production model code stays unchanged.
with patch("torch.cuda.current_device", return_value=0):
    from experiments.iclr import trainer as implementation
    from test_iclr_restart import ToyTrainer, cpu_runtime, set_cpu_seed, tensors_equal


ASSETS = {k: f"/missing/{k}" for k in ("generator", "teacher", "data")}


def test_historical_recipe_only_changes_stop_rule():
    frame = task.config_for(ASSETS, "frame")
    chunk = task.config_for(ASSETS, "chunk", 4261.134528471157)
    assert (frame.seed, chunk.seed) == (0, 0)
    assert [effective_gradient_frames(x) for x in (frame, chunk)] == [17, 21]
    assert frame.max_steps == 200 and chunk.max_steps == task.CAP
    for c in (frame, chunk):
        validate_config(c)
        assert c.gradient_num_blocks == 17 and c.r1_weight == c.r2_weight == 0
        assert c.gan_g_weight == c.gan_d_weight == .03
        assert c.lr == c.lr_critic == 1e-5
        assert c.iclr_eval_schedule == "main_ffe"
        assert c.iclr_eval_weight_source == "generator_ema"
        assert c.iclr_eval_samples_per_prompt == 5
        assert c.prompt_embedding_cache_path == ""  # old run did not enable a cache
    ignored = {"experiment_id", "num_frame_per_block", "max_steps", "iclr_stop_condition",
               "iclr_save_steps", "iclr_budget_seconds"}
    assert {k: v for k, v in config_dict(frame).items() if k not in ignored} == {
        k: v for k, v in config_dict(chunk).items() if k not in ignored}


@pytest.mark.parametrize("key,value", [("seed", 42), ("r1_weight", .1), ("lr", 2e-6),
    ("gradient_num_blocks", 6), ("ema_weight", .9), ("anchor_time_budget", 10)])
def test_config_guards(key, value):
    if key == "anchor_time_budget":
        # Frame's anchor may not itself have a time budget.
        c = task.config_for(ASSETS, "frame")
        c.iclr_budget_seconds = 10
    else:
        c = task.config_for(ASSETS, "chunk", 100)
        c[key] = value
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_budgets(budget):
    with pytest.raises(ValueError):
        task.config_for(ASSETS, "chunk", budget)


def timed_toy(path, budget=7.25):
    toy = ToyTrainer(path)
    toy.config.iclr_stop_condition = "training_seconds"
    toy.config.iclr_budget_seconds = budget
    toy.config.iclr_mode = "formal"
    toy.clock = TrainingTimeClock(10, (0, 5, 10), budget_seconds=budget)
    return toy


def test_real_loop_stops_at_time_and_saves_off_grid_ema_endpoint(cpu_runtime, tmp_path):
    toy = timed_toy(tmp_path / "timed")
    # Toy synchronized duration is 1s per iteration, save and overhead each.
    toy.train()
    assert toy.step == 8 and toy.clock.training_seconds == 8
    assert toy.clock.checkpoint_seconds == 3  # does NOT consume training budget
    assert toy.clock.overhead_seconds == 8
    assert toy.clock.saved_steps == [0, 5, 8]
    done = task.read(tmp_path / "timed/training.done")
    assert done["formal_result"] and done["time_budget_reached"]
    assert not done["step_budget_reached"]
    assert done["budget_overshoot_seconds"] == .75
    checkpoint = tmp_path / "timed/checkpoint_model_000008"
    assert task.read(checkpoint / "complete.json")["reason"] == "time_budget"
    payload = torch.load(checkpoint / "model.pt", weights_only=True)
    assert "generator_ema" in payload and payload["step"] == 8
    assert tensors_equal(payload["generator_ema"], toy.generator_ema.state_dict())


def test_step_cap_cannot_be_published_as_matched_time(cpu_runtime, tmp_path):
    toy = timed_toy(tmp_path / "cap", budget=20)
    toy.train()
    done = task.read(tmp_path / "cap/training.done")
    assert toy.step == 10 and done["step_budget_reached"]
    assert not done["formal_result"] and not done["time_budget_reached"]


def test_time_resume_keeps_optimizer_rng_ema_and_remaining_budget(cpu_runtime, tmp_path):
    set_cpu_seed(0)
    full = timed_toy(tmp_path / "full")
    full.train()
    set_cpu_seed(0)
    partial = timed_toy(tmp_path / "partial")
    save = partial.save
    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if partial.step == 5:
            raise InterruptedError()
    partial.save = interrupt
    with pytest.raises(InterruptedError):
        partial.train()
    set_cpu_seed(999)
    resumed = timed_toy(tmp_path / "resume")
    cp = tmp_path / "partial/checkpoint_model_000005"
    resumed.resume_dir, resumed.restart = implementation.read_complete_checkpoint(cp, resumed.config, 1)
    resumed.clock = TrainingTimeClock.from_state_dict(resumed.restart["clock"])
    resumed._preload_unwrapped_resume_from_checkpoint(cp)
    resumed.generator_optimizer = torch.optim.AdamW(resumed.model.generator.parameters(), lr=.01)
    resumed.critic_optimizer = torch.optim.AdamW(resumed.model.fake_score.parameters(), lr=.01)
    resumed._restore_training_state()
    resumed.train()
    assert resumed.step == 8 and resumed.clock.training_seconds == 8
    for a, b in ((full.model.generator, resumed.model.generator), (full.model.fake_score, resumed.model.fake_score),
                 (full.generator_ema, resumed.generator_ema), (full.generator_optimizer, resumed.generator_optimizer),
                 (full.critic_optimizer, resumed.critic_optimizer)):
        assert tensors_equal(a.state_dict(), b.state_dict())
    assert full.dataloader.state_dict() == resumed.dataloader.state_dict()


def fake_endpoint(output, arm, seconds, step, previous=None):
    """Metadata fixture only: no GPU run, no real scores."""
    run = output / "runs" / arm / "attempt_000"
    cp = run / f"checkpoint_model_{step:06d}"
    cp.mkdir(parents=True)
    (cp / "model.pt").write_bytes(b"cpu_fixture")
    cfg = OmegaConf.load(output / f"{arm}.yaml")
    OmegaConf.save(cfg, run / "resolved_config.yaml")
    done = {"formal_result": True, "step": step, "training_seconds": seconds}
    if arm == "chunk":
        done.update(time_budget_reached=True, budget_seconds=cfg.iclr_budget_seconds)
    task.atomic_json(run / "training.done", done)
    task.atomic_json(cp / "complete.json", {"config": resume_signature(cfg),
                     "clock": done, "files": {"model.pt": (cp / "model.pt").stat().st_size}})
    rows = [{"training_seconds": previous or seconds - 1}, {"training_seconds": seconds}]
    (run / "training_clock.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, cp


def test_budget_is_measured_and_frozen_before_evaluation(tmp_path):
    output = tmp_path / "task"
    task.prepare(output, SimpleNamespace(**ASSETS))
    assert not (output / "chunk.yaml").exists()
    with pytest.raises(FileNotFoundError):
        task.freeze_budget(output)
    fake_endpoint(output, "frame", seconds=1234.5, step=200)
    budget = task.freeze_budget(output)
    assert budget["training_seconds"] == 1234.5  # NOT historical 4261.13
    assert OmegaConf.load(output / "chunk.yaml").iclr_budget_seconds == 1234.5
    task.check(output)
    task.freeze_budget(output)  # identical reentry preserves the budget
    fake_endpoint(output, "chunk", seconds=1238, step=267, previous=1230)
    assert task.endpoint(output, "chunk")[2]["step"] == 267
    args = SimpleNamespace(gpus="0,1,2,3,4,5,6,7", python="PY", vbench_python="VB")
    for arm, method in (("frame", "framewise"), ("chunk", "chunkwise")):
        ea = task.eval_args(output, arm, args)
        command = task.evaluation.command_for_condition(ea, "qwen")
        assert "--use_ema" in command and "main_ffe" in command
        assert command[command.index("--method") + 1] == method
        assert command[command.index("--samples_per_prompt") + 1] == "5"
        assert command[command.index("--checkpoint_path") + 1].endswith(
            f"checkpoint_model_{200 if arm == 'frame' else 267:06d}/model.pt")
    fake_endpoint(output, "chunk", seconds=1250, step=268, previous=1238)
    with pytest.raises(ValueError, match="first complete"):
        task.endpoint(output, "chunk")


def test_partial_outputs_are_preserved_and_only_complete_states_resume(tmp_path):
    output = tmp_path / "task"
    first, resume = task.next_attempt(output, "chunk")
    assert first.name == "attempt_000" and not resume
    cp = first / "checkpoint_model_000050"
    cp.mkdir(parents=True)
    task.atomic_json(cp / "complete.json", {"clock": {"step": 50}})
    (first / "checkpoint_model_000100").mkdir()  # partial save must not resume
    second, resume = task.next_attempt(output, "chunk")
    assert second.name == "attempt_001" and Path(resume) == cp


def test_monitor_requires_ten_real_finite_iterations_and_gan_logs(tmp_path):
    assert not task.monitor(tmp_path, "frame")["step10_observed"]
    run = tmp_path / "runs/frame/attempt_000"
    run.mkdir(parents=True)
    rows = []
    for step in range(1, 11):
        row = {"step": step, "critic_loss": .2, "fake_score_loss": .3,
               "gan_d_loss": -.1, "critic_grad_norm": 1., "r1_loss": 0., "r2_loss": 0.}
        if (step - 1) % 5 == 0:
            row.update(generator_loss=.2, dmd_loss=.1, gan_g_loss=.1, generator_grad_norm=2.)
        rows.append(row)
    path = run / "metrics.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[:9]) + '{"step":')
    assert not task.monitor(tmp_path, "frame")["step10_observed"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert task.monitor(tmp_path, "frame")["step10_observed"]
    assert (run / "observed_step10.json").exists()
    rows[-1]["critic_grad_norm"] = float("nan")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="nonfinite"):
        task.monitor(tmp_path, "frame")


def test_hardware_or_asset_receipt_cannot_silently_change(tmp_path):
    path = tmp_path / "receipt.json"
    task.immutable_json(path, {"gpu": "A", "data": "X"})
    task.immutable_json(path, {"gpu": "A", "data": "X"})
    with pytest.raises(ValueError, match="Immutable"):
        task.immutable_json(path, {"gpu": "B", "data": "X"})


def test_report_uses_both_new_scores_and_labels_recovery(tmp_path, monkeypatch):
    output = tmp_path / "task"
    task.prepare(output, SimpleNamespace(**ASSETS))
    fake_endpoint(output, "frame", 1000, 200)
    task.freeze_budget(output)
    fake_endpoint(output, "chunk", 1002, 251, previous=997)
    task.atomic_json(output / "hardware.json", {"cpu_fixture_only": True})
    for arm in task.ARMS:
        run, _, done = task.endpoint(output, arm)
        done.update(generator_updates=(done["step"] + 4) // 5, critic_updates=done["step"],
                    checkpoint_seconds=20., overhead_seconds=1.)
        task.atomic_json(run / "training.done", done)
        task.atomic_json(run / "initialization.json", {"generator_checkpoint_sha256": task.INIT_SHA256,
            "generator_tensor_sha256": "same_init", "critic_tensor_sha256": "same_critic", "initialization_seed": 0})
        task.atomic_json(run / "launch.json", {"resume": "", "process_wall_seconds": 1050.})
    args = SimpleNamespace(gpus="0,1,2,3,4,5,6,7", python="PY", vbench_python="VB")
    # Existing prompt_ablation tests separately audit real export/scoring metadata.
    def synthetic_scores(ea):
        total = .78 if ea.method == "framewise" else .79
        return {"scores": [{"total_score": total, "quality_score": .8, "semantic_score": .7}]}
    monkeypatch.setattr(task.evaluation, "summarize", synthetic_scores)
    task.report(output, args)
    result = task.read(output / "matched_time_results.json")
    assert result["uninterrupted_pair"]
    assert result["frame_minus_chunk"]["total_score"] == -1  # do not substitute historical 83.45
    assert [r["step"] for r in result["rows"]] == [200, 251]
    task.atomic_json(output / "runs/chunk/attempt_000/launch.json", {"resume": "/fixture/restart", "process_wall_seconds": 1050.})
    task.report(output, args)
    result = task.read(output / "matched_time_results.json")
    assert not result["uninterrupted_pair"] and result["recovery_note"]
