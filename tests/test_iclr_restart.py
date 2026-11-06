"""Exercise the real ICLR loop/checkpoint/restart code with tiny CPU models.

FSDP collectives are replaced by one-rank CPU equivalents; these tests verify
optimizer/RNG/data continuation and do not claim GPU/FSDP validation.
"""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.iclr import trainer as implementation
from experiments.iclr.protocol import REPO
from experiments.iclr.make_configs import make_matrix
from experiments.iclr.state import RestartableStream, TrainingClock
from utils.config import load_config


@pytest.fixture
def cpu_runtime(monkeypatch):
    monkeypatch.setattr(implementation.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(implementation.dist, "barrier", lambda: None)
    monkeypatch.setattr(implementation.dist, "all_reduce", lambda tensor, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU_TEST")
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *args: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *args: None)
    monkeypatch.setattr(implementation, "fsdp_state_dict", lambda module: module.state_dict())


class ToyTrainer(implementation.ICLRTrainer):
    def __init__(self, output):
        base = load_config(str(REPO / "experiments/iclr/configs/base.yaml"))
        self.config = make_matrix(base, generator_ckpt="/missing/ode.pt", teacher="/missing/teacher",
            data="/missing/data", prompt_cache="", budget_seconds=20, smoke=True)[0]
        self.output_path = str(output)
        output.mkdir(parents=True, exist_ok=True)
        self.config.logdir = self.output_path
        self.device = torch.device("cpu")
        self.world_size = 1
        self.is_main_process = True
        self.disable_wandb = True
        self.max_steps = 10
        self.step = 0
        self.restart = None
        self.initialization = {"generator_tensor_sha256": "toy", "critic_tensor_sha256": "toy"}
        self.clock = TrainingClock(20)
        self.model = SimpleNamespace(generator=torch.nn.Linear(2, 2), fake_score=torch.nn.Linear(2, 2))
        self.generator_optimizer = torch.optim.AdamW(self.model.generator.parameters(), lr=.01)
        self.critic_optimizer = torch.optim.AdamW(self.model.fake_score.parameters(), lr=.01)
        self.dataloader = RestartableStream(list(range(13)), batch_size=2, seed=0)
        self.streams = [self.dataloader]

    def _synchronized_seconds(self, started):
        return 1.

    def fwdbwd_one_step(self, batch, train_generator):
        module = self.model.generator if train_generator else self.model.fake_score
        x = torch.randn(len(batch), 2) + float(random.random()) + float(np.random.rand())
        target = batch.float().reshape(-1,1) / 13
        loss = (module(x) - target).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 10)
        prefix = "generator" if train_generator else "critic"
        return {prefix + "_loss": loss.detach(), prefix + "_grad_norm": norm.detach()}


def set_cpu_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensors_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensors_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list,tuple)):
        return len(left) == len(right) and all(tensors_equal(a,b) for a,b in zip(left,right))
    return left == right


def test_real_loop_full_state_restart_equals_uninterrupted_and_never_reloads_init(cpu_runtime, tmp_path):
    set_cpu_seed(71)
    full = ToyTrainer(tmp_path/"full")
    full.train()
    expected_rng = torch.get_rng_state().clone()
    expected_np = np.random.rand()
    expected_py = random.random()
    set_cpu_seed(71)
    interrupted = ToyTrainer(tmp_path/"interrupted")
    save = interrupted.save
    def stop_at_five(*args, **kwargs):
        save(*args, **kwargs)
        if interrupted.step == 5:
            raise InterruptedError("simulated interruption after committed checkpoint")
    interrupted.save = stop_at_five
    with pytest.raises(InterruptedError):
        interrupted.train()
    assert not (tmp_path/"interrupted"/"training.done").exists()
    checkpoint = tmp_path/"interrupted"/"checkpoint_model_000005"
    set_cpu_seed(999)
    resumed = ToyTrainer(tmp_path/"resumed")
    resumed.resume_dir, resumed.restart = implementation.read_complete_checkpoint(checkpoint, resumed.config, 1)
    resumed.clock = TrainingClock.from_state_dict(resumed.restart["clock"])
    resumed._preload_unwrapped_resume_from_checkpoint(checkpoint)
    # Production preloads with assign=True BEFORE FSDP and optimizer creation.
    # Match that order: otherwise optimizers still reference replaced toy parameters.
    resumed.generator_optimizer = torch.optim.AdamW(resumed.model.generator.parameters(), lr=.01)
    resumed.critic_optimizer = torch.optim.AdamW(resumed.model.fake_score.parameters(), lr=.01)
    # Regression: the parent constructor calls this AFTER resume preloading.
    wrong_init = tmp_path/"wrong_init.pt"
    torch.save({"generator": {k: torch.zeros_like(v) for k,v in resumed.model.generator.state_dict().items()}}, wrong_init)
    before = {k: v.clone() for k,v in resumed.model.generator.state_dict().items()}
    resumed._load_generator_checkpoint(wrong_init)
    assert tensors_equal(before, resumed.model.generator.state_dict())
    resumed._restore_training_state()
    resumed.train()
    assert tensors_equal(full.model.generator.state_dict(), resumed.model.generator.state_dict())
    assert tensors_equal(full.model.fake_score.state_dict(), resumed.model.fake_score.state_dict())
    assert tensors_equal(full.generator_optimizer.state_dict(), resumed.generator_optimizer.state_dict())
    assert tensors_equal(full.critic_optimizer.state_dict(), resumed.critic_optimizer.state_dict())
    assert full.dataloader.state_dict() == resumed.dataloader.state_dict()
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert expected_np == np.random.rand()
    assert expected_py == random.random()
    assert (resumed.clock.step, resumed.clock.critic_updates, resumed.clock.generator_updates) == (10,10,2)
    assert full.clock.training_seconds == resumed.clock.training_seconds == 10


def test_weights_only_incomplete_or_different_topology_resume_is_rejected(cpu_runtime, tmp_path):
    toy = ToyTrainer(tmp_path/"toy")
    toy.save()
    directory = tmp_path/"toy"/"checkpoint_model_000000"
    with pytest.raises(ValueError, match="world size"):
        implementation.read_complete_checkpoint(directory, toy.config, 8)
    changed = toy.config.copy()
    changed.lr = .005
    with pytest.raises(ValueError, match="config changed"):
        implementation.read_complete_checkpoint(directory, changed, 1)
    (directory/"rank_000.pt").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        implementation.read_complete_checkpoint(directory, toy.config, 1)
    (directory/"complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        implementation.read_complete_checkpoint(directory, toy.config, 1)
    with pytest.raises(FileExistsError):
        toy.save()  # Never overwrite a partially written checkpoint directory.


def test_raw_loader_never_falls_back_to_ema(cpu_runtime, tmp_path):
    path = tmp_path/"ema_only.pt"
    torch.save({"generator_ema": {"weight": torch.ones(2)}}, path)
    with pytest.raises(ValueError, match="EMA fallback"):
        implementation.load_raw(path)


def test_nonfinite_training_cannot_publish_success(cpu_runtime, tmp_path):
    trainer = ToyTrainer(tmp_path/"nan")
    trainer.model.fake_score.weight.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite critic"):
        trainer.train()
    assert not (tmp_path/"nan"/"training.done").exists()
    assert (tmp_path/"nan"/"nonfinite_failure.json").exists()


def test_nonfinite_on_other_rank_stops_this_rank(cpu_runtime, monkeypatch, tmp_path):
    trainer = ToyTrainer(tmp_path/"remote_nan")
    monkeypatch.setattr(implementation.dist, "all_reduce", lambda flag, **kwargs: flag.fill_(1))
    with pytest.raises(FloatingPointError):
        trainer._require_finite({"critic_loss": torch.tensor(1.)}, "critic")
