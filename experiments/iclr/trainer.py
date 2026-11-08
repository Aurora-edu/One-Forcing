"""ICLR-only trainer: strict raw init, fixed 200 iterations and full restarts.

Legacy train.py never imports this module. Checkpoints support the same world
size, FSDP topology, software and deterministic LMDB dataset only.
"""
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from experiments.iclr.protocol import (
    atomic_json, effective_gradient_frames, resume_signature, sha256_file, tensor_state_sha256, validate_config,
)
from experiments.iclr.state import RestartableStream, TrainingClock
from trainer.one_forcing import Trainer, _move_optimizer_state, _normalize_state_dict_keys
from utils.distributed import EMA_FSDP, fsdp_state_dict


def load_raw(path):
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "generator" in payload:
        source, state = "generator", payload["generator"]
    elif "model" in payload:
        source, state = "model", payload["model"]
    elif payload and all(torch.is_tensor(v) for v in payload.values()):
        source, state = "raw_state_dict", payload
    else:
        raise ValueError("ICLR initialization requires raw generator tensors; EMA fallback is forbidden")
    return path, source, _normalize_state_dict_keys(state)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(torch.cuda.current_device())}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], torch.cuda.current_device())


def atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_complete_checkpoint(directory, config, world_size):
    directory = Path(directory)
    if directory.is_file():
        directory = directory.parent
    manifest = json.loads((directory / "complete.json").read_text())
    if manifest.get("schema_version") != 3:
        raise ValueError("unsupported ICLR checkpoint")
    if manifest["world_size"] != world_size:
        raise ValueError("ICLR restart requires identical world size/FSDP sharding")
    if manifest["config"] != resume_signature(config):
        raise ValueError("ICLR restart config changed; weights-only resume is not supported")
    if manifest["torch_version"] != torch.__version__:
        raise ValueError("ICLR restart requires identical PyTorch version")
    for name, size in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"incomplete ICLR checkpoint: {path}")
    expected = {"model.pt"} | {f"rank_{i:03d}.pt" for i in range(world_size)}
    if set(manifest["files"]) != expected:
        raise ValueError("missing model or per-rank optimizer/RNG/data restart state")
    return directory, manifest


class ICLRTrainer(Trainer):
    def __init__(self, config):
        validate_config(config)
        self.streams = []
        self.initialization = {}
        self.restart = None
        if config.resume_ckpt:
            self.resume_dir, self.restart = read_complete_checkpoint(
                config.resume_ckpt, config, int(os.environ["WORLD_SIZE"]))
            self.initialization = self.restart["initialization"]
        self.clock = (TrainingClock.from_state_dict(self.restart["clock"]) if self.restart
                      else TrainingClock(config.max_steps, tuple(config.iclr_save_steps)))
        super().__init__(config)
        if self.world_size != int(config.iclr_expected_world_size):
            raise ValueError("world size differs from matched ICLR protocol")
        if self.model._resolve_gradient_num_frames(21) != effective_gradient_frames(config):
            raise ValueError("gradient window differs from main's unchanged block-count policy")
        if self.restart:
            self._restore_training_state()
        else:
            self.step = 0
        if self.step != self.clock.step:
            raise ValueError("checkpoint model step and clock step disagree")
        if self.is_main_process:
            atomic_json(Path(self.output_path) / "initialization.json", self.initialization)

    def _build_model(self, config):
        # Preserve main's seed+rank model initialization; pair identities within each seed.
        model = super()._build_model(config)
        if self.is_main_process and not self.restart:
            self.initialization = {
                "initialization_seed": int(config.seed),
                "experiment_id": str(config.experiment_id),
                "critic_tensor_sha256": tensor_state_sha256(model.fake_score.state_dict()),
                "teacher_model_path": str(Path(config.teacher_model_path).resolve()),
                "ode_provenance": "UNVERIFIED: content identity does not establish ODE training provenance",
            }
        return model

    def _preload_unwrapped_generator_from_checkpoint(self, path):
        if not self.is_main_process:
            return
        source_path, source, state = load_raw(path)
        self.model.generator.load_state_dict(state, strict=True, assign=True)
        self.initialization.update({
            "generator_checkpoint": str(source_path.resolve()),
            "generator_checkpoint_sha256": sha256_file(source_path),
            "generator_source_key": source,
            "generator_tensor_sha256": tensor_state_sha256(self.model.generator.state_dict()),
            "strict_load": True, "use_ema": False,
        })

    def _preload_unwrapped_resume_from_checkpoint(self, path):
        if self.is_main_process:
            payload = torch.load(self.resume_dir / "model.pt", map_location="cpu",
                                 mmap=True, weights_only=True)
            for key, module in (("generator", self.model.generator), ("critic", self.model.fake_score)):
                module.load_state_dict(_normalize_state_dict_keys(payload[key]), strict=True, assign=True)
        self.step = int(self.restart["clock"]["step"])

    def _load_generator_checkpoint(self, path):
        # The legacy constructor calls this after resume preloading because it
        # leaves _generator_ckpt_loaded_before_fsdp=False. Never overwrite the
        # resumed generator with its initialization.
        if self.restart:
            return
        raise RuntimeError("ICLR requires strict raw rank-0 preloading before FSDP")

    def _build_dataloader(self, dataset):
        stream = RestartableStream(dataset, batch_size=self.config.batch_size,
                                   seed=0, rank=dist.get_rank(),
                                   world_size=dist.get_world_size(), fixed_sampler_epoch=True)
        self.streams.append(stream)
        return stream

    def _restore_training_state(self):
        rank = dist.get_rank()
        # These are our own trusted files, containing NumPy/Python RNG tuples.
        saved = torch.load(self.resume_dir / f"rank_{rank:03d}.pt", map_location="cpu", weights_only=False)
        if saved["rank"] != rank or saved["step"] != self.clock.step:
            raise ValueError("per-rank state belongs to another checkpoint")
        if saved["gpu_name"] != torch.cuda.get_device_name(self.device):
            raise ValueError("resume GPU type changed")
        self.generator_optimizer.load_state_dict(saved["generator_optimizer"])
        self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        if getattr(self.config, "generator_optimizer_state_cpu_offload", False):
            _move_optimizer_state(self.generator_optimizer, "cpu")
        ema_state = saved["generator_ema"]
        if ema_state is not None:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
            self.generator_ema.load_state_dict(ema_state)
        elif self.clock.step >= self.config.ema_start_step and self.config.ema_weight > 0:
            raise ValueError("checkpoint is missing the active per-rank EMA state")
        else:
            self.generator_ema = None
        if len(saved["streams"]) != len(self.streams):
            raise ValueError("data stream count changed")
        for stream, state in zip(self.streams, saved["streams"]):
            stream.load_state_dict(state)
        restore_rng(saved["rng"])

    def _synchronized_seconds(self, started):
        torch.cuda.synchronize(self.device)
        elapsed = torch.tensor(time.perf_counter() - started, device=self.device, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return float(elapsed.item())

    def _require_finite(self, logs, phase):
        bad = [key for key, value in logs.items() if torch.is_tensor(value)
               and not bool(torch.isfinite(value).all())]
        flag = torch.tensor(int(bool(bad)), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            print(f"NONFINITE rank={dist.get_rank()} step={self.step} phase={phase} keys={bad}", flush=True)
            if self.is_main_process:
                atomic_json(Path(self.output_path) / "nonfinite_failure.json",
                            {"step": self.step, "phase": phase, "rank0_bad_keys": bad,
                             "message": "one or more ranks have nonfinite losses/gradient norms"})
            raise FloatingPointError(f"nonfinite {phase} loss/gradient on at least one rank")

    def save(self, milestones=(), reason="step"):
        output = Path(self.output_path) / f"checkpoint_model_{self.step:06d}"
        started = time.perf_counter()
        if self.is_main_process:
            output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        # Capture at an iteration boundary, after optimizer zero_grad, on every rank.
        rng = capture_rng()
        generator = fsdp_state_dict(self.model.generator)
        critic = fsdp_state_dict(self.model.fake_score)
        ema = self.generator_ema.full_state_dict(self.model.generator) if self.generator_ema is not None else None
        rank = dist.get_rank()
        atomic_torch_save(output / f"rank_{rank:03d}.pt", {
            "rank": rank, "step": self.step, "gpu_name": torch.cuda.get_device_name(self.device),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "generator_ema": self.generator_ema.state_dict() if self.generator_ema is not None else None,
            "rng": rng, "streams": [s.state_dict() for s in self.streams],
        })
        if self.is_main_process:
            payload = {"step": self.step,
                "generator": generator, "critic": critic,
                "experiment_metadata": {"contains_generator_ema": ema is not None,
                    "profile": self.config.iclr_profile, "mode": self.config.iclr_mode}}
            if ema is not None:
                payload["generator_ema"] = ema
            atomic_torch_save(output / "model.pt", payload)
        del generator, critic, ema
        dist.barrier()
        self.clock.checkpoint_seconds += self._synchronized_seconds(started)
        if self.is_main_process:
            names = ["model.pt", *(f"rank_{i:03d}.pt" for i in range(self.world_size))]
            manifest = {"schema_version": 3, "world_size": self.world_size,
                "torch_version": torch.__version__, "config": resume_signature(self.config),
                "clock": self.clock.state_dict(), "initialization": self.initialization,
                "reason": reason, "milestones": list(milestones),
                "files": {name: (output / name).stat().st_size for name in names}}
            atomic_json(output / "complete.json", manifest)
            with (Path(self.output_path) / "checkpoints.jsonl").open("a") as stream:
                stream.write(json.dumps({"checkpoint": str(output / "model.pt"),
                    "experiment_id": self.config.experiment_id, "seed": int(self.config.seed),
                    "r1_weight": float(self.config.r1_weight), "r2_weight": float(self.config.r2_weight),
                    "mode": self.config.iclr_mode,
                    "profile": self.config.iclr_profile,
                    "stop_condition": self.config.iclr_stop_condition,
                    "gradient_num_blocks": self.config.gradient_num_blocks,
                    "gradient_num_frames": int(self.config.get("gradient_num_frames", 0)),
                    "effective_gradient_frames": effective_gradient_frames(self.config),
                    "method": "framewise" if self.config.num_frame_per_block == 1 else "chunkwise",
                    "schedule": "all1", "weight_source": "generator", "reason": reason,
                    "available_weights": ["generator"] + (["generator_ema"] if self.generator_ema is not None else []),
                    **self.clock.state_dict(), "milestones": list(milestones)}) + "\n")
        dist.barrier()
        restore_rng(rng)

    def train(self):
        due = self.clock.due()
        if due:
            self.clock.saved_steps.extend(due)
            self.save(due, reason="initial")
        while self.clock.should_continue():
            dist.barrier()
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            train_generator = self.step % self.config.dfake_gen_update_ratio == 0
            generator_logs = {}
            if train_generator:
                self.model.generator.requires_grad_(True)
                self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                generator_logs = self.fwdbwd_one_step(next(self.dataloader), train_generator=True)
                self._require_finite(generator_logs, "generator")
                if getattr(self.config, "generator_optimizer_state_cpu_offload", False):
                    _move_optimizer_state(self.generator_optimizer, self.device)
                self.generator_optimizer.step()
                if getattr(self.config, "generator_optimizer_state_cpu_offload", False):
                    _move_optimizer_state(self.generator_optimizer, "cpu")
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                self.generator_optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_logs = self.fwdbwd_one_step(next(self.dataloader), train_generator=False)
            self._require_finite(critic_logs, "critic")
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            # Match main: initialize EMA after the iteration reaching ema_start_step;
            # subsequent generator optimizer steps update it before the critic update.
            if (self.clock.step + 1 >= self.config.ema_start_step
                    and self.generator_ema is None and self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
            seconds = self._synchronized_seconds(started)
            self.clock.advance(seconds, generator=train_generator)
            self.step = self.clock.step
            overhead_started = time.perf_counter()
            self._log_step(train_generator, generator_logs, critic_logs, iteration_time=seconds)
            if self.is_main_process:
                with (Path(self.output_path) / "training_clock.jsonl").open("a") as stream:
                    stream.write(json.dumps(self.clock.state_dict()) + "\n")
            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
            self.clock.overhead_seconds += self._synchronized_seconds(overhead_started)
            due = self.clock.due()
            if due:
                self.clock.saved_steps.extend(due)
                self.save(due, reason="step_cap" if self.clock.exhausted else "step")
        if self.is_main_process:
            atomic_json(Path(self.output_path) / "training.done", {
                **self.clock.state_dict(), "mode": self.config.iclr_mode,
                "stop_condition": "fixed_steps", "step_budget_reached": self.clock.exhausted,
                "formal_result": self.config.iclr_mode == "formal" and self.clock.exhausted
                                 and self.step == self.config.iclr_reference_step})
