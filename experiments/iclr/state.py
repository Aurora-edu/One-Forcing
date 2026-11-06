"""Fixed iteration budget, elapsed-time counters and restartable data order."""
from dataclasses import asdict, dataclass, field
import itertools
import math


@dataclass
class TrainingClock:
    max_steps: int = 200
    save_steps: tuple = (0, 50, 100, 150, 200)
    training_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    overhead_seconds: float = 0.0
    critic_updates: int = 0
    generator_updates: int = 0
    step: int = 0
    saved_steps: list = field(default_factory=list)

    def __post_init__(self):
        self.save_steps = tuple(self.save_steps)
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (tuple(sorted(set(self.save_steps))) != self.save_steps
                or not self.save_steps or self.save_steps[0] != 0 or self.save_steps[-1] != self.max_steps):
            raise ValueError("save_steps must increase strictly from zero to max_steps")
        if any(not isinstance(x, int) or not 0 <= x <= self.max_steps for x in self.save_steps):
            raise ValueError("save_steps must be integer iteration boundaries")

    @property
    def exhausted(self):
        return self.step >= self.max_steps

    def advance(self, seconds, *, generator, critic=True):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid synchronized training duration")
        self.training_seconds += seconds
        self.step += 1
        self.critic_updates += int(critic)
        self.generator_updates += int(generator)

    def due(self):
        return [x for x in self.save_steps if x not in self.saved_steps and self.step >= x]

    def should_continue(self):
        return not self.exhausted

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


class RestartableStream:
    """Deterministic epoch permutation, resumable at the next unconsumed batch.

    No model RNG is consumed by DataLoader initialization. Only deterministic
    datasets with num_workers=0 are supported by the ICLR protocol.
    """
    def __init__(self, dataset, *, batch_size, seed, rank=0, world_size=1, fixed_sampler_epoch=False):
        import torch
        self.dataset = dataset
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
        self.fixed_sampler_epoch = bool(fixed_sampler_epoch)
        self.epoch = self.batch = 0
        self._iterator = None
        self.sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            seed=seed, shuffle=True, drop_last=True)
        self.batches_per_epoch = len(self.sampler) // batch_size
        if self.batches_per_epoch < 1:
            raise ValueError("dataset has no complete distributed batch")

    def __iter__(self):
        return self

    def __next__(self):
        import torch
        if self.batch == self.batches_per_epoch:
            self.epoch += 1
            self.batch = 0
            self._iterator = None
        if self._iterator is None:
            # main's cycle() never calls set_epoch; retain that ordering for the main profile.
            self.sampler.set_epoch(0 if self.fixed_sampler_epoch else self.epoch)
            batches = torch.utils.data.BatchSampler(self.sampler, self.batch_size, True)
            remaining = list(itertools.islice(batches, self.batch, None))
            loader = torch.utils.data.DataLoader(
                self.dataset, batch_sampler=remaining, num_workers=0,
                generator=torch.Generator().manual_seed(self.seed + self.epoch))
            self._iterator = iter(loader)
        result = next(self._iterator)
        self.batch += 1
        return result

    def state_dict(self):
        return {"epoch": self.epoch, "batch": self.batch, "dataset_size": len(self.dataset),
                "batch_size": self.batch_size, "seed": self.seed, "rank": self.rank,
                "world_size": self.world_size, "fixed_sampler_epoch": self.fixed_sampler_epoch}

    def load_state_dict(self, state):
        current = self.state_dict()
        for key in current.keys() - {"epoch", "batch"}:
            if current[key] != state[key]:
                raise ValueError(f"data stream changed: {key}")
        if state["epoch"] < 0 or not 0 <= state["batch"] <= self.batches_per_epoch:
            raise ValueError("invalid data stream position")
        self.epoch, self.batch = int(state["epoch"]), int(state["batch"])
        self._iterator = None
