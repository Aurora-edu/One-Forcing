"""CPU-testable training budget, counters and restartable data order."""
from dataclasses import asdict, dataclass, field
import itertools
import math


@dataclass
class TrainingClock:
    budget_seconds: float
    fractions: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    training_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    overhead_seconds: float = 0.0
    critic_updates: int = 0
    generator_updates: int = 0
    step: int = 0
    saved_fractions: list = field(default_factory=list)

    def __post_init__(self):
        self.fractions = tuple(self.fractions)
        if not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be finite and positive")
        if (tuple(sorted(set(self.fractions))) != self.fractions
                or not self.fractions or self.fractions[0] != 0 or self.fractions[-1] != 1):
            raise ValueError("fractions must increase strictly from zero to one")
        if any(not 0 <= x <= 1 for x in self.fractions):
            raise ValueError("fractions must be in [0, 1]")

    @property
    def exhausted(self):
        return self.training_seconds >= self.budget_seconds

    def advance(self, seconds, *, generator, critic=True):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid synchronized training duration")
        self.training_seconds += seconds
        self.step += 1
        self.critic_updates += int(critic)
        self.generator_updates += int(generator)

    def due(self):
        return [x for x in self.fractions if x not in self.saved_fractions
                and self.training_seconds >= self.budget_seconds * x]

    def should_continue(self, mode, max_steps):
        if mode == "smoke":
            return self.step < max_steps
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
    def __init__(self, dataset, *, batch_size, seed, rank=0, world_size=1):
        import torch
        self.dataset = dataset
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
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
            self.sampler.set_epoch(self.epoch)
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
                "world_size": self.world_size}

    def load_state_dict(self, state):
        current = self.state_dict()
        for key in current.keys() - {"epoch", "batch"}:
            if current[key] != state[key]:
                raise ValueError(f"data stream changed: {key}")
        if state["epoch"] < 0 or not 0 <= state["batch"] <= self.batches_per_epoch:
            raise ValueError("invalid data stream position")
        self.epoch, self.batch = int(state["epoch"]), int(state["batch"])
        self._iterator = None
