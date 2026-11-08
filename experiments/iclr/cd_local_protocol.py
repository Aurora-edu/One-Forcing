"""Local progressive trajectory treatment with exclusively all1 evaluation."""
import copy
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from experiments.iclr import cd_original_protocol as original

PROFILE = "local_progressive_cd_all1_v1"
EXTRA = {"local_rectification_point": 1, "local_rectification_ramp_steps": 100,
         "iclr_cd_eval_schedule": "all1"}


def original_config(config):
    c = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else copy.deepcopy(config)
    if c.get("iclr_cd_profile") != PROFILE or c.get("seed") not in (2026, 2027):
        raise ValueError("local all1 CD requires its own profile and seed 2026 or 2027")
    for key, expected in EXTRA.items():
        if c.pop(key, None) != expected:
            raise ValueError(f"local all1 CD requires {key}={expected!r}")
    c.update(iclr_cd_profile=original.PROFILE, seed=2026)
    original.validate_config(c)
    return c


def validate_config(config):
    original_config(config)


def alpha(step, *, enabled, ramp=100):
    if step < 1 or ramp != 100:
        raise ValueError("local rectification uses optimizer steps >=1 and a fixed 100-update ramp")
    return min(step / ramp, 1.) if enabled else 0.


def transform(trajectory, times, *, step, enabled, frames=9):
    """Modify only state 1 inside the trained window, in float32 before bf16."""
    import torch
    original.validate_times(list(times[:-1]))
    if (trajectory.ndim != 6 or trajectory.shape[1] != 6 or times[-1] is not None
            or trajectory.shape[2] < frames or frames != 9):
        raise ValueError("local rectification requires [B,6,F,...] and the first 9 frames")
    strength = alpha(step, enabled=enabled)
    value = trajectory.float()
    if not enabled:
        return value, strength, torch.zeros((), device=value.device)
    high, middle, low = map(float, times[:3])
    weight = (middle - low) / (high - low)
    before = value[:, 1, :frames]
    line = weight * value[:, 0, :frames] + (1 - weight) * value[:, 2, :frames]
    result = value.clone()
    result[:, 1, :frames] = (1 - strength) * before + strength * line
    mse = (result[:, 1, :frames] - before).square().mean()
    return result, strength, mse


def validate_done_pair(curved, rectified, *, mode="formal"):
    normalized = []
    for arm, run in (("curved", curved), ("rectified", rectified)):
        c = original_config(run["config"])
        if run.get("iclr_cd_profile") != PROFILE or run.get("seed") != run["config"]["seed"]:
            raise ValueError("local CD metadata contradicts its declared profile or seed")
        expected = "high_noise_local_progressive" if arm == "rectified" else "none"
        if run.get("rectification_treatment") != expected:
            raise ValueError("local CD treatment metadata differs from the actual arm")
        item = copy.deepcopy(run)
        item.update(config=c, iclr_cd_profile=original.PROFILE, seed=2026)
        normalized.append(item)
    controls = original.validate_done_pair(*normalized, mode=mode)
    for key in controls:
        if curved[key] != rectified[key]:
            raise ValueError(f"local CD pair differs in {key}")
    return {key: curved[key] for key in controls}


def audit_metrics(path, done):
    result = original.audit_metrics(path, done)
    enabled = done["intervention"] == "rectified"
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get("rectification_alpha") != alpha(row["step"], enabled=enabled):
            raise ValueError("local CD ramp in actual metrics differs from the fixed schedule")
        change = row.get("rectification_mse")
        if change is None or not math.isfinite(change) or change < 0 or (not enabled and change != 0):
            raise ValueError("invalid recorded local intervention magnitude")
    result.update(final_rectification_alpha=alpha(done["final_step"], enabled=enabled),
                  evaluation_schedule="all1")
    return result
