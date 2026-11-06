"""Shared protocol validation and content provenance; no CUDA initialization."""
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]
ARM_FIELDS = {"experiment_id", "num_frame_per_block", "r1_weight", "r2_weight", "seed"}
RUNTIME_FIELDS = {"logdir", "resume_ckpt", "config_name", "wandb_save_dir"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def config_dict(config):
    return OmegaConf.to_container(config, resolve=True)


def resume_signature(config):
    return {k: v for k, v in config_dict(config).items() if k not in RUNTIME_FIELDS}


def validate_config(config):
    c = config_dict(config)
    required = {
        "trainer": "iclr", "randomize_seed": False,
        "gradient_num_blocks": 0, "gradient_num_frames": 21,
        "num_training_frames": 21, "score_num_frames": 21,
        "independent_first_frame": False, "i2v": False,
        "rollout_schedule": "fixed", "denoising_step_list": [1000],
        "separate_gan_discriminator_optimizer": False,
        "ema_weight": 0.0, "resume_ema_from_ckpt": False,
        "dataloader_num_workers": 0, "dataset_type": "clean_latent_lmdb",
        "no_save": False, "no_visualize": True,
        "rank0_preload_generator_ckpt": True,
        "rank0_preload_resume_ckpt": True,
    }
    for key, value in required.items():
        if c.get(key) != value:
            raise ValueError(f"ICLR requires {key}={value!r}, got {c.get(key)!r}")
    if "first_frame_denoising_step_list" in c:
        raise ValueError("matched all1 training must not define a first-frame override")
    if c["num_frame_per_block"] not in (1, 3):
        raise ValueError("num_frame_per_block must be 1 or 3")
    if c["image_or_video_shape"][1:] != [21, 16, 60, 104]:
        raise ValueError("ICLR requires 21 latent frames at paper resolution")
    if c["image_or_video_shape"][0] != c["batch_size"]:
        raise ValueError("batch size mismatch")
    if not c.get("generator_ckpt") or c.get("generator_ckpt_after_resume"):
        raise ValueError("one explicit shared raw initialization is required")
    if c["seed"] not in (0, 1, 2):
        raise ValueError("training seeds must be 0,1,2")
    if c["r1_weight"] < 0 or c["r2_weight"] < 0:
        raise ValueError("negative regularization weight")
    if (c["r1_weight"] == 0) != (c["r2_weight"] == 0):
        raise ValueError("R1 and R2 must both be off or both be on")
    if c.get("iclr_mode") not in ("formal", "smoke"):
        raise ValueError("explicit formal/smoke mode required")
    if c["iclr_mode"] == "formal" and c["max_steps"] != 0:
        raise ValueError("formal runs stop by matched time, not a step cap")
    if c["iclr_mode"] == "smoke" and c["max_steps"] < 10:
        raise ValueError("smoke must cover at least 10 iterations")
    from experiments.iclr.state import TrainingClock
    TrainingClock(c["iclr_budget_seconds"], tuple(c["iclr_save_fractions"]))


def validate_matrix(configs):
    if len(configs) != 12:
        raise ValueError("formal matrix requires 12 runs: 2 granularities x 2 regularizers x 3 seeds")
    reference = None
    seen = set()
    on_weights = set()
    for config in configs:
        validate_config(config)
        c = config_dict(config)
        key = (c["num_frame_per_block"], c["r1_weight"] > 0, c["seed"])
        if key in seen:
            raise ValueError(f"duplicate matrix cell: {key}")
        seen.add(key)
        if key[1]:
            on_weights.add((c["r1_weight"], c["r2_weight"]))
        controlled = {k: v for k, v in c.items() if k not in ARM_FIELDS}
        if reference is None:
            reference = controlled
        if controlled != reference:
            diff = sorted(k for k in controlled.keys() | reference.keys()
                          if controlled.get(k) != reference.get(k))
            raise ValueError(f"uncontrolled matrix settings: {diff}")
    if len(on_weights) != 1:
        raise ValueError("all regularized cells must use identical R1/R2 weights")


def tensor_state_sha256(state):
    """Hash actual loaded tensor bytes, names, shapes and dtypes, in CPU chunks."""
    import torch
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach()
        digest.update(json.dumps([key, list(tensor.shape), str(tensor.dtype)]).encode())
        # Chunk transfer avoids a second complete CPU copy of a large GPU model.
        flat = tensor.reshape(-1)
        for start in range(0, flat.numel(), 1024 * 1024):
            chunk = flat[start:start + 1024 * 1024].to("cpu").contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
