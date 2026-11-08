"""Shared protocol validation and content provenance; no CUDA initialization."""
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]
# Execution order: complete the frame/chunk pair for each training seed.
RUN_SPECS = (
    ("frame_roff_s0", 1, 0),
    ("chunk_roff_s0", 3, 0),
    ("frame_roff_s1", 1, 1),
    ("chunk_roff_s1", 3, 1),
)
LEGACY_PROFILE = "main_block_ablation_200_v2"
WINDOW_PROFILE = "matched_window18_s42_v1"
WINDOW_RUN_SPECS = (("frame_w18_s42", 1, 42), ("chunk_w18_s42", 3, 42))
PROFILE_RUN_SPECS = {LEGACY_PROFILE: RUN_SPECS, WINDOW_PROFILE: WINDOW_RUN_SPECS}
ARM_FIELDS = {"experiment_id", "num_frame_per_block", "r1_weight", "r2_weight", "seed"}
RUNTIME_FIELDS = {"logdir", "resume_ckpt", "config_name", "wandb_save_dir"}
MAIN_REFERENCE = REPO / "experiments/iclr/configs/main_reference.yaml"
REFERENCE_RUNTIME_FIELDS = {"trainer", "seed", "num_frame_per_block", "generator_ckpt",
                           "teacher_model_path", "data_path", "resume_ckpt", "prompt_embedding_cache_path", "max_steps"}


def effective_gradient_frames(config):
    """Mirror the model's block precedence, alignment and length clipping."""
    total = int(config.num_training_frames)
    block = int(config.num_frame_per_block)
    blocks = int(config.get("gradient_num_blocks", 0) or 0)
    frames = blocks * block if blocks > 0 else int(config.get("gradient_num_frames", 0) or 0)
    if frames <= 0:
        return total
    if not config.get("independent_first_frame", False):
        frames = ((frames + block - 1) // block) * block
    return max(1, min(total, frames))


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
    profile = c.get("iclr_profile")
    if profile not in PROFILE_RUN_SPECS:
        raise ValueError(f"unknown ICLR profile: {profile!r}")
    matched_window = profile == WINDOW_PROFILE
    required = {
        "trainer": "iclr", "randomize_seed": False,
        "iclr_profile": profile, "iclr_stop_condition": "fixed_steps",
        "iclr_reference_commit": "c9a2350e8740531562011fc9618e1a928d911ae0",
        "iclr_reference_step": 200, "iclr_eval_schedule": "main_ffe",
        "iclr_eval_weight_source": "generator_ema", "iclr_eval_conditioning": "pinned_qwen",
        "iclr_eval_samples_per_prompt": 5,
        "dataloader_num_workers": 0, "dataset_type": "clean_latent_lmdb",
        "no_save": False, "no_visualize": True,
        "rank0_preload_generator_ckpt": True,
        "rank0_preload_resume_ckpt": True,
    }
    for key, value in required.items():
        if c.get(key) != value:
            raise ValueError(f"ICLR requires {key}={value!r}, got {c.get(key)!r}")
    reference = config_dict(OmegaConf.load(MAIN_REFERENCE))
    permitted = REFERENCE_RUNTIME_FIELDS | ({"gradient_num_blocks"} if matched_window else set())
    for key, value in reference.items():
        if key not in permitted and c.get(key) != value:
            raise ValueError(f"main reference setting changed: {key}; expected {value!r}, got {c.get(key)!r}")
    for key, default in {"manual_generator_backward": False, "generator_optimizer_state_cpu_offload": False,
                         "separate_gan_discriminator_optimizer": False,
                         "gradient_num_frames": 18 if matched_window else 0,
                         "rollout_schedule": "fixed", "gradient_window_position": "tail"}.items():
        if c.get(key, default) != default:
            raise ValueError(f"main implementation default changed: {key}")
    if matched_window and (c.get("gradient_num_blocks") != 0 or c.get("gradient_num_frames") != 18
                           or effective_gradient_frames(config) != 18):
        raise ValueError("matched window requires gradient_num_blocks=0 and gradient_num_frames=18")
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
    specs = PROFILE_RUN_SPECS[profile]
    if (c.get("experiment_id"), c["num_frame_per_block"], c["seed"]) not in specs:
        raise ValueError("allowed runs: " + ", ".join(name for name, _, _ in specs))
    if c["r1_weight"] != 0 or c["r2_weight"] != 0:
        raise ValueError("R1 and R2 must both be off for all runs")
    if c.get("iclr_mode") not in ("formal", "smoke"):
        raise ValueError("explicit formal/smoke mode required")
    if c["iclr_mode"] == "formal" and c["max_steps"] != reference["max_steps"]:
        raise ValueError("formal frame/chunk runs must stop at exactly 200 iterations")
    if c["iclr_mode"] == "smoke" and c["max_steps"] != 10:
        raise ValueError("smoke must cover exactly 10 iterations")
    if "iclr_budget_seconds" in c or "iclr_save_fractions" in c:
        raise ValueError("time-budget configs are obsolete; regenerate the fixed-200 matrix")
    expected_saves = [0, 5, 10] if c["iclr_mode"] == "smoke" else [0, 50, 100, 150, 200]
    if c.get("iclr_save_steps") != expected_saves:
        raise ValueError(f"fixed-step checkpoints must be {expected_saves}")
    from experiments.iclr.state import TrainingClock
    TrainingClock(c["max_steps"], tuple(c["iclr_save_steps"]))


def validate_matrix(configs):
    profile = configs[0].get("iclr_profile") if configs else LEGACY_PROFILE
    if profile not in PROFILE_RUN_SPECS:
        raise ValueError(f"unknown ICLR profile: {profile!r}")
    specs = PROFILE_RUN_SPECS[profile]
    if len(configs) != len(specs):
        raise ValueError(f"matrix requires {len(specs)} runs: " + ", ".join(name for name, _, _ in specs)
                         + "; regenerate legacy matrices")
    reference = None
    for config, expected in zip(configs, specs):
        validate_config(config)
        if config.iclr_profile != profile:
            raise ValueError("mixed experiment profiles in matrix")
        c = config_dict(config)
        key = (c["experiment_id"], c["num_frame_per_block"], c["seed"])
        if key != expected:
            raise ValueError(f"matrix run order/cell mismatch: expected {expected}, got {key}")
        controlled = {k: v for k, v in c.items() if k not in ARM_FIELDS}
        if reference is None:
            reference = controlled
        if controlled != reference:
            diff = sorted(k for k in controlled.keys() | reference.keys()
                          if controlled.get(k) != reference.get(k))
            raise ValueError(f"uncontrolled matrix settings: {diff}")


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
