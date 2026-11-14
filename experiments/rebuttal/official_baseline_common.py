"""Pinned official implementations and the minimal Self Forcing FFE adapter.

No torch imports here: preparing the protocol and checking provenance work on CPU.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / "official_baselines.lock.json"
LOCK = json.loads(LOCK_PATH.read_text())
FOUR_STEPS = [1000, 750, 500, 250]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def immutable_json(path, payload):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Different experiment already exists at {path}; use a new output directory")
    else:
        json_write(path, payload)


def check_source(path, model):
    path = Path(path).resolve()
    spec = LOCK[model]
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["commit"]:
        raise ValueError(f"Wrong official source revision: {path}: {commit}")
    status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(f"Official source checkout must be clean: {path}\n{status}")
    if sha256(path / "pipeline/causal_inference.py") != spec["pipeline_sha256"]:
        raise ValueError("Official pipeline content differs from the locked version")


def adapted_pipeline_source(source, model):
    """Keep the official loop/cache/noise/decoder; only select SF steps per block.

    Causal-Forcing already implements denoising_step_list_first_chunk natively.
    The SF adapter deliberately leaves its official four-step __init__ intact,
    so the first block's warped timestep tensor is exactly the official tensor.
    """
    if model == "causal_forcing":
        return source
    if model != "self_forcing":
        raise ValueError(model)
    begin = "            # Step 3.1: Spatial denoising loop\n"
    end = "            # Step 3.2: record the model's output\n"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("Unrecognized official SF loop; do not patch an unknown implementation")
    before, rest = source.split(begin)
    loop, after = rest.split(end)
    if loop.count("self.denoising_step_list") != 3:
        raise ValueError("Unexpected SF denoising-list references")
    selection = (
        "            # One-Forcing evaluation adapter: first native block four steps.\n"
        "            block_steps = (self.denoising_step_list\n"
        "                if current_start_frame == 0 or self.args.baseline_eval_schedule == 'all4'\n"
        "                else self.denoising_step_list[:1])\n"
    )
    return before + selection + begin + loop.replace("self.denoising_step_list", "block_steps") + end + after


def load_config(source_root, model, schedule):
    from omegaconf import OmegaConf
    source_root = Path(source_root)
    config = OmegaConf.merge(
        OmegaConf.load(source_root / "configs/default_config.yaml"),
        OmegaConf.load(source_root / LOCK[model]["config"]),
    )
    if list(config.denoising_step_list) != FOUR_STEPS or config.num_frame_per_block != 3:
        raise ValueError("Unexpected official checkpoint sampling configuration")
    if config.independent_first_frame or not config.warp_denoising_step or config.context_noise != 0:
        raise ValueError("Unexpected native first-frame/warping/context settings")
    config.baseline_eval_schedule = schedule
    if model == "causal_forcing":
        config.denoising_step_list = FOUR_STEPS if schedule == "all4" else [1000]
        config.denoising_step_list_first_chunk = FOUR_STEPS
    return config


def select_weights(checkpoint, key):
    """Follow the official CF prefix normalization, but never ignore missing keys."""
    if key not in checkpoint:
        raise ValueError(f"Required official weight key {key!r} is missing; available: {list(checkpoint)}")
    normalized = {}
    for name, tensor in checkpoint[key].items():
        fixed = name.replace("model._fsdp_wrapped_module.", "model.", 1) if name.startswith("model._fsdp_wrapped_module.") else name
        if fixed in normalized:
            raise ValueError(f"Colliding checkpoint keys after FSDP normalization: {fixed}")
        normalized[fixed] = tensor
    return normalized


def selected_records(manifest, phase):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if phase == "smoke":
        rows = [row for row in rows if row["sample_index"] == 0][:10]
    return rows


def validate_output_name(name):
    if Path(name).name != name or "/" in name or "\x00" in name or len(name.encode()) > 255:
        raise ValueError(f"Unsafe or overlong video filename: {name!r}")


def check_completed_video(video, metadata, record, intent_hash):
    if not video.exists() and not metadata.exists():
        return False
    if not video.is_file() or not metadata.is_file():
        raise ValueError(f"Incomplete video/metadata pair: {video}; move it aside and rerun")
    info = json.loads(metadata.read_text())
    if info.get("record") != record or info.get("intent_sha256") != intent_hash or info.get("video_sha256") != sha256(video):
        raise ValueError(f"Video provenance/hash mismatch: {video}")
    return True
