"""Pinned inputs and controls for endpoint-anchored CD; safe for CPU dry-runs."""
from experiments.iclr.protocol import sha256_file

PROFILE = "endpoint_anchored_cd_v1"
AR_SHA256 = "721ea4d8f6d23099ccbefc6f5741b170402f509738f5a7b2db10e91b76857825"
AR_DEFAULT = "checkpoints/framewise/ar_diffusion.pt"
SAVE_STEPS = (0, 50, 100, 150, 200, 250, 300)


def validate_one_step_config(config):
    mode = config.get("iclr_cd_mode")
    if mode not in ("formal", "smoke"):
        raise ValueError("CD mode must be formal or smoke")
    expected = {"iclr_cd_profile": PROFILE, "max_steps": 10 if mode == "smoke" else 300,
        "training_num_frames": 21, "num_frame_per_block": 1,
        "endpoint_loss_weight": 1., "holdout_rows": 64,
        "checkpoint_interval": 5 if mode == "smoke" else 50, "save_initial_checkpoint": True,
        "required_generator_sha256": AR_SHA256}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"one-step CD requires {key}={value!r}")


def split_rows(size, heldout):
    if heldout < 0 or size <= heldout:
        raise ValueError("trajectory dataset must contain training rows beyond holdout")
    return list(range(heldout, size)), list(range(heldout))


def verify_ar_source(checkpoint, manifest):
    """Use content identity so moving a verified dataset between machines is safe."""
    digest = sha256_file(checkpoint)
    if digest != AR_SHA256:
        raise ValueError("CD initialization must be the published framewise/ar_diffusion.pt (SHA256 mismatch)")
    if manifest.get("generator_checkpoint_sha256") != digest:
        raise ValueError("trajectory source is not hash-verified AR diffusion; rebuild with the pinned checkpoint")
    if manifest.get("use_ema") is not False:
        raise ValueError("CD source must use raw weights")
    times = manifest.get("selected_timesteps", [])
    shape = manifest.get("latents_shape", [])
    if (len(times) != 6 or times[0] != 1000 or times[-2:] != [0., None]
            or len(shape) != 6 or shape[1:3] != [6, 21]):
        raise ValueError("one-step CD requires 21-frame trajectories spanning t=1000 to 0 with four adjacent pairs")
    return digest


def validate_done_pair(curved, rectified, *, mode="formal"):
    """Fail before evaluation if either training arm changed a scientific control."""
    if mode not in ("formal", "smoke"):
        raise ValueError("CD audit mode must be formal or smoke")
    keys = ("iclr_cd_profile", "initial_generator_sha256", "dataset_manifest_sha256",
        "seed", "world_size", "training_num_frames", "training_timesteps", "adjacent_pairs",
        "pair_schedule", "training_objective", "target_ema_decay", "endpoint_loss_weight",
        "holdout_row_indices", "global_sample_order_sha256", "history_conditioning", "final_step", "config")
    for arm, payload in (("curved", curved), ("rectified", rectified)):
        if (payload.get("intervention") != arm or payload.get("status") != "complete"
                or payload.get("iclr_cd_profile") != PROFILE
                or payload.get("initial_generator_sha256") != AR_SHA256
                or payload.get("final_step") != (10 if mode == "smoke" else 300)
                or payload.get("use_ema") is not False):
            raise ValueError(f"invalid completed one-step CD arm: {arm}")
        if any(key not in payload for key in keys):
            raise ValueError(f"missing CD controls in {arm}")
        validate_one_step_config(payload["config"])
        if (payload["config"]["iclr_cd_mode"] != mode
                or payload["training_num_frames"] != 21 or payload["endpoint_loss_weight"] != 1
                or payload["history_conditioning"] != "teacher_forcing_clean"
                or payload["training_objective"] != "adjacent_cd_plus_endpoint_l2"):
            raise ValueError("CD metadata contradicts the declared training protocol")
        if payload["holdout_row_indices"] != list(range(64)):
            raise ValueError("CD holdout differs from fixed rows 0..63")
    for key in keys:
        if curved[key] != rectified[key]:
            raise ValueError(f"CD pair differs in {key}")
    return {key: curved[key] for key in keys}


def validate_checkpoint(payload, done, step):
    metadata = payload.get("experiment_metadata", {})
    if (payload.get("step") != step or metadata.get("checkpoint_step") != step
            or metadata.get("use_ema") is not False or "generator" not in payload
            or "generator_ema" in payload):
        raise ValueError("CD evaluation requires the declared raw checkpoint")
    for key in ("iclr_cd_profile", "initial_generator_sha256", "dataset_manifest_sha256",
                "intervention", "seed", "training_num_frames", "endpoint_loss_weight",
                "holdout_row_indices", "config", "history_conditioning"):
        if key not in done or metadata.get(key) != done[key]:
            raise ValueError(f"CD checkpoint differs from completed run in {key}")
    return metadata
