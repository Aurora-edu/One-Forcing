"""Frozen historical CD recipe and audits, independent of the endpoint variant."""
import hashlib
import json
import math
from pathlib import Path
import re

from omegaconf import OmegaConf

from experiments.iclr.protocol import REPO, sha256_file

PROFILE = "original_adjacent_cd_9f_v1"
REFERENCE = REPO / "experiments/rebuttal/configs/train_curvature_cd_small.yaml"
REFERENCE_SHA256 = "d4bdfeb74bfd466ba19ed766ddf0fd9ad7613596e72a8b114c0a90ebfa391a64"
TIMESTEPS = [1000., 937.5, 833.3333129882812, 625., 0.]


def validate_config(config):
    if sha256_file(REFERENCE) != REFERENCE_SHA256:
        raise ValueError("historical CD reference YAML changed; do not silently alter the recipe")
    reference = OmegaConf.to_container(OmegaConf.load(REFERENCE), resolve=True)
    mode = config.get("iclr_cd_mode")
    if mode not in ("formal", "smoke"):
        raise ValueError("original CD mode must be formal or smoke")
    if mode == "smoke":
        reference.update(max_steps=10, checkpoint_interval=5, print_interval=1)
    reference.update(iclr_cd_profile=PROFILE, iclr_cd_mode=mode,
                     endpoint_loss_weight=0., holdout_rows=0,
                     save_initial_checkpoint=mode == "smoke")
    actual = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    for key, value in reference.items():
        if key == "prompt_embedding_cache_path":
            continue  # Runtime asset path, not a training hyperparameter.
        if actual.get(key) != value:
            raise ValueError(f"original CD requires {key}={value!r}, got {actual.get(key)!r}")
    if set(actual) - set(reference):
        raise ValueError(f"unrecognized original CD overrides: {sorted(set(actual) - set(reference))}")


def validate_times(times):
    if (len(times) != 5 or any(not math.isfinite(float(t)) or abs(float(t) - expected) > 1e-4
                             for t, expected in zip(times, TIMESTEPS))):
        raise ValueError("original CD requires the historical four aligned trajectory inputs")


def verify_source(checkpoint, manifest):
    """Permit existing trajectories only when their content source matches the init."""
    digest = sha256_file(checkpoint)
    if manifest.get("generator_checkpoint_sha256") != digest:
        raise ValueError("trajectory source hash differs from initialization or is missing; do not relabel the dataset")
    if manifest.get("use_ema") is not False or manifest.get("guidance_scale") != 6.:
        raise ValueError("original CD requires raw/no-EMA trajectories with CFG=6")
    times = manifest.get("selected_timesteps", [])
    if len(times) != 6 or times[-1] is not None:
        raise ValueError("original CD requires five ODE states and separate clean conditioning")
    validate_times(times[:-1])
    shape = manifest.get("latents_shape", [])
    if (len(shape) != 6 or shape[1:3] != [6, 21]
            or any(not isinstance(n, int) or n < 1 for n in shape)):
        raise ValueError("reuse requires the existing 21-frame trajectory dataset; training slices only the first 9")
    return digest


def validate_done_pair(curved, rectified, *, mode="formal"):
    keys = ("iclr_cd_profile", "initial_generator_sha256", "dataset_manifest_sha256",
            "seed", "world_size", "training_num_frames", "training_timesteps", "adjacent_pairs",
            "pair_schedule", "training_objective", "target_network", "target_ema_decay",
            "endpoint_loss_weight", "holdout_row_indices", "global_sample_order_sha256",
            "history_conditioning", "evaluation_use_ema", "final_step", "config")
    for arm, run in (("curved", curved), ("rectified", rectified)):
        if any(key not in run for key in keys):
            raise ValueError(f"missing original CD controls in {arm}")
        validate_config(run["config"])
        expected = {"intervention": arm, "status": "complete", "iclr_cd_profile": PROFILE,
                    "seed": 2026, "world_size": 8, "training_num_frames": 9,
                    "training_objective": "adjacent_state_consistency_l2",
                    "target_network": "in_training_ema_stop_gradient", "target_ema_decay": .95,
                    "endpoint_loss_weight": 0., "holdout_row_indices": [],
                    "history_conditioning": "teacher_forcing_clean", "use_ema": False,
                    "evaluation_use_ema": False, "final_step": 10 if mode == "smoke" else 300,
                    "pair_schedule": "deterministic_low_to_high_sweep"}
        if run["config"]["iclr_cd_mode"] != mode:
            raise ValueError("original CD audit mode differs from completed config")
        for key, value in expected.items():
            if run.get(key) != value:
                raise ValueError(f"original CD {arm} requires {key}={value!r}")
        validate_times(run["training_timesteps"])
        if run["adjacent_pairs"] != [list(pair) for pair in zip(run["training_timesteps"][:-1], run["training_timesteps"][1:])]:
            raise ValueError("original CD adjacent pair metadata is inconsistent")
        for key in ("initial_generator_sha256", "dataset_manifest_sha256", "global_sample_order_sha256"):
            if not isinstance(run[key], str) or not re.fullmatch(r"[0-9a-f]{64}", run[key]):
                raise ValueError(f"original CD requires a recorded {key}")
    for key in keys:
        if curved[key] != rectified[key]:
            raise ValueError(f"original CD pair differs in {key}")
    return {key: curved[key] for key in keys}


def audit_metrics(path, done):
    """Verify actual updates, including zero endpoint gradients and sample order."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if [row["step"] for row in rows] != list(range(1, done["final_step"] + 1)):
        raise ValueError("original CD metrics must cover every update exactly once")
    digest = hashlib.sha256()
    pairs = [0] * 4
    for row in rows:
        step = row["step"]
        pair = 3 - ((step - 1) % 4)
        if (row["pair_index"] != pair or row["endpoint_loss_weight"] != 0
                or row["endpoint_updates"] != 0 or row["endpoint_loss"] != 0
                or row["intervention"] != done["intervention"]):
            raise ValueError("original CD metrics show changed pair schedule or endpoint supervision")
        if any(not math.isfinite(float(row[key])) for key in ("loss", "cd_loss", "grad_norm", "seconds")):
            raise ValueError("nonfinite original CD training metrics")
        if row["loss"] != row["cd_loss"] or len(row["global_row_indices"]) != 8:
            raise ValueError("original CD objective or global batch differs from the historical recipe")
        digest.update(f"{step}:".encode() + ",".join(str(i) for i in row["global_row_indices"]).encode() + b"\n")
        pairs[pair] += 1
    if digest.hexdigest() != done["global_sample_order_sha256"]:
        raise ValueError("original CD sample-order digest differs from actual metrics")
    return {"updates": len(rows), "updates_per_adjacent_pair": pairs, "endpoint_updates": 0,
            "metrics_sha256": sha256_file(path)}


def historical_comparison(records):
    if len(records) != 4 or any(row["cd_profile"] != PROFILE for row in records):
        raise ValueError("historical comparison requires four original-recipe CD cells")
    source = REPO / "experiments/rebuttal/results/final_metrics.json"
    old = json.loads(source.read_text())["reviewer_runbook_experiments"]["curvature_cd_small"]["runs"]
    rows = []
    for record in records:
        name = ("curved" if record["arm"] == "original" else "rectified") + "_" + record["schedule"]
        for metric in ("total_score", "quality_score", "semantic_score"):
            previous = old[name]["normalized_aggregates"][metric]
            rows.append({"arm": record["arm"], "schedule": record["schedule"], "metric": metric,
                         "historical_score": previous, "new_score": record[metric],
                         "new_minus_historical": record[metric] - previous})
    return rows
