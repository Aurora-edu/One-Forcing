"""Audit the new seed-2026 DMD-only run against the reused framewise GAN run."""
import argparse
import json
import math
from pathlib import Path

from experiments.iclr.protocol import (DMD_ONLY_PROFILE, SEED2026_PROFILE, RUNTIME_FIELDS,
    atomic_json, config_dict, resume_signature, validate_config)
from utils.config import load_config


PAIR_DIFFERENCES = {"experiment_id", "iclr_profile", "gan_g_weight", "gan_d_weight",
                    "iclr_eval_schedule", "iclr_eval_weight_source", "iclr_eval_samples_per_prompt"}
SHARED_INITIALIZATION = ("generator_checkpoint_sha256", "generator_tensor_sha256",
                        "generator_source_key", "critic_backbone_tensor_sha256", "teacher_model_path")


def read(path):
    return json.loads(Path(path).read_text())


def controlled_config(config):
    return {k: v for k, v in config_dict(config).items()
            if k not in PAIR_DIFFERENCES | RUNTIME_FIELDS}


def audit_endpoint(run, config, initialization, *, dmd_only):
    from experiments.rebuttal.audit_noema_checkpoint import audit_checkpoint
    import torch

    expected = {"step": 200, "generator_updates": 40, "critic_updates": 200, "max_steps": 200}
    done = read(run / "training.done")
    if (any(done.get(k) != v for k, v in expected.items())
            or done.get("formal_result") is not True or done.get("step_budget_reached") is not True):
        raise ValueError("training must finish at 200 iterations / 40 G / 200 critic updates")
    checkpoint = run / "checkpoint_model_000200/model.pt"
    complete = read(checkpoint.parent / "complete.json")
    if (complete.get("schema_version") != 3 or complete.get("world_size") != 8
            or complete.get("config") != resume_signature(config)
            or complete.get("initialization") != initialization
            or any(complete.get("clock", {}).get(k) != v for k, v in expected.items())
            or complete.get("files", {}).get("model.pt") != checkpoint.stat().st_size):
        raise ValueError("saved checkpoint configuration/initialization/budget differs from run")
    raw = audit_checkpoint(checkpoint, expected_step=200)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    metadata = payload.get("experiment_metadata", {})
    if metadata.get("profile") != config.iclr_profile or metadata.get("mode") != "formal":
        raise ValueError("checkpoint belongs to a different experiment")
    del payload
    if (run / "nonfinite_failure.json").exists():
        raise ValueError("run contains a nonfinite failure record")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("step") for row in rows] != list(range(1, 201)):
        raise ValueError("require complete ordered metrics for steps 1..200; preserve pre-resume logs too")
    for row in rows:
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise ValueError("nonfinite training metrics")
        required = ["critic_loss", "critic_grad_norm", "gan_d_loss", "r1_loss", "r2_loss"]
        if (row["step"] - 1) % 5 == 0:
            required += ["generator_loss", "generator_grad_norm", "gan_g_loss", "dmd_loss"]
        if any(k not in row for k in required):
            raise ValueError("missing loss/gradient measurements")
        for key in ("r1_loss", "r2_loss") + (("gan_g_loss", "gan_d_loss") if dmd_only else ()):
            if row.get(key, 0) != 0:
                raise ValueError(f"disabled loss is nonzero: {key}")
    if not dmd_only and any(not any(row.get(k, 0) > 0 for row in rows)
                            for k in ("gan_g_loss", "gan_d_loss")):
        raise ValueError("GAN run must record active generator and discriminator losses")
    return {**raw, "training_seconds": done["training_seconds"], **expected}


def audit(gan_run, dmd_run, *, initialization_only=False):
    runs = [Path(gan_run).resolve(), Path(dmd_run).resolve()]
    configs = [load_config(str(run / "resolved_config.yaml")) for run in runs]
    for config, profile, name in zip(configs, (SEED2026_PROFILE, DMD_ONLY_PROFILE),
                                     ("frame_roff_s2026", "dmd_only_s2026")):
        validate_config(config)
        if (config.iclr_profile != profile or config.experiment_id != name
                or config.iclr_mode != "formal" or config.num_frame_per_block != 1):
            raise ValueError("expected the new seed-2026 framewise GAN / DMD-only pair")
    left, right = map(controlled_config, configs)
    differences = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
    if differences:
        raise ValueError(f"uncontrolled training settings: {differences}")
    initializations = [read(run / "initialization.json") for run in runs]
    for init, cfg, gan_enabled in zip(initializations, configs, (True, False)):
        if (init.get("initialization_seed") != 2026 or init.get("experiment_id") != cfg.experiment_id
                or init.get("strict_load") is not True or init.get("use_ema") is not False
                or init.get("use_gan_branch") is not gan_enabled):
            raise ValueError("expected strict raw initialization, seed 2026 and declared GAN branch")
    for key in SHARED_INITIALIZATION:
        a, b = (init.get(key) for init in initializations)
        if not a or a != b:
            raise ValueError(f"shared initialization missing or different: {key}")
    sources = [read(run / "source.json") for run in runs]
    if not sources[0].get("git_commit") or sources[0] != sources[1]:
        raise ValueError("training source records differ; use the same source tree for both runs")
    endpoints = {} if initialization_only else {
        label: audit_endpoint(run, cfg, init, dmd_only=label == "dmd_only")
        for label, run, cfg, init in zip(("dmd_gan", "dmd_only"), runs, configs, initializations)}
    return {"status": "initialization_matched" if initialization_only else "pair_verified",
        "runs": dict(zip(("dmd_gan", "dmd_only"), map(str, runs))), "seed": 2026,
        "shared_initialization": {k: initializations[0][k] for k in SHARED_INITIALIZATION},
        "endpoints": endpoints, "evaluation": {"weight_source": "generator", "schedule": "ffe",
            "conditioning": "pinned_qwen", "samples_per_prompt": 1},
        "interpretation": "new paired training; not historical Table-4a checkpoint reevaluation",
        "randomness": "same training seed and data order; GAN changes RNG consumption, not paired training noise",
        "source_limit": "git records do not fingerprint dirty file contents; archive the shared source/diff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gan-run", required=True)
    parser.add_argument("--dmd-run", required=True)
    parser.add_argument("--initialization-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.gan_run, args.dmd_run, initialization_only=args.initialization_only)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
