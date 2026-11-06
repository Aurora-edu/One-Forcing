"""Generate/audit 12 matched runs. Dry-run needs no checkpoints or GPUs."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from experiments.iclr.protocol import atomic_json, config_dict, validate_matrix
from utils.config import load_config


def make_matrix(base, *, generator_ckpt, teacher, data, prompt_cache,
                budget_seconds, smoke=False):
    configs = []
    for block, method in ((1, "frame"), (3, "chunk")):
        for regularized in (False, True):
            for seed in (0, 1, 2):
                name = f"{method}_r{'on' if regularized else 'off'}_s{seed}"
                config = OmegaConf.merge(base, {
                    "experiment_id": name, "num_frame_per_block": block, "seed": seed,
                    "generator_ckpt": str(generator_ckpt), "teacher_model_path": str(teacher),
                    "data_path": str(data), "prompt_embedding_cache_path": str(prompt_cache),
                    "r1_weight": 1.0 if regularized else 0.0,
                    "r2_weight": 0.001 if regularized else 0.0,
                    "iclr_budget_seconds": float(budget_seconds),
                    "iclr_mode": "smoke" if smoke else "formal",
                    "max_steps": 10 if smoke else 0,
                })
                configs.append(config)
    validate_matrix(configs)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(Path(__file__).parent / "configs/base.yaml"))
    parser.add_argument("--generator-ckpt", required=True)
    parser.add_argument("--teacher", default="/data/fengjiaqi/wan_models/Wan2.1-T2V-14B")
    parser.add_argument("--data", default="/data/fengjiaqi/One-Forcing/mixkit_latents_lmdb")
    parser.add_argument("--prompt-cache", default="")
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/configs")
    parser.add_argument("--smoke", action="store_true", help="10-step configs, never formal results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = make_matrix(load_config(args.base), generator_ckpt=args.generator_ckpt,
                          teacher=args.teacher, data=args.data, prompt_cache=args.prompt_cache,
                          budget_seconds=args.budget_seconds, smoke=args.smoke)
    print(json.dumps({"num_runs": len(configs), "mode": configs[0].iclr_mode,
                      "runs": [c.experiment_id for c in configs],
                      "initialization_status": "UNVERIFIED: explicit raw checkpoint must be audited; filename is not evidence of ODE provenance",
                      "evaluation": "raw/all1/same paired manifest"}, indent=2))
    if args.dry_run:
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite configs: {root}")
    entries = []
    for config in configs:
        filename = f"{config.experiment_id}.yaml"
        OmegaConf.save(config, root / filename, resolve=True)
        entries.append({"name": config.experiment_id, "config": filename})
    atomic_json(root / "matrix.json", {"schema_version": 1, "runs": entries,
                "mode": configs[0].iclr_mode, "configs": [config_dict(c) for c in configs]})


if __name__ == "__main__":
    main()
