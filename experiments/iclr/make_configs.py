"""Generate 4 fixed-200-step runs. Dry-run needs no checkpoints or GPUs."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from experiments.iclr.protocol import RUN_SPECS, atomic_json, config_dict, validate_matrix
from utils.config import load_config


def make_matrix(base, *, generator_ckpt, teacher, data, prompt_cache, smoke=False):
    configs = []
    for name, block, seed in RUN_SPECS:
        config = OmegaConf.merge(base, {
            "experiment_id": name, "num_frame_per_block": block, "seed": seed,
            "generator_ckpt": str(generator_ckpt), "teacher_model_path": str(teacher),
            "data_path": str(data), "prompt_embedding_cache_path": str(prompt_cache),
            "r1_weight": 0.0, "r2_weight": 0.0,
            "iclr_mode": "smoke" if smoke else "formal",
            "iclr_save_steps": [0, 5, 10] if smoke else [0, 50, 100, 150, 200],
            "max_steps": 10 if smoke else int(base.max_steps),
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
    parser.add_argument("--output", default="/data/fengjiaqi/iclr/configs")
    parser.add_argument("--smoke", action="store_true", help="10-step configs, never formal results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = make_matrix(load_config(args.base), generator_ckpt=args.generator_ckpt,
                          teacher=args.teacher, data=args.data, prompt_cache=args.prompt_cache,
                          smoke=args.smoke)
    print(json.dumps({"num_runs": len(configs), "mode": configs[0].iclr_mode,
                      "runs": [c.experiment_id for c in configs],
                      "initialization_status": "UNVERIFIED: explicit raw checkpoint must be audited; filename is not evidence of ODE provenance",
                      "max_steps": configs[0].max_steps, "stop_condition": "fixed_steps",
                      "evaluation": "EMA/main_ffe/pinned Qwen; fixed-step curves and step-200 table; actual training costs recorded"}, indent=2))
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
