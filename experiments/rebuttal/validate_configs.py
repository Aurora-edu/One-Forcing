#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
ONE_FORCING = CONFIG_DIR / "train_1step_one_forcing.yaml"
DMD_ONLY = CONFIG_DIR / "train_1step_dmd_only.yaml"
FOUR_STEP = CONFIG_DIR / "train_4step_one_forcing.yaml"
EVAL_ALL1 = CONFIG_DIR / "eval_all1.yaml"
EVAL_FFE = CONFIG_DIR / "eval_ffe.yaml"
EVAL_ALL4 = CONFIG_DIR / "eval_all4.yaml"


def flatten(config):
    container = OmegaConf.to_container(config, resolve=True)
    output = {}

    def visit(value, prefix):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}.{key}" if prefix else key)
        else:
            output[prefix] = value

    visit(container, "")
    return output


def differences(left, right):
    left_flat = flatten(left)
    right_flat = flatten(right)
    keys = sorted(set(left_flat) | set(right_flat))
    return {
        key: (left_flat.get(key), right_flat.get(key))
        for key in keys
        if left_flat.get(key) != right_flat.get(key)
    }


def require_exact_differences(left_path, right_path, expected):
    actual = set(differences(load_config(str(left_path)), load_config(str(right_path))))
    if actual != set(expected):
        raise AssertionError(
            f"{left_path.name} vs {right_path.name}: expected differences "
            f"{sorted(expected)}, got {sorted(actual)}"
        )


def validate_semantics(path):
    config = load_config(str(path))
    if config.seed < 0 or config.randomize_seed:
        raise AssertionError(f"{path.name}: rebuttal runs require a fixed non-negative seed")
    if config.dataset_type != "clean_latent_lmdb":
        raise AssertionError(f"{path.name}: paired GAN ablations require clean_latent_lmdb")
    if config.max_steps < 600 or config.log_iters > 100:
        raise AssertionError(
            f"{path.name}: the 48-hour stability sweep must cover >=600 steps "
            "at <=100-step intervals"
        )
    if config.num_frame_per_block != 1 or config.rollout_schedule != "fixed":
        raise AssertionError(f"{path.name}: all training comparisons must use framewise fixed rollout")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    for path in (ONE_FORCING, DMD_ONLY, FOUR_STEP):
        validate_semantics(path)

    require_exact_differences(
        ONE_FORCING,
        DMD_ONLY,
        {
            "experiment_id",
            "gan_g_weight",
            "gan_d_weight",
        },
    )
    require_exact_differences(
        ONE_FORCING,
        FOUR_STEP,
        {
            "experiment_id",
            "denoising_step_list",
        },
    )
    all1 = load_config(str(EVAL_ALL1))
    ffe = load_config(str(EVAL_FFE))
    all4 = load_config(str(EVAL_ALL4))
    if hasattr(all1, "first_frame_denoising_step_list"):
        raise AssertionError("eval_all1 must not contain a first-frame override")
    if hasattr(all4, "first_frame_denoising_step_list"):
        raise AssertionError("eval_all4 must not contain a first-frame override")
    if list(all1.denoising_step_list) != [1000]:
        raise AssertionError("eval_all1 has the wrong denoising schedule")
    if ffe.rollout_schedule != "first4then1":
        raise AssertionError("eval_ffe must use first4then1")
    if ffe.first_rollout_num_frames != 4:
        raise AssertionError("eval_ffe must generate four latent frames in its first block")
    if list(ffe.first_frame_denoising_step_list) != [1000, 750, 500, 250]:
        raise AssertionError("eval_ffe has the wrong first-block denoising schedule")
    if list(all4.denoising_step_list) != [1000, 750, 500, 250]:
        raise AssertionError("eval_all4 has the wrong denoising schedule")
    for path, config in ((EVAL_ALL1, all1), (EVAL_FFE, ffe), (EVAL_ALL4, all4)):
        if config.num_frame_per_block != 1:
            raise AssertionError(f"{path.name}: rebuttal inference must be framewise")
        if config.model_kwargs.local_attn_size != 21:
            raise AssertionError(f"{path.name}: long rollout requires local_attn_size=21")
    print("Rebuttal configs are paired and valid.")


if __name__ == "__main__":
    main()
