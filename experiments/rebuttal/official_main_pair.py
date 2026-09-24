#!/usr/bin/env python3
"""Prepare and audit a GAN ablation trained by the unmodified official main code."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import secrets
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal.audit_noema_checkpoint import audit_checkpoint
from experiments.rebuttal.paper_vbench_protocol import audit_run
from experiments.rebuttal.summarize_paper_aligned_vbench import PAIRED_FIELDS


OFFICIAL_COMMIT = "c9a2350e8740531562011fc9618e1a928d911ae0"
ARMS = ("full", "dmd")
CONFIG_NAMES = {"full": "one_forcing_official.yaml", "dmd": "dmd_only_official.yaml"}
RUN_NAMES = {"full": "one_forcing_official_main", "dmd": "dmd_only_official_main"}
GAN_KEYS = ("gan_g_weight", "gan_d_weight")
SCORE_KEYS = ("total_score", "quality_score", "semantic_score")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(path: Path) -> dict:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def official_config(official_root: Path) -> dict:
    official_root = official_root.resolve()
    if _git(official_root, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError(f"Official source must be main@{OFFICIAL_COMMIT}")
    if _git(official_root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official source has tracked modifications")
    config_path = official_root / "config.yaml"
    committed = subprocess.check_output(
        ["git", "-C", str(official_root), "show", f"{OFFICIAL_COMMIT}:config.yaml"]
    )
    if config_path.read_bytes() != committed:
        raise ValueError("Official config.yaml differs from the pinned commit")
    config = _config(config_path)
    expected = {
        "max_steps": 200,
        "log_iters": 50,
        "num_frame_per_block": 1,
        "denoising_step_list": [1000],
        "batch_size": 1,
        "dfake_gen_update_ratio": 5,
        "gan_g_weight": 0.03,
        "gan_d_weight": 0.03,
        "seed": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Official main changed {key}: {config.get(key)!r}")
    if any(key in config for key in ("rollout_schedule", "first_rollout_num_frames", "first_frame_denoising_step_list")):
        raise ValueError("Official main must not use FFE during training")
    return config


def prepare(official_root: Path, pair_dir: Path, seed: int | None = None) -> dict:
    official_root = official_root.resolve()
    pair_dir = pair_dir.resolve()
    original = official_config(official_root)
    # Official main draws one seed in [0, 10^7) when seed=0. Draw one
    # nonzero value externally so both arms receive the same effective seed
    # without re-triggering its seed=0 sentinel.
    if seed is None:
        seed = secrets.randbelow(9_999_999) + 1
    if not 1 <= seed <= 9_999_999:
        raise ValueError("Shared effective seed must be in [1, 9999999]")
    if pair_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pair directory: {pair_dir}")
    pair_dir.mkdir(parents=True)
    full = copy.deepcopy(original)
    full["seed"] = seed
    dmd = copy.deepcopy(full)
    dmd.update({key: 0.0 for key in GAN_KEYS})
    for arm, config in (("full", full), ("dmd", dmd)):
        (pair_dir / CONFIG_NAMES[arm]).write_text(
            OmegaConf.to_yaml(OmegaConf.create(config), sort_keys=False), encoding="utf-8"
        )
    manifest = {
        "schema_version": 1,
        "official_main_commit": OFFICIAL_COMMIT,
        "official_root": str(official_root),
        "official_config_sha256": _hash(official_root / "config.yaml"),
        "shared_effective_seed": seed,
        "seed_policy": "one nonzero draw from the official runtime-seed range, shared by both arms",
        "configs_sha256": {arm: _hash(pair_dir / CONFIG_NAMES[arm]) for arm in ARMS},
        "only_arm_difference": list(GAN_KEYS),
    }
    (pair_dir / "pair_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return check(official_root, pair_dir)


def check(official_root: Path, pair_dir: Path) -> dict:
    official_root = official_root.resolve()
    pair_dir = pair_dir.resolve()
    original = official_config(official_root)
    manifest = json.loads((pair_dir / "pair_manifest.json").read_text(encoding="utf-8"))
    seed = manifest["shared_effective_seed"]
    if not isinstance(seed, int) or not 1 <= seed <= 9_999_999:
        raise ValueError("Invalid shared effective seed")
    full, dmd = (_config(pair_dir / CONFIG_NAMES[arm]) for arm in ARMS)
    expected_full = dict(original, seed=seed)
    expected_dmd = dict(expected_full, **{key: 0.0 for key in GAN_KEYS})
    if full != expected_full:
        raise ValueError("DMD+GAN config differs from official main except for the shared runtime seed")
    if dmd != expected_dmd:
        raise ValueError("DMD-only config differs from official main beyond seed and GAN weights")
    expected_manifest = {
        "schema_version": 1,
        "official_main_commit": OFFICIAL_COMMIT,
        "official_root": str(official_root),
        "official_config_sha256": _hash(official_root / "config.yaml"),
        "shared_effective_seed": seed,
        "seed_policy": "one nonzero draw from the official runtime-seed range, shared by both arms",
        "configs_sha256": {arm: _hash(pair_dir / CONFIG_NAMES[arm]) for arm in ARMS},
        "only_arm_difference": list(GAN_KEYS),
    }
    if manifest != expected_manifest:
        raise ValueError("Pair manifest differs from the official-code and config audit")
    return manifest


def official_command(
    python: str, config_path: Path, run_dir: Path, asset_paths: dict,
) -> list[str]:
    """The official README's torchrun invocation with paired absolute paths."""
    return [
        python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
        "train.py", "--config_path", str(config_path.resolve()),
        "--generator_ckpt", asset_paths["generator_ckpt"],
        "--teacher_model_path", asset_paths["teacher_model_path"],
        "--data_path", asset_paths["data_path"],
        "--logdir", str(run_dir.resolve()), "--disable-wandb", "--no_visualize",
    ]


def _audit_arm(arm: str, run_dir: Path, eval_root: Path, pair_dir: Path, manifest: dict) -> dict:
    run_dir = run_dir.resolve()
    checkpoint = run_dir / "checkpoint_model_000200/model.pt"
    launch = json.loads((run_dir / "official_launch.json").read_text(encoding="utf-8"))
    expected_launch = {
        "arm": arm,
        "official_main_commit": OFFICIAL_COMMIT,
        "config_path": str((pair_dir / CONFIG_NAMES[arm]).resolve()),
        "config_sha256": manifest["configs_sha256"][arm],
        "world_size": 8,
        "run_dir": str(run_dir),
    }
    for key, value in expected_launch.items():
        if launch.get(key) != value:
            raise ValueError(f"{arm}: launch {key} does not match: {launch.get(key)!r}")
    if launch.get("command") is None or launch.get("asset_paths") is None:
        raise ValueError(f"{arm}: missing executed command or asset paths")
    expected_command = official_command(
        launch["python"], pair_dir / CONFIG_NAMES[arm], run_dir, launch["asset_paths"]
    )
    if launch["command"] != expected_command:
        raise ValueError(f"{arm}: command differs from the official-main invocation")
    records = []
    for line in (run_dir / "train.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("{'step':") and "'critic_loss':" in line:
            record = ast.literal_eval(line)
            if not all(
                math.isfinite(value) for value in record.values()
                if isinstance(value, (int, float))
            ):
                raise ValueError(f"{arm}: non-finite training metric at step {record.get('step')}")
            records.append(record)
    if [record["step"] for record in records] != list(range(1, 201)):
        raise ValueError(f"{arm}: official training log does not contain finite steps 1–200")
    checkpoint_audit = audit_checkpoint(checkpoint, expected_step=200)
    evaluation = audit_run(eval_root.resolve(), RUN_NAMES[arm], 5, "ffe", "generator")
    if Path(evaluation["checkpoint_path"]).resolve() != checkpoint.resolve():
        raise ValueError(f"{arm}: evaluated a different checkpoint")
    if evaluation["checkpoint_size_bytes"] != checkpoint_audit["checkpoint_size_bytes"]:
        raise ValueError(f"{arm}: checkpoint size changed between training and evaluation")
    return {"launch": launch, "checkpoint": checkpoint_audit, "evaluation": evaluation}


def report(
    official_root: Path, pair_dir: Path, full_run_dir: Path, dmd_run_dir: Path,
    full_eval_root: Path, dmd_eval_root: Path,
) -> dict:
    manifest = check(official_root, pair_dir)
    arms = {
        "full": _audit_arm("full", full_run_dir, full_eval_root, pair_dir, manifest),
        "dmd": _audit_arm("dmd", dmd_run_dir, dmd_eval_root, pair_dir, manifest),
    }
    for key in ("generator_ckpt", "teacher_model_path", "data_path", "wan_1_3b_path", "cuda_visible_devices"):
        left = arms["full"]["launch"]["asset_paths"].get(key)
        right = arms["dmd"]["launch"]["asset_paths"].get(key)
        if not left or left != right:
            raise ValueError(f"Arms did not share {key}: {left!r} != {right!r}")
    if arms["full"]["launch"].get("asset_stats") != arms["dmd"]["launch"].get("asset_stats"):
        raise ValueError("Assets changed between the two training arms")
    if arms["full"]["launch"].get("rebuttal_commit") != arms["dmd"]["launch"].get("rebuttal_commit"):
        raise ValueError("The two arms were launched from different rebuttal revisions")
    first = arms["full"]["evaluation"]["protocol"]
    second = arms["dmd"]["evaluation"]["protocol"]
    for key in PAIRED_FIELDS:
        if first.get(key) != second.get(key):
            raise ValueError(f"Evaluation protocol differs in {key}")
    scores = {
        arm: {key: arms[arm]["evaluation"]["normalized_aggregates"][key] * 100 for key in SCORE_KEYS}
        for arm in ARMS
    }
    dimensions = {arm: arms[arm]["evaluation"]["scores"] for arm in ARMS}
    if set(dimensions["full"]) != set(dimensions["dmd"]) or len(dimensions["full"]) != 16:
        raise ValueError("The two evaluations do not share all 16 VBench dimensions")
    return {
        "status": "pass",
        "training_source": f"Aurora-edu/One-Forcing main@{OFFICIAL_COMMIT}",
        "shared_effective_seed": manifest["shared_effective_seed"],
        "only_arm_difference": list(GAN_KEYS),
        "dmd_gan": scores["full"],
        "dmd_only": scores["dmd"],
        "gan_gain": {key: scores["full"][key] - scores["dmd"][key] for key in SCORE_KEYS},
        "dimensions": dimensions,
        "eval_protocol": first,
        "checkpoints": {arm: arms[arm]["checkpoint"] for arm in ARMS},
    }


def render_markdown(result: dict) -> str:
    def row(name: str, values: dict) -> str:
        return f"| {name} | " + " | ".join(f"{values[key]:.2f}" for key in SCORE_KEYS) + " |"

    lines = [
        "# Official-main-code paired GAN ablation", "",
        f"Training code: `{result['training_source']}`. Both arms use effective seed "
        f"`{result['shared_effective_seed']}` and differ only in the two GAN weights. "
        "Both are evaluated with raw generator weights and the same pinned 944×5 Qwen/FFE "
        "full-16 VBench protocol.", "",
        "| Arm | Total | Quality | Semantic |", "|---|---:|---:|---:|",
        row("DMD+GAN", result["dmd_gan"]),
        row("DMD-only", result["dmd_only"]),
        row("GAN gain", result["gan_gain"]), "",
        "The published 83.76 is a separate reference checkpoint, not a substituted arm in this pair.", "",
        "| VBench dimension | DMD+GAN | DMD-only | Gain |",
        "|---|---:|---:|---:|",
    ]
    for dimension in sorted(result["dimensions"]["full"]):
        full = result["dimensions"]["full"][dimension]
        dmd = result["dimensions"]["dmd"][dimension]
        lines.append(f"| {dimension} | {full:.4f} | {dmd:.4f} | {full - dmd:+.4f} |")
    return "\n".join([*lines, ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "check", "report"):
        item = sub.add_parser(command)
        item.add_argument("--official_root", type=Path, required=True)
        item.add_argument("--pair_dir", type=Path, required=True)
        if command == "prepare":
            item.add_argument("--seed", type=int)
        if command == "report":
            for name in ("full_run_dir", "dmd_run_dir", "full_eval_root", "dmd_eval_root"):
                item.add_argument(f"--{name}", type=Path, required=True)
            item.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.official_root, args.pair_dir, args.seed)
    elif args.command == "check":
        result = check(args.official_root, args.pair_dir)
    else:
        result = report(
            args.official_root, args.pair_dir, args.full_run_dir, args.dmd_run_dir,
            args.full_eval_root, args.dmd_eval_root,
        )
        prefix = args.output_prefix.resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs = {
            Path(str(prefix) + ".json"): json.dumps(result, indent=2, sort_keys=True) + "\n",
            Path(str(prefix) + ".md"): render_markdown(result),
        }
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"Refusing to replace an existing report: {path}")
        for path, content in outputs.items():
            path.write_text(content, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
