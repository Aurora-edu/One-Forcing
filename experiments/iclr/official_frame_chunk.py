"""Frame/chunk pair executed by an untouched, pinned official checkout.

This is an external launcher/auditor, not a Trainer subclass. GPU operations
require --execute and run in tmux; prepare/check/plans use CPU only.
"""
import argparse
import ast
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from omegaconf import OmegaConf

from experiments.iclr.launch import parse_gpus, require_idle
from experiments.iclr.protocol import REPO, atomic_json, sha256_file
from experiments.rebuttal.paper_vbench_protocol import ASSETS, pinned_inputs

OFFICIAL = "c9a2350e8740531562011fc9618e1a928d911ae0"
ARMS = {"frame": 1, "chunk": 3}
SEED = 42
SAMPLES = 5


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def read_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def official_source(root):
    root = Path(root).resolve()
    if git(root, "rev-parse", "HEAD") != OFFICIAL:
        raise ValueError(f"Official checkout must be pinned at {OFFICIAL}")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official tracked files changed; even EMA export patches are forbidden")
    # Include byte checks: git status alone can miss assume-unchanged flags.
    files = git(root, "ls-tree", "-r", "--name-only", OFFICIAL).splitlines()
    for name in files:
        expected = git(root, "rev-parse", f"{OFFICIAL}:{name}")
        if not (root / name).is_file() or git(root, "hash-object", "--no-filters", name) != expected:
            raise ValueError(f"Official source content changed: {name}")
    extras = git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    if any(Path(name).suffix in {".py", ".so", ".pth"} for name in extras):
        raise ValueError("Untracked executable/import files in official checkout")
    return read_yaml(root / "config.yaml")


def split_prompts(prompts, rewrites):
    if len(prompts) != 944 or len(rewrites) != 944 or len(set(prompts)) != 944:
        raise ValueError("Require the complete, unique 944-prompt protocol")
    if any(any(c in p for c in ("/", "\x00", "\n", "\r")) for p in prompts):
        raise ValueError("Prompt cannot be used with official raw_prompt_index filenames")
    return [{"shard": i, "start": i * 118, "stop": (i + 1) * 118,
             "process_seed": i * 118 * SAMPLES,
             "prompts": prompts[i * 118:(i + 1) * 118],
             "rewrites": rewrites[i * 118:(i + 1) * 118]} for i in range(8)]


def prepare(root, pair):
    root, pair = Path(root).resolve(), Path(pair).resolve()
    if pair.exists():
        raise FileExistsError(f"Pair directory exists: {pair}")
    if not root.exists():
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(root), OFFICIAL], check=True)
    original = official_source(root)
    pair.mkdir(parents=True)
    (pair / "configs").mkdir()
    for arm, block in ARMS.items():
        cfg = dict(original, seed=SEED, num_frame_per_block=block)
        OmegaConf.save(OmegaConf.create(cfg), pair / "configs" / f"{arm}.yaml")
        # Same warm-up as official scripts/infer.sh, with native block size.
        cfg["first_frame_denoising_step_list"] = [1000, 750, 500, 250]
        OmegaConf.save(OmegaConf.create(cfg), pair / "configs" / f"{arm}_infer.yaml")
    prompts, rewrites = pinned_inputs()
    (pair / "inputs").mkdir()
    for shard in split_prompts(prompts, rewrites):
        for kind in ("prompts", "rewrites"):
            (pair / "inputs" / f"{kind}_{shard['shard']}.txt").write_text("\n".join(shard[kind]) + "\n")
    manifest = {"schema_version": 1, "official_commit": OFFICIAL, "official_root": str(root),
                "seed": SEED, "steps": 200, "only_arm_difference": ["num_frame_per_block"],
                "weight_source": "generator", "samples_per_prompt": SAMPLES,
                "training_configs": {arm: sha256_file(pair / "configs" / f"{arm}.yaml") for arm in ARMS},
                "inference_configs": {arm: sha256_file(pair / "configs" / f"{arm}_infer.yaml") for arm in ARMS},
                "input_hashes": {p.name: sha256_file(p) for p in sorted((pair / "inputs").iterdir())},
                "full_info_sha256": sha256_file(ASSETS / "VBench_full_info.json"),
                "effective_gradient_frames": {"frame": 17, "chunk": 21},
                "first_block_ffe_latent_frames": ARMS,
                "rng_protocol": "official exporter, 8 contiguous prompt shards; initial seed=5*global_prompt_index+sample; intermediate noise uses each shard's global RNG"}
    atomic_json(pair / "pair.json", manifest)
    return check(pair)


def check(pair):
    pair = Path(pair).resolve()
    manifest = json.loads((pair / "pair.json").read_text())
    root = Path(manifest["official_root"])
    original = official_source(root)
    fixed = {"official_commit": OFFICIAL, "seed": SEED, "steps": 200,
             "only_arm_difference": ["num_frame_per_block"], "weight_source": "generator",
             "samples_per_prompt": SAMPLES, "effective_gradient_frames": {"frame": 17, "chunk": 21},
             "first_block_ffe_latent_frames": ARMS}
    if any(manifest.get(k) != v for k, v in fixed.items()):
        raise ValueError("Pair protocol changed")
    for arm, block in ARMS.items():
        expected = dict(original, seed=SEED, num_frame_per_block=block)
        for suffix, field in (("", "training_configs"), ("_infer", "inference_configs")):
            cfg = pair / "configs" / f"{arm}{suffix}.yaml"
            if suffix:
                expected = dict(expected, first_frame_denoising_step_list=[1000, 750, 500, 250])
            if read_yaml(cfg) != expected or sha256_file(cfg) != manifest[field][arm]:
                raise ValueError(f"{arm}{suffix}: config differs from official recipe")
    prompts, rewrites = pinned_inputs()
    expected_hashes = {}
    for shard in split_prompts(prompts, rewrites):
        for kind in ("prompts", "rewrites"):
            p = pair / "inputs" / f"{kind}_{shard['shard']}.txt"
            if p.read_text() != "\n".join(shard[kind]) + "\n":
                raise ValueError(f"Changed evaluation input: {p}")
            expected_hashes[p.name] = sha256_file(p)
    if expected_hashes != manifest["input_hashes"] or manifest["full_info_sha256"] != sha256_file(ASSETS / "VBench_full_info.json"):
        raise ValueError("Evaluation input hashes changed")
    return manifest


def train_command(pair, arm, python, assets):
    return [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", "train.py",
            "--config_path", str(pair / "configs" / f"{arm}.yaml"),
            "--generator_ckpt", assets["generator"], "--teacher_model_path", assets["teacher"],
            "--data_path", assets["data"], "--logdir", str(pair / "runs" / arm),
            "--disable-wandb", "--no_visualize"]


def export_commands(pair, arm, python):
    commands = []
    for i in range(8):
        commands.append([python, "scripts/export_videos.py", "--config_path", str(pair / "configs" / f"{arm}_infer.yaml"),
            "--checkpoint_path", str(pair / "runs" / arm / "checkpoint_model_000200/model.pt"),
            "--prompt_path", str(pair / "inputs" / f"prompts_{i}.txt"),
            "--extended_prompt_path", str(pair / "inputs" / f"rewrites_{i}.txt"),
            "--output_folder", str(pair / "eval" / arm / "shards" / str(i)),
            "--num_output_frames", "21", "--seed", str(i * 118 * SAMPLES),
            "--num_samples_per_prompt", str(SAMPLES), "--naming", "raw_prompt_index"])
    return commands


def metric_records(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        start = line.find("{'step':")
        if start < 0 or "'critic_loss':" not in line:
            continue
        row = ast.literal_eval(line[start:])
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise ValueError(f"Nonfinite metric at step {row['step']}")
        rows.append(row)
    if [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Training logs have missing, duplicate or out-of-order steps")
    return rows


def monitor(pair, arm, min_step=10):
    rows = metric_records(pair / "runs" / arm / "train.log")
    if len(rows) < min_step:
        raise ValueError(f"{arm}: only {len(rows)} completed iterations; need {min_step}")
    updates = [r["step"] for r in rows if "generator_loss" in r]
    if updates != list(range(1, len(rows) + 1, 5)):
        raise ValueError("Unexpected generator update cadence")
    return {"arm": arm, "last_step": len(rows), "generator_updates": len(updates),
            "critic_updates": len(rows), "finite": True}


def raw_checkpoint(path, initialization):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    reference = torch.load(initialization, map_location="cpu", weights_only=True, mmap=True)["generator"]
    state = payload.get("generator")
    if not isinstance(state, dict) or not state or state.keys() != reference.keys():
        raise ValueError("Final raw generator keys do not match initialization; no strict=False fallback")
    for key, tensor in state.items():
        if tensor.shape != reference[key].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid raw tensor: {key}")
    return {"weight_source": "generator", "use_ema": False, "num_tensors": len(state),
            "checkpoint_sha256": sha256_file(path), "checkpoint_size": Path(path).stat().st_size}


def clean_env(gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)  # Prevent importing this checkout's model/trainer.
    return env


def runtime_versions():
    record = {"python": sys.version, "executable": sys.executable}
    for package in ("torch", "torchvision", "numpy", "transformers", "diffusers", "flash-attn"):
        try:
            record[package] = version(package)
        except PackageNotFoundError:
            record[package] = None
    return record


def assets_for(args):
    return {k: str(Path(getattr(args, k)).expanduser().resolve()) for k in ("generator", "teacher", "data", "wan")}


def lock_assets(pair, root, assets):
    for key in ("generator", "teacher", "data", "wan"):
        if not Path(assets[key]).exists():
            raise FileNotFoundError(assets[key])
    for p in (Path(assets["data"]) / "data.mdb", Path(assets["wan"]) / "models_t5_umt5-xxl-enc-bf16.pth",
              Path(assets["wan"]) / "Wan2.1_VAE.pth"):
        if not p.is_file():
            raise FileNotFoundError(p)
    link = root / "wan_models/Wan2.1-T2V-1.3B"
    if link.exists() or link.is_symlink():
        if link.resolve() != Path(assets["wan"]):
            raise ValueError("Official checkout points to different Wan assets")
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(assets["wan"], target_is_directory=True)
    inventory = {}
    for key, value in assets.items():
        p = Path(value)
        paths = [p] if p.is_file() else sorted(p.rglob("*"))
        inventory[key] = {str(f.relative_to(p) if p.is_dir() else f.name):
                          [f.stat().st_size, f.stat().st_mtime_ns]
                          for f in paths if f.is_file() and f.name != "lock.mdb" and ".cache" not in f.parts}
    record = {"paths": assets, "inventory_size_mtime": inventory, "generator_sha256": sha256_file(assets["generator"])}
    path = pair / "assets.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Assets changed between paired runs")
    if not path.exists():
        atomic_json(path, record)
    return record


def worker_train(args):
    pair = Path(args.pair).resolve()
    manifest = check(pair)
    root = Path(manifest["official_root"])
    assets = assets_for(args)
    require_idle(parse_gpus(args.gpus))
    lock_assets(pair, root, assets)
    check(pair)
    environment = runtime_versions()
    env_path = pair / "training_environment.json"
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError("Training Python/package versions changed between arms")
    if not env_path.exists():
        atomic_json(env_path, environment)
    run = pair / "runs" / args.arm
    run.mkdir(parents=True, exist_ok=False)
    command = train_command(pair, args.arm, args.python, assets)
    atomic_json(run / "launch.json", {"official_commit": OFFICIAL, "working_directory": str(root),
        "command": command, "config_sha256": manifest["training_configs"][args.arm],
        "launcher_commit": git(REPO, "rev-parse", "HEAD"), "launcher_sha256": sha256_file(__file__),
        "assets": assets, "gpus": args.gpus, "seed": SEED, "weight_source": "generator",
        "environment": environment})
    started = time.monotonic()
    with (run / "train.log").open("x") as log:
        process = subprocess.Popen(command, cwd=root, env=clean_env(args.gpus),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                if "'step': 10," in line and "'critic_loss':" in line:
                    atomic_json(run / "step10_health.json", monitor(pair, args.arm, 10))
                    print(f"{args.arm}: step 10 finite; inspect {run / 'train.log'}", flush=True)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()
    if code:
        raise RuntimeError(f"Official training exited {code}; original logs retained")
    health = monitor(pair, args.arm, 200)
    if health["last_step"] != 200 or not (run / "step10_health.json").is_file():
        raise ValueError("Expected exactly 200 steps and a step-10 health record")
    check(pair)
    audit = raw_checkpoint(run / "checkpoint_model_000200/model.pt", assets["generator"])
    atomic_json(run / "training.done", {**health, **audit, "elapsed_seconds": time.monotonic() - started,
                                        "official_commit": OFFICIAL, "seed": SEED})


def audit_training(pair, arm, manifest):
    run = pair / "runs" / arm
    launch_record = json.loads((run / "launch.json").read_text())
    assets = json.loads((pair / "assets.json").read_text())["paths"]
    command = launch_record["command"]
    expected = train_command(pair, arm, command[0], assets)
    if command != expected or launch_record["assets"] != assets:
        raise ValueError("Executed training command or paired assets changed")
    if launch_record["environment"] != json.loads((pair / "training_environment.json").read_text()):
        raise ValueError("Training runtime differs between arms")
    for key, value in {"official_commit": OFFICIAL, "seed": SEED, "weight_source": "generator",
                       "working_directory": manifest["official_root"],
                       "config_sha256": manifest["training_configs"][arm]}.items():
        if launch_record.get(key) != value:
            raise ValueError(f"Training receipt mismatch: {key}")
    health = monitor(pair, arm, 200)
    done = json.loads((run / "training.done").read_text())
    required = {**health, "last_step": 200, "generator_updates": 40, "critic_updates": 200,
                "official_commit": OFFICIAL, "seed": SEED, "weight_source": "generator", "use_ema": False}
    if any(done.get(k) != v for k, v in required.items()) or health["last_step"] != 200:
        raise ValueError("Training completion does not match the official pair")
    if not (run / "step10_health.json").is_file():
        raise ValueError("Missing step-10 health record")
    return done


def worker_eval(args):
    pair = Path(args.pair).resolve()
    manifest = check(pair)
    root = Path(manifest["official_root"])
    gpus = parse_gpus(args.gpus)
    require_idle(gpus)
    run = pair / "runs" / args.arm
    completed = audit_training(pair, args.arm, manifest)
    checkpoint = run / "checkpoint_model_000200/model.pt"
    if sha256_file(checkpoint) != completed["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after training audit")
    output = pair / "eval" / args.arm
    output.mkdir(parents=True, exist_ok=False)
    commands = export_commands(pair, args.arm, args.python)
    atomic_json(output / "export_plan.json", {"commands": commands, "official_commit": OFFICIAL,
        "weight_source": "generator", "use_ema": False, "checkpoint_sha256": completed["checkpoint_sha256"],
        "rng_protocol": manifest["rng_protocol"], "protocol": manifest})
    processes, streams = [], []
    try:
        for i, cmd in enumerate(commands):
            log = (output / f"export_{i}.log").open("x"); streams.append(log)
            processes.append(subprocess.Popen(cmd, cwd=root, env=clean_env(gpus[i]), stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Official export failed: {codes}; no partial VBench scoring")
    finally:
        # These are only the processes created above, never another session's jobs.
        for process in processes:
            if process.poll() is None:
                process.terminate(); process.wait()
        for stream in streams:
            stream.close()
    check(pair)
    prompts, rewrites = pinned_inputs()
    videos = output / "videos"; videos.mkdir()
    for shard in split_prompts(prompts, rewrites):
        directory = output / "shards" / str(shard["shard"])
        names = {f"{p}-{s}.mp4" for p in shard["prompts"] for s in range(SAMPLES)}
        if (directory / "export.done").read_text().strip() != "ok" or {p.name for p in directory.glob("*.mp4")} != names:
            raise ValueError("Incomplete official export shard")
        import av
        for name in sorted(names):
            source = directory / name
            if not source.stat().st_size:
                raise ValueError(f"Empty video: {source}")
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                if stream.frames != 81 or float(stream.average_rate) != 16:
                    raise ValueError(f"Video metadata must be 81 frames / 16 fps: {source}")
            (videos / name).symlink_to(source)
    atomic_json(output / "visual_samples.json", {
        "selection_fixed_before_scores": True,
        "pairs": [{"prompt_index": i, "sample_index": i % SAMPLES,
                   "initial_noise_seed": i * SAMPLES + i % SAMPLES,
                   "video": str(videos / f"{prompts[i]}-{i % SAMPLES}.mp4")}
                  for i in range(0, 944, 59)],
        "inspection_status": "pending: metadata validation is not visual inspection"})
    atomic_json(output / "export.done", {"num_videos": 944 * SAMPLES, "weight_source": "generator",
        "use_ema": False, "official_commit": OFFICIAL, "checkpoint_sha256": completed["checkpoint_sha256"]})
    require_idle(gpus)
    scoring = [args.vbench_python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
        str(REPO / "scripts/run_vbench.py"), "--videos_path", str(videos),
        "--full_info_path", str(ASSETS / "VBench_full_info.json"), "--output_dir", str(output / "vbench"),
        "--name", args.arm, "--device", "cuda", "--samples_per_prompt", str(SAMPLES)]
    atomic_json(output / "scoring_command.json", {"command": scoring})
    subprocess.run(scoring, cwd=REPO, env=clean_env(args.gpus), check=True)
    print(f"Completed {args.arm}; inspect actual MP4s before drawing conclusions: {videos}", flush=True)


def report(pair):
    from experiments.rebuttal.consolidate_results import load_vbench_results, official_totals
    pair = Path(pair).resolve()
    manifest = check(pair)
    rows = []
    for arm in ARMS:
        done = audit_training(pair, arm, manifest)
        output = pair / "eval" / arm
        exported = json.loads((output / "export.done").read_text())
        expected = {"num_videos": 944 * SAMPLES, "weight_source": "generator", "use_ema": False,
                    "official_commit": OFFICIAL, "checkpoint_sha256": done["checkpoint_sha256"]}
        plan = json.loads((output / "export_plan.json").read_text())
        if exported != expected or plan["protocol"] != manifest:
            raise ValueError("Evaluation provenance does not match paired training")
        if plan["commands"] != export_commands(pair, arm, plan["commands"][0][0]):
            raise ValueError("Official exporter arguments changed")
        protocol = json.loads((output / "vbench" / f"{arm}_protocol.json").read_text())
        if protocol.get("mode") != "vbench_standard" or protocol.get("samples_per_prompt") != SAMPLES:
            raise ValueError("VBench protocol changed")
        scores = load_vbench_results(output / "vbench" / f"{arm}_eval_results.json")
        totals = official_totals(scores)
        if len(scores) != 16 or totals is None or any(not math.isfinite(v) for v in scores.values()):
            raise ValueError("Incomplete VBench scores")
        rows.append({"arm": arm, "scores": scores, "totals": totals,
                     "training_elapsed_seconds": done["elapsed_seconds"]})
    result = {"protocol": manifest, "runs": rows,
              "frame_minus_chunk_percentage_points": {k: 100 * (rows[0]["totals"][k] - rows[1]["totals"][k]) for k in rows[0]["totals"]},
              "note": "One paired training seed; raw weights. Do not merge with historical EMA results. Visual review remains mandatory."}
    atomic_json(pair / "summary.json", result)
    return result


def launch(args):
    pair = Path(args.pair).resolve()
    manifest = check(pair)
    commands = ([train_command(pair, args.arm, args.python, assets_for(args))] if args.action == "train"
                else export_commands(pair, args.arm, args.python))
    for cmd in commands:
        print(f"cd {shlex.quote(manifest['official_root'])} && {shlex.join(cmd)}")
    if not args.execute:
        print("DRY RUN: official source only; no GPU query, model load or process launch")
        return
    require_idle(parse_gpus(args.gpus))
    destination = pair / ("runs" if args.action == "train" else "eval") / args.arm
    if destination.exists():
        raise FileExistsError(f"Refuse existing output: {destination}")
    session = args.session or f"official_{args.action}_{args.arm}_s{SEED}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("Invalid tmux session name")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise FileExistsError(f"Existing tmux session: {session}")
    argv = [args.python, "-m", "experiments.iclr.official_frame_chunk", args.action,
            "--pair", str(pair), "--arm", args.arm, "--python", args.python, "--gpus", args.gpus, "--worker"]
    if args.action == "train":
        for key, value in assets_for(args).items():
            argv += [f"--{key}", value]
    else:
        argv += ["--vbench-python", args.vbench_python]
    logs = pair / "logs"; logs.mkdir(exist_ok=True)
    shell = f"cd {shlex.quote(str(REPO))} && exec {shlex.join(argv)} >> {shlex.quote(str(logs / (session + '.log')))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
    print(f"tmux attach -t {session}; inspect training through step 10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--official-root", required=True)
    for name in ("check", "report", "monitor", "train", "evaluate"):
        p = sub.add_parser(name)
        if name not in ("check", "report"):
            p.add_argument("--arm", choices=ARMS, required=True)
        if name == "monitor":
            p.add_argument("--min-step", type=int, default=10)
        if name in ("train", "evaluate"):
            p.add_argument("--python", default=sys.executable)
            p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
            p.add_argument("--session", default="")
            mode = p.add_mutually_exclusive_group()
            mode.add_argument("--execute", action="store_true")
            mode.add_argument("--dry-run", action="store_true")
            p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
            if name == "train":
                for key in ("generator", "teacher", "data", "wan"):
                    p.add_argument(f"--{key}", required=True)
            else:
                p.add_argument("--vbench-python", required=True)
    for p in sub.choices.values():
        p.add_argument("--pair", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.official_root, args.pair), indent=2))
    elif args.action == "check":
        print(json.dumps(check(args.pair), indent=2))
    elif args.action == "report":
        print(json.dumps(report(args.pair), indent=2))
    elif args.action == "monitor":
        print(json.dumps(monitor(Path(args.pair).resolve(), args.arm, args.min_step), indent=2))
    elif args.worker:
        if not os.environ.get("TMUX"):
            raise RuntimeError("Long-running worker requires its own tmux session")
        (worker_train if args.action == "train" else worker_eval)(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
