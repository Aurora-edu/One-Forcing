"""Paired held-out history diagnostics for endpoint-anchored CD, never VBench.

Both paths use the same framewise KV sampler. The teacher-forced path changes
only the clean history cached AFTER a frame is predicted. Targets are the
stored teacher ODE endpoints, not a claim of optimal targets for free histories.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys

from experiments.iclr.cd_protocol import SAVE_STEPS, validate_checkpoint, validate_done_pair
from experiments.iclr.protocol import REPO, atomic_json, sha256_file


def frame_errors(forced, free, target):
    import torch
    if forced.shape != free.shape or free.shape != target.shape or target.ndim != 5:
        raise ValueError("diagnostic tensors must share [B,F,C,H,W] shape")
    if not all(torch.isfinite(value).all() for value in (forced, free, target)):
        raise ValueError("nonfinite diagnostic prediction")
    # The first frame has no past, so changing history must not change it.
    if not torch.allclose(forced[:, 0], free[:, 0], atol=1e-5, rtol=1e-5):
        raise ValueError("first-frame predictions differ without any history")
    axes = (0, 2, 3, 4)
    tf = (forced.float() - target.float()).square().mean(axes)
    ar = (free.float() - target.float()).square().mean(axes)
    shift = (free.float() - forced.float()).square().mean(axes)
    return [{"frame": i, "forced_mse": float(tf[i]), "free_mse": float(ar[i]),
             "history_gap": float(ar[i] - tf[i]), "prediction_shift": float(shift[i])}
            for i in range(target.shape[1])]


def summarize_records(records, steps, row_indices):
    """Require complete paired coverage; negative gaps/gains remain valid."""
    import math
    expected = {(arm, step, row, frame) for arm in ("curved", "rectified")
                for step in steps for row in row_indices for frame in range(21)}
    seen = set()
    identities = {}
    groups = {}
    metrics = ("forced_mse", "free_mse", "history_gap", "prediction_shift")
    for record in records:
        key = tuple(record[k] for k in ("arm", "step", "row_index", "frame"))
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected diagnostic record")
        seen.add(key)
        identity = (record["prompt"], record["noise_seed"])
        if identities.setdefault(record["row_index"], identity) != identity:
            raise ValueError("diagnostic prompts/noises are not paired")
        if not all(math.isfinite(record[k]) for k in metrics):
            raise ValueError("nonfinite diagnostic metric")
        groups.setdefault(key[:2], []).append(record)
    if seen != expected:
        raise ValueError("incomplete diagnostic coverage")
    rows = [{"arm": arm, "step": step,
             **{k: sum(r[k] for r in group) / len(group) for k in metrics}}
            for (arm, step), group in sorted(groups.items())]
    by_key = {(r["arm"], r["step"]): r for r in rows}
    deltas = [{"step": step, **{k: by_key["rectified", step][k] - by_key["curved", step][k]
                                for k in metrics}} for step in steps]
    return rows, deltas


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output, training_root, steps, row_indices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = []
    for path in sorted(output.glob("rank_*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    rows, deltas = summarize_records(records, steps, row_indices)
    write_csv(output / "per_frame.csv", records)
    write_csv(output / "history_curves.csv", rows)
    write_csv(output / "rectified_minus_original.csv", deltas)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for arm in ("curved", "rectified"):
        subset = [row for row in rows if row["arm"] == arm]
        for ax, key in zip(axes[0], ("forced_mse", "free_mse", "history_gap")):
            ax.plot([r["step"] for r in subset], [r[key] for r in subset], marker="o", label=arm)
            ax.set(xlabel="Training updates", ylabel=key)
        logs = [json.loads(line) for line in (training_root / arm / "metrics.jsonl").read_text().splitlines()]
        for ax, key in zip(axes[1], ("cd_loss", "endpoint_loss", "grad_norm")):
            ax.plot([r["step"] for r in logs], [r[key] for r in logs], alpha=.8, label=arm)
            ax.set(xlabel="Training updates", ylabel=key)
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)
    atomic_json(output / "summary.json", {"status": "complete", "curves": rows, "paired_deltas": deltas,
        "score": "latent MSE, not VBench; lower MSE need not mean better video",
        "interpretation": "history_gap is sensitivity to replacing past context under a fixed teacher-forced target; not a causal decomposition",
        "video_review": "pending human inspection; no metric certifies visual quality"})


def run_worker(args):
    # Delay GPU and Wan imports so --help/--dry-run work on CPU-only machines.
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from experiments.rebuttal.curvature_intervention import CurvatureTrajectoryDataset, MANIFEST_NAME
    from pipeline.causal_inference import CausalInferencePipeline
    from scripts.export_videos import (CachedPromptTextEncoder, load_generator_state,
                                       write_video_with_fallback, validate_video)
    from utils.config import load_config
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

    if not os.environ.get("TMUX"):
        raise RuntimeError("GPU diagnostics require tmux")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    training_root, output = Path(args.training_root), Path(args.output)
    done = {arm: json.loads((training_root / arm / "training.done").read_text())
            for arm in ("curved", "rectified")}
    controls = validate_done_pair(done["curved"], done["rectified"], mode="smoke" if args.smoke else "formal")
    manifest_hash = sha256_file(Path(args.data) / MANIFEST_NAME)
    if controls["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("diagnostic data differs from the shared training dataset")
    steps = (0, 5, 10) if args.smoke else SAVE_STEPS
    rows = list(range(8 if args.smoke else 64))
    video_rows = 2 if args.smoke else 4
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "videos").mkdir()
        atomic_json(output / "intent.json", {"controls": controls, "steps": steps, "row_indices": rows,
            "sampling": "raw/all1/noFFE/21_frames/original_prompts", "videos_at_step": steps[-1],
            "video_row_indices": rows[:video_rows], "mode": "smoke" if args.smoke else "formal"})
    dist.barrier()
    dataset = CurvatureTrajectoryDataset(args.data, "curved")
    config = load_config(str(REPO / "experiments/rebuttal/configs/eval_all1.yaml"))
    config.denoising_step_list = [1000]
    config.context_noise = 0
    config.independent_first_frame = False
    if "first_frame_denoising_step_list" in config:
        del config.first_frame_denoising_step_list
    generator = WanDiffusionWrapper(**OmegaConf.to_container(config.model_kwargs), is_causal=True)
    generator.model.num_frame_per_block = 1
    generator.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    encoder = CachedPromptTextEncoder(args.prompt_cache) if args.prompt_cache else WanTextEncoder()
    encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    # Only ranks exporting videos need a VAE on their GPU.
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16) if rank < video_rows else torch.nn.Identity()
    pipeline = CausalInferencePipeline(config, device, generator=generator, text_encoder=encoder, vae=vae).eval()
    with torch.inference_mode(), (output / f"rank_{rank}.jsonl").open("x") as stream:
        for arm in ("curved", "rectified"):
            for step in steps:
                checkpoint = training_root / arm / f"checkpoint_model_{step:06d}" / "model.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                validate_checkpoint(payload, done[arm], step)
                del payload
                generator.load_state_dict(load_generator_state(str(checkpoint), use_ema=False), strict=True)
                for index in rows[rank::world]:
                    item = dataset[index]
                    trajectory = item["trajectory"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                    noise, clean, target = trajectory[:, 0], trajectory[:, -1], trajectory[:, -2]
                    _, forced = pipeline.inference(noise, [item["prompt"]], return_latents=True,
                                                   return_video=False, context_latents=clean)
                    _, free = pipeline.inference(noise, [item["prompt"]], return_latents=True, return_video=False)
                    for errors in frame_errors(forced, free, target):
                        stream.write(json.dumps({"arm": arm, "step": step, "row_index": index,
                            "prompt": item["prompt"], "noise_seed": item["noise_seed"], **errors}, allow_nan=False) + "\n")
                    stream.flush()
                    if step == steps[-1] and index < video_rows:
                        for history, latents in (("forced", forced), ("free", free)):
                            video = (pipeline.vae.decode_to_pixel(latents, use_cache=False) * .5 + .5).clamp(0, 1)
                            frames = (video[0].permute(0, 2, 3, 1) * 255).cpu()
                            path = output / "videos" / f"{arm}_row{index:04d}_{history}.mp4"
                            write_video_with_fallback(str(path), frames, 16)
                            validate_video(str(path), frames.shape[0], 16)
                    print(f"diagnostic rank={rank} arm={arm} step={step} row={index}", flush=True)
                dist.barrier()
    dist.barrier()
    if rank == 0:
        report(output, training_root, steps, rows)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, help="root containing curved/ and rectified/")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="new directory; never overwrite diagnostics")
    parser.add_argument("--prompt-cache", default="")
    parser.add_argument("--smoke", action="store_true", help="10-step runs: 8 held-out rows, no paper results")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--session", default="iclr_cd_diagnostics")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    command = [args.python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               "--module", "experiments.iclr.cd_diagnostics", "--worker",
               "--training-root", str(Path(args.training_root).resolve()), "--data", str(Path(args.data).resolve()),
               "--output", str(Path(args.output).resolve())]
    if args.prompt_cache:
        command += ["--prompt-cache", str(Path(args.prompt_cache).resolve())]
    if args.smoke:
        command.append("--smoke")
    print(shlex.join(command))
    if not args.execute or args.dry_run:
        print("DRY RUN: paired forced/free history, latent MSE curves and MP4s; no GPU access")
        return
    from experiments.iclr.launch import parse_gpus, require_idle, start_tmux
    require_idle(parse_gpus(args.gpus))
    # env is part of the tmux command, avoiding changes to this session's environment.
    start_tmux(args.session, ["env", f"CUDA_VISIBLE_DEVICES={args.gpus}", *command],
               Path(args.output).parent / "logs" / f"{args.session}.log")


if __name__ == "__main__":
    main()
