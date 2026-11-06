"""Collect full bidirectional Wan 1.3B teacher paths; default is CPU dry-run.

Each Euler state is copied to a CPU memory-mapped buffer immediately. The
published schema-2 tensor contains every real scheduler time and the final t=0
ODE endpoint, with no dataset-clean latent and no clean-video conditioning.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.iclr.protocol import REPO, atomic_json, sha256_file, tensor_state_sha256
from utils.config import load_config
from utils.scheduler import FlowMatchScheduler


def make_schedule(steps, shift=8.0):
    if steps < 1 or not np.isfinite(shift) or shift <= 0:
        raise ValueError("invalid teacher solver schedule")
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0., extra_one_step=True)
    scheduler.set_timesteps(steps)
    times = [float(t) for t in scheduler.timesteps] + [0.0]
    if times[0] != 1000. or not all(a > b for a, b in zip(times, times[1:])):
        raise ValueError("collector requires full descending physical time, including t=0")
    return scheduler, times


def euler_update(scheduler, latent, flow, timestep):
    """Float32 solver update on every frame, including the final step to zero."""
    return scheduler.step(flow.float().flatten(0, 1), timestep.flatten(0, 1),
                          latent.float().flatten(0, 1)).unflatten(0, latent.shape[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="fixed original prompt txt, same order at both resolutions")
    parser.add_argument("--model-path", required=True, help="Wan2.1-T2V-1.3B pretrained directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, choices=(50, 200), default=50)
    parser.add_argument("--shift", type=float, default=8.)
    parser.add_argument("--cfg", type=float, default=6.)
    parser.add_argument("--negative-prompt", default=str(load_config(str(REPO / "config.yaml")).negative_prompt))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu", default="0", help="physical GPU id; independently shard prompt indices")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scheduler, times = make_schedule(args.steps, args.shift)
    if args.seed < 0 or args.limit < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid seed, prompt limit or shard")
    if not args.gpu.isdigit() or not np.isfinite(args.cfg):
        raise ValueError("invalid GPU id or CFG")
    intent = {"trajectory_kind": "bidirectional_teacher_full", "is_causal": False,
        "model_name": "Wan2.1-T2V-1.3B", "model_path": str(Path(args.model_path).resolve()),
        "num_solver_steps": args.steps, "shift": args.shift, "guidance_scale": args.cfg,
        "selected_timesteps": times, "seed": args.seed, "limit": args.limit,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "clean_conditioning": False, "use_ema": False, "weight_source": "pretrained_teacher",
        "negative_prompt": args.negative_prompt, "latent_shape": [21,16,60,104],
        "solver": "Euler / repository FlowMatchScheduler; original paper solver provenance unverified"}
    print(json.dumps(intent, indent=2))
    if not args.execute:
        print("DRY RUN: full teacher path; no model, prompt or GPU access")
        return
    from experiments.iclr.launch import require_idle
    require_idle([args.gpu])
    # Configure visibility before the first CUDA call, without touching other GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    prompts = Path(args.prompts).read_text().splitlines()[:args.limit]
    if len(prompts) != args.limit or any(not p.strip() for p in prompts):
        raise ValueError("prompt file must contain the requested number of nonempty prompts")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    intent["prompt_sha256"] = sha256_file(args.prompts)
    # Shards may share an output root; each owns only its explicitly named files.
    intent_path = output / f"collection.shard_{args.shard_index:03d}.json"
    if intent_path.exists():
        raise FileExistsError("collection shard already exists; use a fresh output directory")
    atomic_json(intent_path, intent)
    teacher = WanDiffusionWrapper(model_name="Wan2.1-T2V-1.3B", model_path=args.model_path,
                                  is_causal=False, timestep_shift=args.shift).eval().requires_grad_(False)
    intent["teacher_tensor_sha256"] = tensor_state_sha256(teacher.state_dict())
    atomic_json(intent_path, intent)
    teacher = teacher.to(device=device, dtype=torch.bfloat16)
    teacher.scheduler = scheduler
    encoder = WanTextEncoder().eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        unconditional = encoder(text_prompts=[args.negative_prompt])
        for index in range(args.shard_index, len(prompts), args.num_shards):
            final = output / f"{index:05d}.pt"
            temporary = output / f".{index:05d}.{os.getpid()}.pt.tmp"
            backing = output / f".{index:05d}.{os.getpid()}.states.tmp"
            if final.exists():
                raise FileExistsError(final)
            seed = args.seed + index
            generator = torch.Generator(device=device).manual_seed(seed)
            latent = torch.randn((1,21,16,60,104), device=device, dtype=torch.float32, generator=generator)
            states = np.memmap(backing, mode="w+", dtype=np.float32, shape=(len(times),21,16,60,104))
            states[0] = latent[0].cpu().numpy()
            states.flush()
            conditional = encoder(text_prompts=[prompts[index]])
            for step, t in enumerate(scheduler.timesteps.clone()):
                timestep = torch.full((1,21), float(t), device=device, dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    conditioned, _ = teacher(latent, conditional, timestep)
                    unconditioned, _ = teacher(latent, unconditional, timestep)
                flow = unconditioned.float() + args.cfg * (conditioned.float() - unconditioned.float())
                latent = euler_update(scheduler, latent, flow, timestep)
                if not bool(torch.isfinite(latent).all()):
                    raise FloatingPointError(f"nonfinite teacher state: prompt {index}, step {step}")
                states[step+1] = latent[0].cpu().numpy()
                states.flush()
            payload = {**intent, "schema_version": 2, "trajectory": torch.from_numpy(states),
                       "prompt": prompts[index], "prompt_index": index, "noise_seed": seed}
            torch.save(payload, temporary)
            os.replace(temporary, final)
            del payload, states
            backing.unlink()  # Only our own temporary backing file, after successful publication.
            print(f"teacher trajectory {index}: {len(times)} states, seed={seed}, {final}", flush=True)


if __name__ == "__main__":
    main()
