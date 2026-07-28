from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
from utils.distributed import launch_distributed_job

import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os
import json
from utils.dataset import LatentLMDBDataset
from scripts.export_videos import load_generator_state

def init_model(device):
    model = WanDiffusionWrapper(is_causal=True).to(device).to(torch.float32).eval()
    model.model.num_frame_per_block = 1 # !!
    encoder = WanTextEncoder().to(device).to(torch.float32).eval()
    

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--rawdata_path", type=str, required=True)
    parser.add_argument("--generator_ckpt", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--require_no_ema",
        action="store_true",
        help="Fail closed if EMA loading is requested.",
    )
    parser.add_argument("--limit", type=int, default=-1)


    args = parser.parse_args()

    if args.require_no_ema and args.use_ema:
        raise ValueError("--require_no_ema cannot be combined with --use_ema")
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be -1 or positive")
    if not os.path.isfile(args.generator_ckpt):
        raise FileNotFoundError(args.generator_ckpt)
    if not os.path.isdir(args.rawdata_path):
        raise FileNotFoundError(args.rawdata_path)
    if os.path.exists(args.output_folder) and os.listdir(args.output_folder):
        raise FileExistsError(
            f"Refusing to mix trajectories with existing output: {args.output_folder}"
        )

    launch_distributed_job()
    global_rank = dist.get_rank()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(device=device)
    state_dict = load_generator_state(args.generator_ckpt, use_ema=args.use_ema)
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict



    dataset = LatentLMDBDataset(args.rawdata_path)
    dataset_size = len(dataset) if args.limit < 0 else min(len(dataset), args.limit)

    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
        with open(
            os.path.join(args.output_folder, "trajectory_generation.intent.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "rawdata_path": os.path.realpath(args.rawdata_path),
                    "generator_ckpt": os.path.realpath(args.generator_ckpt),
                    "guidance_scale": args.guidance_scale,
                    "seed": args.seed,
                    "num_trajectories": dataset_size,
                    "use_ema": bool(args.use_ema),
                    "weight_source": "generator_ema" if args.use_ema else "generator",
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    dist.barrier()
        
    total_steps = int(math.ceil(dataset_size / dist.get_world_size()))
    for index in tqdm(
        range(total_steps), disable=(dist.get_rank() != 0),
    ):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= dataset_size:
            continue
        sample = dataset[prompt_index]
        prompt = sample["prompts"]
       
        clean_latent = sample["clean_latent"].to(device).unsqueeze(0)
        
        

        conditional_dict = encoder(
            text_prompts=[prompt]
        )

        noise_seed = args.seed + prompt_index
        generator = torch.Generator(device=torch.device("cuda", device))
        generator.manual_seed(noise_seed)
        latents = torch.randn(
            [1, 21, 16, 60, 104],
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        
        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps, disable=(dist.get_rank() != 0))):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)
            noisy_input.append(latents)
            f_cond, x0_pred_cond = model(
                latents, conditional_dict, timestep, clean_x = clean_latent
            )

            f_uncond, x0_pred_uncond = model(
                latents, unconditional_dict, timestep, clean_x = clean_latent
            )

            flow_pred = f_uncond + args.guidance_scale * (
                f_cond - f_uncond
            )
            
            
            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)
        noisy_input.append(clean_latent)
        
        noisy_inputs = torch.stack(noisy_input, dim=1)
        selected_indices = [0, 12, 24, 36, len(noisy_input) - 2, len(noisy_input) - 1]
        noisy_inputs = noisy_inputs[:, selected_indices]

        stored_data = noisy_inputs
        selected_timesteps = [
            float(scheduler.timesteps[index].item())
            for index in (0, 12, 24, 36)
        ]
        selected_timesteps.extend([0.0, None])

        torch.save(
            {
                "schema_version": 2,
                "prompt": prompt,
                "trajectory": stored_data.cpu().detach(),
                "noise_seed": noise_seed,
                "selected_timesteps": selected_timesteps,
                "guidance_scale": args.guidance_scale,
                "generator_ckpt": os.path.realpath(args.generator_ckpt),
                "use_ema": args.use_ema,
                "weight_source": "generator_ema" if args.use_ema else "generator",
            },
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
