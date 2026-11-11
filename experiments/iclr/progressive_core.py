"""Deterministic flow/DDIM step halving; no Wan imports or GPU initialization.

For x_t=(1-t)x0+t*noise, a deterministic step is
x_s=(s/t)x_t+(1-s/t)*D(x_t,t).  Two frozen-teacher steps are
collapsed into the x0 target that makes ONE student step reach the same x_s.
This is progressive distillation, not adjacent-state/EMA consistency training.
"""
import torch

# Actual shifted times / 1000; never warp these a second time.
GRIDS = {4: (1.0, .9375, 5.0 / 6.0, .625, 0.0),
         2: (1.0, 5.0 / 6.0, 0.0), 1: (1.0, 0.0)}
STAGES = {'4to2': (4, 2), '2to1': (2, 1)}


def flow_step(x, x0, t, s):
    if not 0 <= s < t <= 1:
        raise ValueError('Require 0 <= next sigma < current sigma <= 1')
    return (s / t) * x.float() + (1 - s / t) * x0.float()


@torch.no_grad()
def two_step_target(predict, x, t, middle, end):
    """predict(state,sigma) returns x0; no random noise or teacher gradients."""
    if not 0 <= end < middle < t <= 1:
        raise ValueError('Teacher times must strictly decrease')
    mid = flow_step(x, predict(x, t), t, middle)
    destination = flow_step(mid, predict(mid, middle), middle, end)
    target = (destination - (end / t) * x.float()) / (1 - end / t)
    return target.detach(), destination.detach()


def stage_times(stage, update):
    teacher_steps, student_steps = STAGES[stage]
    if update < 1:
        raise ValueError('Updates are one-based')
    # Exactly 150 examples of each coarse interval at 4->2; 300 at 2->1.
    interval = (update - 1) % student_steps
    grid = GRIDS[teacher_steps]
    return interval, grid[2 * interval:2 * interval + 3]


def distillation_loss(student_predict, teacher_predict, clean, noise, stage, update):
    interval, (t, middle, end) = stage_times(stage, update)
    x = ((1 - t) * clean.float() + t * noise.float()).to(clean.dtype)
    target, destination = two_step_target(teacher_predict, x, t, middle, end)
    prediction = student_predict(x, t)
    loss = (prediction.float() - target).square().mean()
    return loss, dict(interval=interval, times=(t, middle, end),
                      target=target, destination=destination, input=x)


@torch.no_grad()
def sample_latents(pipeline, noise, prompts, steps, *, legacy_renoise=False):
    """Framewise AR with generated history, uniform NFE on EVERY latent frame.

    legacy_renoise is only for source-quality comparison with historical all4.
    Training and formal all1 evaluation never enable it.
    """
    if steps not in GRIDS or (legacy_renoise and steps != 4):
        raise ValueError('Only deterministic 4/2/1 or legacy source all4')
    pipeline.generator.eval()
    conditioning = pipeline.text_encoder(text_prompts=prompts)
    pipeline._initialize_kv_cache(noise.shape[0], noise.dtype, noise.device)
    pipeline._initialize_crossattn_cache(noise.shape[0], noise.dtype, noise.device)
    result = torch.empty_like(noise)
    grid = GRIDS[steps]
    for frame in range(noise.shape[1]):
        x = noise[:, frame:frame + 1].float()
        for t, s in zip(grid, grid[1:]):
            time = torch.full((noise.shape[0], 1), t * 1000,
                              device=noise.device, dtype=torch.float32)
            _, x0 = pipeline.generator(noisy_image_or_video=x.to(noise.dtype),
                conditional_dict=conditioning, timestep=time, kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=frame * pipeline.frame_seq_length)
            if legacy_renoise and s > 0:
                x = (1 - s) * x0.float() + s * torch.randn_like(x0).float()
            else:
                x = flow_step(x, x0, t, s)
        result[:, frame:frame + 1] = x.to(noise.dtype)
        # Cache the generated clean frame, never a ground-truth future frame.
        pipeline.generator(noisy_image_or_video=result[:, frame:frame + 1],
            conditional_dict=conditioning, timestep=torch.zeros_like(time),
            kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
            current_start=frame * pipeline.frame_seq_length)
    return result
