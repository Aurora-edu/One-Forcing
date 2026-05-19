from typing import Optional, Tuple

import torch

from model.base import SelfForcingModel
from pipeline import SelfForcingTrainingPipeline


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.inference_pipeline: SelfForcingTrainingPipeline = None

        self.num_train_timestep = args.num_train_timestep
        min_step = getattr(args, "min_step", None)
        max_step = getattr(args, "max_step", None)
        min_step_fraction = float(getattr(args, "min_step_fraction", 0.02))
        max_step_fraction = float(getattr(args, "max_step_fraction", 0.98))
        self.min_step = int(min_step) if min_step is not None else int(min_step_fraction * self.num_train_timestep)
        self.max_step = int(max_step) if max_step is not None else int(max_step_fraction * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.dmd_grad_normalization = getattr(args, "dmd_grad_normalization", "p_real_l1")
        self.dmd_grad_normalization_eps = float(getattr(args, "dmd_grad_normalization_eps", 1e-6))
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _weighted_square_mean(
        self,
        square_error: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gradient_mask is not None:
            return square_error[gradient_mask].mean()
        return square_error.mean()

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
        )
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        grad = pred_fake_image - pred_real_image

        if normalization:
            if self.dmd_grad_normalization == "one_forcing_l1":
                normalizer = torch.abs(grad).mean(dim=[1, 2, 3, 4], keepdim=True)
            elif self.dmd_grad_normalization == "p_real_l1":
                p_real = estimated_clean_image_or_video - pred_real_image
                normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            else:
                raise ValueError(f"Unsupported dmd_grad_normalization: {self.dmd_grad_normalization}")
            normalizer = normalizer.clamp_min(self.dmd_grad_normalization_eps)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        return_pred_grad: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        original_latent = image_or_video
        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True,
            )

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * (timestep / 1000) / (
                    1 + (self.timestep_shift - 1) * (timestep / 1000)
                ) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        loss_dtype = torch.float32 if original_latent.dtype in (torch.float16, torch.bfloat16) else original_latent.dtype
        original_latent_loss = original_latent.to(loss_dtype)
        target_latent = (original_latent_loss - grad.to(loss_dtype)).detach()
        dmd_loss = 0.5 * self._weighted_square_mean(
            (original_latent_loss - target_latent) ** 2,
            gradient_mask=gradient_mask,
        )

        if return_pred_grad:
            pred_grad = original_latent_loss - target_latent
            if gradient_mask is not None:
                masked_pred_grad = torch.zeros_like(pred_grad)
                numel = gradient_mask.sum().clamp_min(1)
                masked_pred_grad[gradient_mask] = pred_grad[gradient_mask] / numel
                pred_grad = masked_pred_grad
            else:
                pred_grad = pred_grad / original_latent_loss.numel()
            return dmd_loss, dmd_log_dict, pred_grad.to(original_latent.dtype)
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
        )
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        generator_log_dict = {
            **dmd_log_dict,
            "dmd_loss": dmd_loss.detach(),
        }
        return dmd_loss, generator_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )

        batch_size, num_frames = generated_image.shape[:2]
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=True,
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * (critic_timestep / 1000) / (
                1 + (self.timestep_shift - 1) * (critic_timestep / 1000)
            ) * 1000
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, generated_image.shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
        )

        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper

            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, generated_image.shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
        )
        return denoising_loss, {
            "critic_timestep": critic_timestep.detach(),
            "fake_score_loss": denoising_loss.detach(),
        }
