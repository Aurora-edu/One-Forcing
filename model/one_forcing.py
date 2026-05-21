import copy
import math
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn.functional as F

from model.dmd import DMD
from utils.wan_wrapper import WanDiffusionWrapper


class OneForcing(DMD):
    """
    One-Forcing video distillation model.

    The open-source training path keeps the core DMD generator loss and the
    adversarial fake-score branch. Historical experimental branches are
    intentionally not part of this public interface.
    """

    def __init__(self, args, device):
        super().__init__(args, device)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)
        self.relativistic_discriminator = getattr(args, "relativistic_discriminator", False)
        self.gan_discriminator_type = getattr(args, "gan_discriminator_type", "attention")
        self.gan_feature_source = getattr(args, "gan_feature_source", "fake_score")
        self.critic_timestep_shift = getattr(args, "critic_timestep_shift", self.timestep_shift)
        self.gan_g_weight = getattr(args, "gan_g_weight", 0.03)
        self.gan_d_weight = getattr(args, "gan_d_weight", 0.03)
        self.r1_weight = getattr(args, "r1_weight", 0.0)
        self.r2_weight = getattr(args, "r2_weight", 0.0)
        self.r1_sigma = getattr(args, "r1_sigma", 0.01)
        self.r2_sigma = getattr(args, "r2_sigma", 0.01)
        self.r1_loss_type = getattr(args, "r1_loss_type", "finite_difference")
        self.discriminator_feature_dim = getattr(args, "discriminator_feature_dim", 1536)
        self.gan_num_class = getattr(args, "gan_num_class", 1)
        self.gan_feature_layers = list(getattr(args, "gan_feature_layers", [13, 21, 29]))
        self.gan_num_registers = getattr(args, "gan_num_registers", len(self.gan_feature_layers))
        self.gan_block_ffn_dim = getattr(args, "gan_block_ffn_dim", 8192)
        self.gan_block_num_heads = getattr(args, "gan_block_num_heads", 12)

        self.use_gan_branch = any(
            weight > 0.0 for weight in (
                self.gan_g_weight,
                self.gan_d_weight,
                self.r1_weight,
                self.r2_weight,
            )
        )
        self.real_score_gradient_checkpointing = bool(
            getattr(
                args,
                "real_score_gradient_checkpointing",
                self.gan_discriminator_type == "conv3d"
                and self.gan_feature_source == "real_score"
                and self.gan_g_weight > 0.0,
            )
        )
        if getattr(args, "gradient_checkpointing", False) and self.real_score_gradient_checkpointing:
            self.real_score.enable_gradient_checkpointing()

        if self.use_gan_branch:
            self.fake_score._gan_feature_layers = self.gan_feature_layers
            self.fake_score._gan_num_registers = self.gan_num_registers
            self.fake_score._gan_block_ffn_dim = self.gan_block_ffn_dim
            self.fake_score._gan_block_num_heads = self.gan_block_num_heads

            if self.gan_discriminator_type == "conv3d":
                if self.gan_feature_source == "real_score":
                    source_model = self.real_score.model
                elif self.gan_feature_source == "fake_score":
                    source_model = self.fake_score.model
                else:
                    raise ValueError(f"Unsupported gan_feature_source: {self.gan_feature_source}")
                source_feature_channels = source_model.dim // math.prod(source_model.patch_size)
                self.fake_score.adding_conv3d_cls_branch(
                    feature_channels=source_feature_channels,
                    hidden_dim=getattr(args, "gan_conv3d_hidden_dim", 256),
                    num_class=self.gan_num_class,
                    pooled_shape=getattr(args, "gan_conv3d_pooled_shape", [8, 16, 16]),
                )
            elif self.gan_discriminator_type == "attention":
                self.fake_score.adding_cls_branch(
                    atten_dim=self.discriminator_feature_dim,
                    num_class=self.gan_num_class,
                    time_embed_dim=self.discriminator_feature_dim if self.concat_time_embeddings else 0,
                )
            else:
                raise ValueError(f"Unsupported gan_discriminator_type: {self.gan_discriminator_type}")

    def set_discriminator_requires_grad(self, enabled: bool) -> None:
        self.fake_score.requires_grad_(enabled)

    def _sample_critic_timestep(
        self,
        batch_size: int,
        num_frames: int,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> torch.Tensor:
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
        if self.critic_timestep_shift > 1:
            critic_timestep = self.critic_timestep_shift * (critic_timestep / 1000) / (
                1 + (self.critic_timestep_shift - 1) * (critic_timestep / 1000)
            ) * 1000
        return critic_timestep.clamp(self.min_step, self.max_step)

    def _prepare_noisy_latent(
        self,
        image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(image_or_video)
        noisy_latent = self.scheduler.add_noise(
            image_or_video.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, image_or_video.shape[:2])
        return noisy_latent, noise

    def _run_cls_pred_branch(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        detach_features: bool = False,
    ) -> torch.Tensor:
        if self.gan_discriminator_type == "conv3d":
            return self._run_conv3d_discriminator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                detach_features=detach_features,
            )

        _, _, noisy_logit = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings,
        )
        return noisy_logit

    def _run_conv3d_discriminator(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        detach_features: bool = False,
    ) -> torch.Tensor:
        if self.gan_feature_source == "real_score":
            feature_model = self.real_score
        elif self.gan_feature_source == "fake_score":
            feature_model = self.fake_score
        else:
            raise ValueError(f"Unsupported gan_feature_source: {self.gan_feature_source}")

        feature_context = torch.no_grad() if detach_features else nullcontext()
        with feature_context:
            features = feature_model(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                features_only=True,
                feature_layers=self.gan_feature_layers,
            )
        if detach_features:
            features = [feature.detach() for feature in features]

        _, _, noisy_logit = self.fake_score(
            discriminator_features=features,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        return noisy_logit

    def _match_real_latent_to_reference(
        self,
        clean_latent: torch.Tensor,
        reference_latent: torch.Tensor,
    ) -> torch.Tensor:
        real_latent = clean_latent.to(
            device=reference_latent.device,
            dtype=reference_latent.dtype,
        )
        if real_latent.shape[1] > reference_latent.shape[1]:
            real_latent = real_latent[:, -reference_latent.shape[1]:]
        if real_latent.shape != reference_latent.shape:
            raise ValueError(
                f"Real latent shape {tuple(real_latent.shape)} does not match "
                f"reference shape {tuple(reference_latent.shape)}"
            )
        return real_latent.detach()

    def _compute_gan_generator_loss(
        self,
        fake_latent: torch.Tensor,
        real_latent: torch.Tensor,
        conditional_dict: dict,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        fake_latent = self._crop_score_window(fake_latent)
        real_latent = self._crop_score_window(real_latent)

        if self.gan_g_weight <= 0.0:
            zero = fake_latent.new_zeros(())
            return zero, {
                "gan_g_loss": zero.detach(),
                "gan_timestep": fake_latent.new_zeros((fake_latent.shape[0], fake_latent.shape[1]), dtype=torch.long),
                "gan_fake_logit": fake_latent.new_zeros((fake_latent.shape[0], 1)),
                "gan_real_logit": fake_latent.new_zeros((fake_latent.shape[0], 1)),
            }

        batch_size, num_frames = fake_latent.shape[:2]
        critic_timestep = self._sample_critic_timestep(
            batch_size=batch_size,
            num_frames=num_frames,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        noisy_fake_latent, _ = self._prepare_noisy_latent(fake_latent, critic_timestep)
        gan_activation_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=False)
            if getattr(self.args, "gan_activation_cpu_offload", True)
            else nullcontext()
        )
        if not self.relativistic_discriminator:
            with gan_activation_context:
                noisy_fake_logit = self._run_cls_pred_branch(
                    noisy_image_or_video=noisy_fake_latent,
                    conditional_dict=conditional_dict,
                    timestep=critic_timestep,
                )
            noisy_real_logit = torch.zeros_like(noisy_fake_logit)
            gan_g_loss = F.softplus(-noisy_fake_logit.float()).mean() * self.gan_g_weight
        else:
            noisy_real_latent, _ = self._prepare_noisy_latent(real_latent, critic_timestep)
            conditional_dict_gan = copy.deepcopy(conditional_dict)
            conditional_dict_gan["prompt_embeds"] = torch.cat(
                (conditional_dict_gan["prompt_embeds"], conditional_dict_gan["prompt_embeds"]), dim=0
            )
            noisy_latent = torch.cat((noisy_fake_latent, noisy_real_latent), dim=0)
            timestep = torch.cat((critic_timestep, critic_timestep), dim=0)
            with gan_activation_context:
                noisy_logit = self._run_cls_pred_branch(
                    noisy_image_or_video=noisy_latent,
                    conditional_dict=conditional_dict_gan,
                    timestep=timestep,
                )
            noisy_fake_logit, noisy_real_logit = noisy_logit.chunk(2, dim=0)
            relative_fake_logit = noisy_fake_logit - noisy_real_logit
            gan_g_loss = F.softplus(-relative_fake_logit.float()).mean() * self.gan_g_weight

        return gan_g_loss, {
            "gan_g_loss": gan_g_loss.detach(),
            "gan_timestep": critic_timestep.detach(),
            "gan_fake_logit": noisy_fake_logit.detach(),
            "gan_real_logit": noisy_real_logit.detach(),
        }

    def _compute_gan_discriminator_loss(
        self,
        fake_latent: torch.Tensor,
        real_latent: torch.Tensor,
        conditional_dict: dict,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        fake_latent = self._crop_score_window(fake_latent)
        real_latent = self._crop_score_window(real_latent)

        if self.gan_d_weight <= 0.0 and self.r1_weight <= 0.0 and self.r2_weight <= 0.0:
            zero = fake_latent.new_zeros(())
            return zero, {
                "gan_d_loss": zero.detach(),
                "r1_loss": zero.detach(),
                "r2_loss": zero.detach(),
                "gan_timestep": fake_latent.new_zeros((fake_latent.shape[0], fake_latent.shape[1]), dtype=torch.long),
                "gan_fake_logit": fake_latent.new_zeros((fake_latent.shape[0], 1)),
                "gan_real_logit": fake_latent.new_zeros((fake_latent.shape[0], 1)),
            }

        batch_size, num_frames = fake_latent.shape[:2]
        critic_timestep = self._sample_critic_timestep(
            batch_size=batch_size,
            num_frames=num_frames,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )

        critic_noise = torch.randn_like(fake_latent)
        noisy_fake_latent, _ = self._prepare_noisy_latent(fake_latent, critic_timestep, noise=critic_noise)
        noisy_real_latent, _ = self._prepare_noisy_latent(real_latent, critic_timestep, noise=critic_noise)

        conditional_dict_gan = copy.deepcopy(conditional_dict)
        conditional_dict_gan["prompt_embeds"] = torch.cat(
            (conditional_dict_gan["prompt_embeds"], conditional_dict_gan["prompt_embeds"]), dim=0
        )
        combined_latent = torch.cat((noisy_fake_latent, noisy_real_latent), dim=0)
        combined_timestep = torch.cat((critic_timestep, critic_timestep), dim=0)
        if self.gan_discriminator_type == "attention":
            _, _, noisy_logit = self.fake_score(
                noisy_image_or_video=combined_latent,
                conditional_dict=conditional_dict_gan,
                timestep=combined_timestep,
                classify_mode=True,
                concat_time_embeddings=self.concat_time_embeddings,
            )
        else:
            noisy_logit = self._run_cls_pred_branch(
                noisy_image_or_video=combined_latent,
                conditional_dict=conditional_dict_gan,
                timestep=combined_timestep,
                detach_features=True,
            )
        noisy_fake_logit, noisy_real_logit = noisy_logit.chunk(2, dim=0)

        if not self.relativistic_discriminator:
            gan_d_loss = F.softplus(-noisy_real_logit.float()).mean() + F.softplus(noisy_fake_logit.float()).mean()
        else:
            relative_real_logit = noisy_real_logit - noisy_fake_logit
            gan_d_loss = F.softplus(-relative_real_logit.float()).mean()
        gan_d_loss = gan_d_loss * self.gan_d_weight

        if self.r1_weight > 0.0:
            noisy_real_latent_perturbed = noisy_real_latent + self.r1_sigma * torch.randn_like(noisy_real_latent)
            noisy_real_logit_perturbed = self._run_cls_pred_branch(
                noisy_image_or_video=noisy_real_latent_perturbed,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                detach_features=True,
            )
            r1_delta = noisy_real_logit_perturbed.float() - noisy_real_logit.float()
            if self.r1_loss_type != "apt":
                r1_delta = r1_delta / self.r1_sigma
            r1_loss = self.r1_weight * torch.mean(r1_delta ** 2)
        else:
            r1_loss = torch.zeros_like(gan_d_loss)

        if self.r2_weight > 0.0:
            noisy_fake_latent_perturbed = noisy_fake_latent + self.r2_sigma * torch.randn_like(noisy_fake_latent)
            noisy_fake_logit_perturbed = self._run_cls_pred_branch(
                noisy_image_or_video=noisy_fake_latent_perturbed,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                detach_features=True,
            )
            r2_grad = (noisy_fake_logit_perturbed - noisy_fake_logit) / self.r2_sigma
            r2_loss = self.r2_weight * torch.mean(r2_grad ** 2)
        else:
            r2_loss = torch.zeros_like(gan_d_loss)

        r1_multiplier = 1.0 if self.r1_loss_type == "apt" else 0.5
        total_loss = gan_d_loss + r1_multiplier * r1_loss + 0.5 * r2_loss
        return total_loss, {
            "gan_d_loss": gan_d_loss.detach(),
            "r1_loss": r1_loss.detach(),
            "r2_loss": r2_loss.detach(),
            "gan_timestep": critic_timestep.detach(),
            "gan_fake_logit": noisy_fake_logit.detach(),
            "gan_real_logit": noisy_real_logit.detach(),
        }

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
        real_latent = self._match_real_latent_to_reference(clean_latent, pred_image)
        gan_g_loss, gan_log_dict = self._compute_gan_generator_loss(
            fake_latent=pred_image,
            real_latent=real_latent,
            conditional_dict=conditional_dict,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        generator_loss = dmd_loss + gan_g_loss
        return generator_loss, {
            **dmd_log_dict,
            **gan_log_dict,
            "dmd_loss": dmd_loss.detach(),
            "generator_loss": generator_loss.detach(),
        }

    def generator_loss_and_backward(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> dict:
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
        )
        dmd_loss, dmd_log_dict, pred_grad = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            return_pred_grad=True,
        )
        real_latent = self._match_real_latent_to_reference(clean_latent, pred_image)
        gan_g_loss, gan_log_dict = self._compute_gan_generator_loss(
            fake_latent=pred_image,
            real_latent=real_latent,
            conditional_dict=conditional_dict,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        dmd_loss_detached = dmd_loss.detach()
        gan_g_loss_detached = gan_g_loss.detach()
        if gan_g_loss.requires_grad:
            gan_grad = torch.autograd.grad(gan_g_loss, pred_image, retain_graph=False)[0]
            pred_grad = pred_grad + gan_grad.to(pred_grad.dtype)
            del gan_grad
        pred_grad = pred_grad.detach()

        del dmd_loss, gan_g_loss, gradient_mask, denoised_timestep_from, denoised_timestep_to
        torch.cuda.empty_cache()
        torch.autograd.backward(pred_image, pred_grad)
        del pred_image, pred_grad
        torch.cuda.empty_cache()

        generator_loss = dmd_loss_detached + gan_g_loss_detached
        return {
            **dmd_log_dict,
            **gan_log_dict,
            "dmd_loss": dmd_loss_detached,
            "generator_loss": generator_loss,
        }

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        include_gan_loss: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )

        generated_image = self._crop_score_window(generated_image)
        batch_size, num_frames = generated_image.shape[:2]
        critic_timestep = self._sample_critic_timestep(
            batch_size=batch_size,
            num_frames=num_frames,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        noisy_generated_image, critic_noise = self._prepare_noisy_latent(generated_image, critic_timestep)

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
        )
        if self.args.denoising_loss_type == "flow":
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
        if include_gan_loss:
            real_latent = self._match_real_latent_to_reference(clean_latent, generated_image)
            gan_d_total_loss, gan_log_dict = self._compute_gan_discriminator_loss(
                fake_latent=generated_image,
                real_latent=real_latent,
                conditional_dict=conditional_dict,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
            )
        else:
            gan_d_total_loss = denoising_loss.new_zeros(())
            gan_log_dict = {}
        critic_loss = denoising_loss + gan_d_total_loss
        return critic_loss, {
            **gan_log_dict,
            "critic_timestep": critic_timestep.detach(),
            "fake_score_loss": denoising_loss.detach(),
            "denoising_loss": denoising_loss.detach(),
            "critic_loss": critic_loss.detach(),
        }

    def discriminator_loss(
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

        generated_image = self._crop_score_window(generated_image)
        real_latent = self._match_real_latent_to_reference(clean_latent, generated_image)
        gan_d_total_loss, gan_log_dict = self._compute_gan_discriminator_loss(
            fake_latent=generated_image,
            real_latent=real_latent,
            conditional_dict=conditional_dict,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        return gan_d_total_loss, {
            **gan_log_dict,
            "discriminator_loss": gan_d_total_loss.detach(),
        }
