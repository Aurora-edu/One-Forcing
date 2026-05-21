from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long, device=self.device)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.teacher_model_path = getattr(args, "teacher_model_path", "")
        self.iscausal = getattr(args, "causal", True)

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=self.iscausal)
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(
            model_name=self.real_model_name,
            model_path=self.teacher_model_path or None,
            is_causal=False,
        )
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        if getattr(args, "prompt_embedding_cache_path", ""):
            self.text_encoder = None
        else:
            self.text_encoder = WanTextEncoder()
            self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        if uniform_timestep:
            return torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long,
            ).repeat(1, num_frame)

        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size, num_frame],
            device=self.device,
            dtype=torch.long,
        )
        if self.independent_first_frame:
            timestep_from_second = timestep[:, 1:]
            timestep_from_second = timestep_from_second.reshape(timestep_from_second.shape[0], -1, num_frame_per_block)
            timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
            timestep_from_second = timestep_from_second.reshape(timestep_from_second.shape[0], -1)
            timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
        else:
            timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep

    def _resolve_gradient_num_frames(self, total_num_frames: int) -> int:
        active_gradient_num_frames = getattr(self, "_active_gradient_num_frames", None)
        if active_gradient_num_frames is not None:
            return max(1, min(total_num_frames, int(active_gradient_num_frames)))

        gradient_num_blocks = int(getattr(self.args, "gradient_num_blocks", 0) or 0)
        gradient_num_frames = int(getattr(self.args, "gradient_num_frames", 0) or 0)
        if gradient_num_blocks > 0:
            gradient_num_frames = gradient_num_blocks * getattr(self, "num_frame_per_block", 1)

        if gradient_num_frames <= 0:
            return total_num_frames

        if not getattr(self.args, "independent_first_frame", False):
            block_size = max(1, getattr(self, "num_frame_per_block", 1))
            gradient_num_frames = ((gradient_num_frames + block_size - 1) // block_size) * block_size

        return max(1, min(total_num_frames, gradient_num_frames))

    def _get_gradient_window_position(self) -> str:
        return getattr(
            self,
            "_active_gradient_window_position",
            getattr(self.args, "gradient_window_position", "tail"),
        )

    @contextmanager
    def training_window(
        self,
        *,
        gradient_window_position: Optional[str] = None,
        gradient_num_frames: Optional[int] = None,
        score_window_position: Optional[str] = None,
        score_num_frames: Optional[int] = None,
    ):
        old_gradient_window_position = getattr(self, "_active_gradient_window_position", None)
        old_gradient_num_frames = getattr(self, "_active_gradient_num_frames", None)
        old_score_window_position = getattr(self, "_active_score_window_position", None)
        old_score_num_frames = getattr(self, "_active_score_num_frames", None)

        if gradient_window_position is not None:
            self._active_gradient_window_position = gradient_window_position
        if gradient_num_frames is not None:
            self._active_gradient_num_frames = int(gradient_num_frames)
        if score_window_position is not None:
            self._active_score_window_position = score_window_position
        if score_num_frames is not None:
            self._active_score_num_frames = int(score_num_frames)

        try:
            yield
        finally:
            self._restore_active_window_attr("_active_gradient_window_position", old_gradient_window_position)
            self._restore_active_window_attr("_active_gradient_num_frames", old_gradient_num_frames)
            self._restore_active_window_attr("_active_score_window_position", old_score_window_position)
            self._restore_active_window_attr("_active_score_num_frames", old_score_num_frames)

    def _restore_active_window_attr(self, name: str, value):
        if value is None:
            if hasattr(self, name):
                delattr(self, name)
        else:
            setattr(self, name, value)

    def _build_gradient_mask(
        self,
        pred_image_or_video: torch.Tensor,
        num_generated_frames: int,
        min_num_frames: int,
    ) -> Optional[torch.Tensor]:
        gradient_mask = None

        if num_generated_frames != min_num_frames:
            gradient_mask = torch.ones_like(pred_image_or_video, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False

        gradient_num_frames = self._resolve_gradient_num_frames(pred_image_or_video.shape[1])
        if gradient_num_frames < pred_image_or_video.shape[1]:
            if gradient_mask is None:
                gradient_mask = torch.ones_like(pred_image_or_video, dtype=torch.bool)
            window_position = self._get_gradient_window_position()
            if window_position == "first":
                gradient_mask[:, gradient_num_frames:] = False
            elif window_position == "tail":
                gradient_mask[:, :pred_image_or_video.shape[1] - gradient_num_frames] = False
            else:
                raise ValueError(f"Unsupported gradient_window_position: {window_position}")

        return gradient_mask


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        self.inference_pipeline = None

    def _slice_generated_output(self, pred_image_or_video: torch.Tensor) -> torch.Tensor:
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        return pred_image_or_video.to(self.dtype)

    def generate_from_noise(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        pred_image_or_video, _, _ = self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            clean_image_or_video=None,
            initial_latent=initial_latent,
            **conditional_dict,
        )
        return self._slice_generated_output(pred_image_or_video)

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent=None,
        initial_latent: torch.Tensor = None,
        denoise_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_frames = num_generated_blocks.item() * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        noise_shape[1] = num_generated_frames

        clean_image_or_video = None
        if clean_latent is not None:
            clean_image_or_video = clean_latent.to(self.device, dtype=self.dtype)
            assert clean_image_or_video.shape == tuple(noise_shape), f"{clean_image_or_video.shape} != {tuple(noise_shape)}"

        simulation_kwargs = dict(
            noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
            clean_image_or_video=clean_image_or_video,
            denoise_steps=denoise_steps,
            **conditional_dict,
        )
        if getattr(self.args, "generator_activation_cpu_offload", False):
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
                    self._consistency_backward_simulation(**simulation_kwargs)
                )
        else:
            pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
                self._consistency_backward_simulation(**simulation_kwargs)
            )

        pred_image_or_video_last_21 = self._slice_generated_output(pred_image_or_video)
        gradient_mask = self._build_gradient_mask(
            pred_image_or_video=pred_image_or_video_last_21,
            num_generated_frames=num_generated_frames,
            min_num_frames=min_num_frames,
        )
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor,
        denoise_steps: Optional[int] = None,
        **conditional_dict: dict,
    ):
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        self.inference_pipeline.gradient_num_frames = self._resolve_gradient_num_frames(self.num_training_frames)
        self.inference_pipeline.gradient_window_position = self._get_gradient_window_position()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            clean_image_or_video=clean_image_or_video,
            denoise_steps=denoise_steps,
            **conditional_dict,
        )

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            gradient_num_frames=self._resolve_gradient_num_frames(self.num_training_frames),
            gradient_window_position=self._get_gradient_window_position(),
        )
