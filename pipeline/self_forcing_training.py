from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 first_frame_denoising_step_list: Optional[List[int]] = None,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 gradient_num_frames: Optional[int] = None,
                 gradient_window_position: str = "tail",
                 rollout_schedule: str = "fixed",
                 first_rollout_num_frames: int = 4,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = self._trim_zero_timestep(denoising_step_list)
        self.first_frame_denoising_step_list = self._trim_zero_timestep(
            first_frame_denoising_step_list
        )

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.gradient_num_frames = gradient_num_frames
        self.gradient_window_position = gradient_window_position
        self.rollout_schedule = rollout_schedule.lower()
        self.first_rollout_num_frames = int(first_rollout_num_frames)
        if self.first_rollout_num_frames <= 0:
            raise ValueError("first_rollout_num_frames must be positive")

    @staticmethod
    def _trim_zero_timestep(denoising_step_list):
        if denoising_step_list is None:
            return None
        if len(denoising_step_list) == 0:
            raise ValueError("denoising_step_list must not be empty")
        if int(denoising_step_list[-1]) == 0:
            return denoising_step_list[:-1]
        return denoising_step_list

    def _build_rollout_frame_counts(
        self,
        num_frames: int,
        initial_latent: Optional[torch.Tensor],
    ) -> List[int]:
        if self.rollout_schedule == "fixed":
            if not self.independent_first_frame or (
                self.independent_first_frame and initial_latent is not None
            ):
                assert num_frames % self.num_frame_per_block == 0
                return [self.num_frame_per_block] * (num_frames // self.num_frame_per_block)

            assert (num_frames - 1) % self.num_frame_per_block == 0
            return [1] + [self.num_frame_per_block] * ((num_frames - 1) // self.num_frame_per_block)

        if self.rollout_schedule != "first4then1":
            raise ValueError(f"Unsupported rollout_schedule: {self.rollout_schedule}")
        if self.independent_first_frame:
            raise NotImplementedError("first4then1 rollout does not support independent_first_frame")
        if initial_latent is not None:
            raise NotImplementedError("first4then1 rollout does not support initial_latent")
        if self.first_frame_denoising_step_list is None:
            raise ValueError("first4then1 rollout requires first_frame_denoising_step_list")
        if num_frames < self.first_rollout_num_frames:
            raise ValueError(
                f"first4then1 rollout needs at least {self.first_rollout_num_frames} frames, "
                f"got {num_frames}"
            )
        return [self.first_rollout_num_frames] + [1] * (num_frames - self.first_rollout_num_frames)

    def _block_denoising_step_list(self, block_index: int):
        if self.rollout_schedule == "first4then1" and block_index == 0:
            return self.first_frame_denoising_step_list
        return self.denoising_step_list

    @staticmethod
    def _same_denoising_step_list(left, right) -> bool:
        if torch.is_tensor(left) and torch.is_tensor(right):
            return torch.equal(left, right)
        if len(left) != len(right):
            return False
        return all(int(left_step) == int(right_step) for left_step, right_step in zip(left, right))

    def _block_exit_index(
        self,
        block_index: int,
        block_denoising_step_list,
        exit_flags: List[int],
        denoise_steps: Optional[int],
    ) -> int:
        if self.rollout_schedule == "first4then1" and block_index == 0:
            return len(block_denoising_step_list) - 1
        if denoise_steps is not None:
            denoise_steps = int(denoise_steps)
            if denoise_steps < 0 or denoise_steps >= len(block_denoising_step_list):
                raise ValueError(
                    f"denoise_steps={denoise_steps} is outside [0, {len(block_denoising_step_list)})"
                )
            return denoise_steps
        if self.same_step_across_blocks:
            return exit_flags[0]
        return exit_flags[block_index]

    def _timestep_to_train_index(self, timestep) -> int:
        timestep = timestep.to(self.scheduler.timesteps.device)
        return 1000 - torch.argmin((self.scheduler.timesteps - timestep).abs(), dim=0).item()

    def _denoised_timestep_bounds(self, denoising_step_list, exit_index: int):
        if exit_index == len(denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = self._timestep_to_train_index(denoising_step_list[exit_index])
        else:
            denoised_timestep_to = self._timestep_to_train_index(denoising_step_list[exit_index + 1])
            denoised_timestep_from = self._timestep_to_train_index(denoising_step_list[exit_index])
        return denoised_timestep_from, denoised_timestep_to

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            # In our training, self.last_step_only is False
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            clean_image_or_video: torch.Tensor = None, # same shape as noise
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            denoise_steps: Optional[int] = None,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        all_num_frames = self._build_rollout_frame_counts(num_frames, initial_latent)
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Release the previous step's caches before allocating new buffers.
        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        torch.cuda.empty_cache()

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None: # Never met
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        gradient_num_frames = (
            num_output_frames
            if self.gradient_num_frames is None or self.gradient_num_frames <= 0
            else min(self.gradient_num_frames, num_output_frames)
        )
        if self.gradient_window_position == "first":
            start_gradient_frame_index = 0
            end_gradient_frame_index = gradient_num_frames
        elif self.gradient_window_position == "tail":
            start_gradient_frame_index = max(0, num_output_frames - gradient_num_frames)
            end_gradient_frame_index = num_output_frames
        else:
            raise ValueError(f"Unsupported gradient_window_position: {self.gradient_window_position}")

        # for block_index in range(num_blocks):
        range_denoising_step_list = None
        range_exit_index = None
        for block_index, current_num_frames in enumerate(all_num_frames):
            current_denoising_step_list = self._block_denoising_step_list(block_index)
            current_exit_index = self._block_exit_index(
                block_index=block_index,
                block_denoising_step_list=current_denoising_step_list,
                exit_flags=exit_flags,
                denoise_steps=denoise_steps,
            )
            
            if True:
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # Step 3.1: Spatial denoising loop
                # Such a loop corresponds to the truncated denoising algorithm:
                #    T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
                # For many-step model, we certainly cannot use this method, but for 4-step DMD, 
                # we can inherit it for a fair comaprison. Note that as long as the conditions 
                # are clean GT rather than self-generated frames, we can perform TF. So this 
                # method does not conflict with TF in the frame- dimension.
                for index, current_timestep in enumerate(current_denoising_step_list):
                    exit_flag = index == current_exit_index
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                            next_timestep = current_denoising_step_list[index + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        enable_grad = (
                            current_start_frame < end_gradient_frame_index
                            and current_start_frame + current_num_frames > start_gradient_frame_index
                        )
                        if enable_grad:
                            if range_denoising_step_list is None:
                                range_denoising_step_list = current_denoising_step_list
                                range_exit_index = current_exit_index
                            elif (
                                current_exit_index != range_exit_index
                                or len(current_denoising_step_list) != len(range_denoising_step_list)
                                or not self._same_denoising_step_list(
                                    current_denoising_step_list,
                                    range_denoising_step_list,
                                )
                            ):
                                raise ValueError(
                                    "Gradient window spans blocks with different denoising schedules. "
                                    "Adjust gradient_num_frames/gradient_window_position so the optimized "
                                    "frames use one rollout schedule."
                                )
                        if not enable_grad:
                            with torch.no_grad():
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length
                                )
                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        break
                    
            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks: # Useless, never met
            denoised_timestep_from, denoised_timestep_to = None, None
        # T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
        # denoised_timestep_from = \tau
        # denoised_timestep_to = next timestep smaller than \tau
        # These are just engineering tricks
        # to align DMD timestep sampling with the actual denoising range used by the generator
        else:
            if range_denoising_step_list is None:
                range_denoising_step_list = self.denoising_step_list
                range_exit_index = exit_flags[0]
            denoised_timestep_from, denoised_timestep_to = self._denoised_timestep_bounds(
                range_denoising_step_list,
                range_exit_index,
            )

        if return_sim_step: # False
            sim_step = range_exit_index + 1 if range_exit_index is not None else exit_flags[0] + 1
            return output, denoised_timestep_from, denoised_timestep_to, sim_step

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
