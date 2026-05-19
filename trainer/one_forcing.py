import gc
import logging
import os
import time

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from model import OneForcing
from utils.dataset import CleanLatentLMDBDataset, cycle
from utils.distributed import (
    EMA_FSDP,
    fsdp_state_dict,
    fsdp_wrap,
    get_fsdp_wrap_kwargs,
    launch_distributed_job,
)
from utils.misc import set_seed
from utils.prompt_embedding_cache import PromptEmbeddingLMDBCache


def _normalize_state_dict_keys(state_dict):
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    return fixed


def _build_optimizer(optimizer_type, params, lr, betas, weight_decay):
    optimizer_type = (optimizer_type or "adamw").lower()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    if optimizer_type == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            params,
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.max_steps = int(getattr(config, "max_steps", 0))
        self.resume_ema_from_ckpt = bool(getattr(config, "resume_ema_from_ckpt", True))

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )

        self.output_path = config.logdir
        self.model = OneForcing(config, device=self.device)
        self.prompt_embedding_cache = None
        if getattr(config, "prompt_embedding_cache_path", ""):
            self.prompt_embedding_cache = PromptEmbeddingLMDBCache(config.prompt_embedding_cache_path)

        self._resume_loaded_before_fsdp = False
        rank0_preload_resume = bool(getattr(config, "rank0_preload_resume_ckpt", False)) and bool(
            getattr(config, "resume_ckpt", "")
        )
        if rank0_preload_resume:
            self._preload_unwrapped_resume_from_checkpoint(config.resume_ckpt)
            self._resume_loaded_before_fsdp = True
        self._generator_after_resume_loaded_before_fsdp = False
        if rank0_preload_resume and getattr(config, "generator_ckpt_after_resume", ""):
            self._preload_unwrapped_generator_from_checkpoint(config.generator_ckpt_after_resume)
            self._generator_after_resume_loaded_before_fsdp = True

        generator_fsdp_kwargs = get_fsdp_wrap_kwargs(
            config,
            "generator",
            default_transformer_modules=["causal_wan_block"],
        )
        if rank0_preload_resume:
            generator_fsdp_kwargs["sync_module_states"] = True
        self.model.generator = fsdp_wrap(
            self.model.generator,
            **generator_fsdp_kwargs,
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            **get_fsdp_wrap_kwargs(
                config,
                "real_score",
                default_transformer_modules=["wan_block"],
            ),
        )
        fake_score_fsdp_kwargs = get_fsdp_wrap_kwargs(
            config,
            "fake_score",
            default_transformer_modules=["wan_block", "gan_block"],
        )
        if rank0_preload_resume:
            fake_score_fsdp_kwargs["sync_module_states"] = True
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            **fake_score_fsdp_kwargs,
        )
        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                **get_fsdp_wrap_kwargs(
                    config,
                    "text_encoder",
                    default_cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
                ),
            )
            if (
                getattr(config, "text_encoder_fsdp_wrap_strategy", "none") == "none"
                and not getattr(config, "text_encoder_cpu_offload", False)
            ):
                self.model.text_encoder = self.model.text_encoder.to(
                    device=self.device,
                    dtype=self.dtype,
                )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device,
                dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
            )

        self.generator_optimizer = _build_optimizer(
            getattr(config, "generator_optimizer_type", "adamw"),
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.separate_gan_discriminator_optimizer = bool(
            getattr(config, "separate_gan_discriminator_optimizer", False)
        )
        if self.separate_gan_discriminator_optimizer:
            discriminator_params = []
            critic_params = []
            for name, param in self.model.fake_score.named_parameters():
                if not param.requires_grad:
                    continue
                if self._is_gan_discriminator_param(name):
                    discriminator_params.append(param)
                else:
                    critic_params.append(param)
            if not discriminator_params:
                raise ValueError("separate_gan_discriminator_optimizer found no discriminator parameters")
            if not critic_params:
                raise ValueError("separate_gan_discriminator_optimizer found no critic parameters")
            self.discriminator_optimizer = _build_optimizer(
                getattr(
                    config,
                    "discriminator_optimizer_type",
                    getattr(config, "critic_optimizer_type", "adamw"),
                ),
                discriminator_params,
                lr=getattr(config, "lr_discriminator", getattr(config, "lr_critic", config.lr)),
                betas=(
                    getattr(config, "beta1_discriminator", config.beta1_critic),
                    getattr(config, "beta2_discriminator", config.beta2_critic),
                ),
                weight_decay=config.weight_decay,
            )
            critic_optimizer_params = critic_params
        else:
            self.discriminator_optimizer = None
            critic_optimizer_params = [
                param for param in self.model.fake_score.parameters() if param.requires_grad
            ]
        self.critic_optimizer = _build_optimizer(
            getattr(config, "critic_optimizer_type", "adamw"),
            critic_optimizer_params,
            lr=getattr(config, "lr_critic", config.lr),
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay,
        )

        dataset = CleanLatentLMDBDataset(
            config.data_path,
            max_pair=int(getattr(config, "max_pair", 1e8)),
            readahead=bool(getattr(config, "lmdb_readahead", False)),
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "dataloader_num_workers", 8),
        )
        self.dataloader = cycle(dataloader)
        if self.is_main_process:
            print(f"DATASET SIZE {len(dataset)}")

        ema_weight = getattr(config, "ema_weight", 0.0)
        self.generator_ema = None
        self._resume_ema_state = None

        if getattr(config, "generator_ckpt", ""):
            self._load_generator_checkpoint(config.generator_ckpt)
        if getattr(config, "resume_ckpt", "") and not self._resume_loaded_before_fsdp:
            self._resume_from_checkpoint(config.resume_ckpt)
        if (
            getattr(config, "generator_ckpt_after_resume", "")
            and not self._generator_after_resume_loaded_before_fsdp
        ):
            self._load_generator_checkpoint(config.generator_ckpt_after_resume)
        if ema_weight and ema_weight > 0.0 and self.step >= config.ema_start_step:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
            if self._resume_ema_state is not None:
                self.generator_ema.load_state_dict(self._resume_ema_state)

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    @staticmethod
    def _is_gan_discriminator_param(name: str) -> bool:
        return (
            "_cls_pred_branch" in name
            or "_conv3d_cls_branch" in name
            or "_gan_ca_blocks" in name
            or "_register_tokens" in name
        )

    def _set_fake_score_param_groups(self, *, discriminator: bool, critic: bool) -> None:
        for name, param in self.model.fake_score.named_parameters():
            param.requires_grad_(discriminator if self._is_gan_discriminator_param(name) else critic)

    def _load_generator_checkpoint(self, checkpoint_path):
        print(f"Loading pretrained generator from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "generator" in state_dict:
            state_dict = _normalize_state_dict_keys(state_dict["generator"])
        elif "generator_ema" in state_dict:
            state_dict = _normalize_state_dict_keys(state_dict["generator_ema"])
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.generator.load_state_dict(state_dict, strict=True)

    def _preload_unwrapped_generator_from_checkpoint(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        if self.is_main_process:
            print(f"Rank0 preloading generator override from {checkpoint_path}")
            try:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            except TypeError:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "generator" in state_dict:
                state_dict = _normalize_state_dict_keys(state_dict["generator"])
            elif "generator_ema" in state_dict:
                state_dict = _normalize_state_dict_keys(state_dict["generator_ema"])
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=True, assign=True)
            del state_dict
            gc.collect()

    def _preload_unwrapped_resume_from_checkpoint(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        if self.is_main_process:
            print(f"Rank0 preloading One-Forcing weights from {checkpoint_path}")
            try:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            except TypeError:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "generator" in state_dict:
                self.model.generator.load_state_dict(
                    _normalize_state_dict_keys(state_dict["generator"]),
                    strict=True,
                    assign=True,
                )
            elif "generator_ema" in state_dict:
                self.model.generator.load_state_dict(
                    _normalize_state_dict_keys(state_dict["generator_ema"]),
                    strict=True,
                    assign=True,
                )
            if "critic" in state_dict:
                critic_state_dict = _normalize_state_dict_keys(state_dict["critic"])
                critic_current_state = self.model.fake_score.state_dict()
                filtered_critic_state = {}
                skipped_critic_keys = []
                for key, value in critic_state_dict.items():
                    if key not in critic_current_state:
                        skipped_critic_keys.append((key, "missing"))
                        continue
                    if critic_current_state[key].shape != value.shape:
                        skipped_critic_keys.append(
                            (key, f"shape {tuple(value.shape)} -> {tuple(critic_current_state[key].shape)}")
                        )
                        continue
                    filtered_critic_state[key] = value
                self.model.fake_score.load_state_dict(
                    filtered_critic_state,
                    strict=False,
                    assign=True,
                )
                if skipped_critic_keys:
                    preview = ", ".join(f"{key} ({reason})" for key, reason in skipped_critic_keys[:8])
                    suffix = " ..." if len(skipped_critic_keys) > 8 else ""
                    print(
                        f"Skipped {len(skipped_critic_keys)} critic keys when preloading due to architecture mismatch: "
                        f"{preview}{suffix}"
                    )
            self.step = int(state_dict.get("step", self.step))
            del state_dict
            gc.collect()
        step_tensor = torch.tensor([self.step], device=self.device, dtype=torch.long)
        dist.broadcast(step_tensor, src=0)
        self.step = int(step_tensor.item())

    def _resume_from_checkpoint(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        print(f"Resuming One-Forcing weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "generator" in state_dict:
            self.model.generator.load_state_dict(
                _normalize_state_dict_keys(state_dict["generator"]),
                strict=True,
            )
        elif "generator_ema" in state_dict:
            self.model.generator.load_state_dict(
                _normalize_state_dict_keys(state_dict["generator_ema"]),
                strict=True,
            )
        if "critic" in state_dict:
            critic_state_dict = _normalize_state_dict_keys(state_dict["critic"])
            critic_current_state = self.model.fake_score.state_dict()
            filtered_critic_state = {}
            skipped_critic_keys = []
            for key, value in critic_state_dict.items():
                if key not in critic_current_state:
                    skipped_critic_keys.append((key, "missing"))
                    continue
                if critic_current_state[key].shape != value.shape:
                    skipped_critic_keys.append(
                        (key, f"shape {tuple(value.shape)} -> {tuple(critic_current_state[key].shape)}")
                    )
                    continue
                filtered_critic_state[key] = value
            self.model.fake_score.load_state_dict(
                filtered_critic_state,
                strict=False,
            )
            if self.is_main_process and skipped_critic_keys:
                preview = ", ".join(f"{key} ({reason})" for key, reason in skipped_critic_keys[:8])
                suffix = " ..." if len(skipped_critic_keys) > 8 else ""
                print(
                    f"Skipped {len(skipped_critic_keys)} critic keys when resuming due to architecture mismatch: "
                    f"{preview}{suffix}"
                )
        if self.resume_ema_from_ckpt and "generator_ema" in state_dict:
            self._resume_ema_state = {
                key: value.detach().clone().float().cpu()
                for key, value in state_dict["generator_ema"].items()
            }
        self.step = int(state_dict.get("step", self.step))

    def save(self):
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)

        state_dict = {
            "step": self.step,
            "generator": generator_state_dict,
            "critic": critic_state_dict,
        }
        if self.generator_ema is not None and self.step >= self.config.ema_start_step:
            state_dict["generator_ema"] = self.generator_ema.full_state_dict(self.model.generator)

        if self.is_main_process:
            output_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "model.pt")
            torch.save(state_dict, output_path)
            print(f"Model saved to {output_path}")

    def _get_conditioning(self, text_prompts):
        batch_size = len(text_prompts)
        if self.prompt_embedding_cache is not None:
            conditional_dict = {
                "prompt_embeds": self.prompt_embedding_cache.get_batch(
                    text_prompts,
                    device=self.device,
                    dtype=self.dtype,
                )
            }
            unconditional_dict = {
                "prompt_embeds": self.prompt_embedding_cache.get_batch(
                    [self.config.negative_prompt] * batch_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            }
            return conditional_dict, unconditional_dict

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            cache_key = f"unconditional_dict_{batch_size}"
            if not hasattr(self, cache_key):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size,
                )
                unconditional_dict = {key: value.detach() for key, value in unconditional_dict.items()}
                setattr(self, cache_key, unconditional_dict)
            unconditional_dict = getattr(self, cache_key)
        return conditional_dict, unconditional_dict

    def fwdbwd_one_step(
        self,
        batch,
        train_generator,
        train_discriminator: bool = False,
    ):
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1] if self.config.i2v else None

        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = len(text_prompts)
        conditional_dict, unconditional_dict = self._get_conditioning(text_prompts)

        if train_generator:
            if hasattr(self.model, "set_discriminator_requires_grad"):
                self.model.set_discriminator_requires_grad(False)
            manual_generator_backward = bool(getattr(self.config, "manual_generator_backward", False))
            if manual_generator_backward:
                if getattr(self.config, "generator_activation_cpu_offload", False):
                    with torch.autograd.graph.save_on_cpu(pin_memory=False):
                        generator_log_dict = self.model.generator_loss_and_backward(
                            image_or_video_shape=image_or_video_shape,
                            conditional_dict=conditional_dict,
                            unconditional_dict=unconditional_dict,
                            clean_latent=clean_latent,
                            initial_latent=image_latent,
                        )
                else:
                    generator_log_dict = self.model.generator_loss_and_backward(
                        image_or_video_shape=image_or_video_shape,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        clean_latent=clean_latent,
                        initial_latent=image_latent,
                    )
            elif getattr(self.config, "generator_activation_cpu_offload", False):
                with torch.autograd.graph.save_on_cpu(pin_memory=False):
                    generator_loss, generator_log_dict = self.model.generator_loss(
                        image_or_video_shape=image_or_video_shape,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        clean_latent=clean_latent,
                        initial_latent=image_latent,
                    )
            else:
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent,
                )
            if not manual_generator_backward:
                torch.cuda.empty_cache()
                generator_loss.backward()
                generator_log_dict["generator_loss"] = generator_loss.detach()
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
            generator_log_dict["generator_grad_norm"] = generator_grad_norm.detach()
            if hasattr(self.model, "set_discriminator_requires_grad"):
                self.model.set_discriminator_requires_grad(True)
            return generator_log_dict

        if train_discriminator:
            discriminator_loss, discriminator_log_dict = self.model.discriminator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent,
            )
            torch.cuda.empty_cache()
            discriminator_loss.backward()
            discriminator_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
            discriminator_log_dict.update(
                {
                    "discriminator_loss": discriminator_loss.detach(),
                    "discriminator_grad_norm": discriminator_grad_norm.detach(),
                }
            )
            return discriminator_log_dict

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            include_gan_loss=not self.separate_gan_discriminator_optimizer,
        )
        torch.cuda.empty_cache()
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
        critic_log_dict.update(
            {
                "critic_loss": critic_loss.detach(),
                "critic_grad_norm": critic_grad_norm.detach(),
            }
        )
        return critic_log_dict

    def _log_step(self, train_generator, generator_log_dict, critic_log_dict):
        if not self.is_main_process:
            return

        log_dict = {"step": self.step}
        if critic_log_dict:
            if "critic_loss" in critic_log_dict:
                log_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item(),
                    }
                )
            if "discriminator_loss" in critic_log_dict:
                log_dict.update(
                    {
                        "discriminator_loss": critic_log_dict["discriminator_loss"].mean().item(),
                        "discriminator_grad_norm": critic_log_dict["discriminator_grad_norm"].mean().item(),
                    }
                )
            for key in (
                "fake_score_loss",
                "denoising_loss",
                "gan_d_loss",
                "r1_loss",
                "r2_loss",
                "critic_timestep",
                "gan_timestep",
                "gan_fake_logit",
                "gan_real_logit",
            ):
                if key in critic_log_dict:
                    log_dict[key] = critic_log_dict[key].float().mean().item()

        if train_generator and generator_log_dict:
            log_dict.update(
                {
                    "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                    "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                }
            )
            for key in (
                "dmd_loss",
                "gan_g_loss",
                "dmdtrain_gradient_norm",
                "timestep",
                "gan_timestep",
                "gan_fake_logit",
                "gan_real_logit",
            ):
                if key in generator_log_dict:
                    log_dict[key] = generator_log_dict[key].float().mean().item()

        if not self.disable_wandb:
            wandb.log(log_dict, step=self.step)
        else:
            print(log_dict, flush=True)

    def train(self):
        while self.max_steps <= 0 or self.step < self.max_steps:
            if self.separate_gan_discriminator_optimizer:
                update_slot = self.step % self.config.dfake_gen_update_ratio
                train_generator = update_slot == 0
                train_discriminator = update_slot == getattr(
                    self.config,
                    "discriminator_gen_update_ratio",
                    1,
                )
            else:
                train_generator = self.step % self.config.dfake_gen_update_ratio == 0
                train_discriminator = False

            generator_log_dict = {}
            critic_log_dict = {}

            if train_generator:
                self.model.generator.requires_grad_(True)
                self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                if self.discriminator_optimizer is not None:
                    self.discriminator_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, train_generator=True)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                self.generator_optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            if self.separate_gan_discriminator_optimizer:
                if train_discriminator:
                    self.model.generator.requires_grad_(False)
                    self.model.fake_score.requires_grad_(False)
                    self._set_fake_score_param_groups(discriminator=True, critic=False)
                    self.discriminator_optimizer.zero_grad(set_to_none=True)
                    batch = next(self.dataloader)
                    critic_log_dict = self.fwdbwd_one_step(
                        batch,
                        train_generator=False,
                        train_discriminator=True,
                    )
                    self.discriminator_optimizer.step()
                    self.discriminator_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                elif not train_generator:
                    self.model.generator.requires_grad_(False)
                    self.model.fake_score.requires_grad_(False)
                    self._set_fake_score_param_groups(discriminator=False, critic=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    batch = next(self.dataloader)
                    critic_log_dict = self.fwdbwd_one_step(batch, train_generator=False)
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            else:
                self.critic_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                critic_log_dict = self.fwdbwd_one_step(batch, train_generator=False)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            self.step += 1

            if (
                self.step >= self.config.ema_start_step
                and self.generator_ema is None
                and self.config.ema_weight > 0
            ):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            if not self.config.no_save and self.step > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            self._log_step(train_generator, generator_log_dict, critic_log_dict)

            if self.step % self.config.gc_interval == 0:
                if self.is_main_process:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    iteration_time = current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    else:
                        print(
                            {"step": self.step, "per_iteration_time": iteration_time},
                            flush=True,
                        )
                    self.previous_time = current_time

        if not self.config.no_save:
            self.save()
