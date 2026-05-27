import gc
import logging
import os
import time

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from model import OneForcing
from utils.dataset import CleanLatentLMDBDataset, TextDataset, cycle
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


def _infer_dataset_type(data_path: str) -> str:
    if os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, "data.mdb")):
        return "clean_latent_lmdb"
    if os.path.isfile(data_path):
        return "text"
    raise FileNotFoundError(
        f"Could not infer dataset_type for {data_path}. "
        "Use dataset_type=text for prompt files or dataset_type=clean_latent_lmdb for LMDB latents."
    )


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

        dataset, self.dataset_type = self._build_dataset(
            data_path=config.data_path,
            dataset_type=getattr(config, "dataset_type", "auto"),
            requires_clean_latent=False,
            prompt_extended_path=getattr(config, "extended_prompt_path", None),
            max_pair=int(getattr(config, "max_pair", 1e8)),
            readahead=bool(getattr(config, "lmdb_readahead", False)),
        )
        self.dataloader = self._build_dataloader(dataset)
        self.real_dataloader = None
        self.real_dataset_type = self.dataset_type
        if self.dataset_type == "text" and self._requires_real_latents():
            real_data_path = getattr(config, "real_data_path", "")
            if not real_data_path:
                raise ValueError(
                    "Self-forcing prompt training with One-Forcing GAN enabled requires real_data_path. "
                    "Set real_data_path to a clean-latent LMDB, or set gan_g_weight/gan_d_weight/"
                    "r1_weight/r2_weight to 0 for DMD-only training."
                )
            real_dataset, self.real_dataset_type = self._build_dataset(
                data_path=real_data_path,
                dataset_type=getattr(config, "real_dataset_type", "clean_latent_lmdb"),
                requires_clean_latent=True,
                prompt_extended_path=None,
                max_pair=int(getattr(config, "real_max_pair", getattr(config, "max_pair", 1e8))),
                readahead=bool(getattr(config, "real_lmdb_readahead", getattr(config, "lmdb_readahead", False))),
            )
            self.real_dataloader = self._build_dataloader(real_dataset)
        if self.is_main_process:
            print(f"DATASET TYPE {self.dataset_type} SIZE {len(dataset)}")
            if self.real_dataloader is not None:
                print(f"REAL DATASET TYPE {self.real_dataset_type} SIZE {len(real_dataset)}")

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

    def _build_dataset(
        self,
        *,
        data_path: str,
        dataset_type: str,
        requires_clean_latent: bool,
        prompt_extended_path: str = None,
        max_pair: int,
        readahead: bool,
    ):
        dataset_type = (dataset_type or "auto").lower()
        if dataset_type == "auto":
            dataset_type = _infer_dataset_type(data_path)

        if dataset_type in {"text", "prompt", "prompts"}:
            if requires_clean_latent:
                raise ValueError("A text prompt dataset cannot provide clean latents")
            return TextDataset(data_path, extended_prompt_path=prompt_extended_path), "text"

        if dataset_type in {"clean_latent_lmdb", "latent_lmdb", "lmdb"}:
            return (
                CleanLatentLMDBDataset(
                    data_path,
                    max_pair=max_pair,
                    readahead=readahead,
                ),
                "clean_latent_lmdb",
            )

        raise ValueError(
            f"Unsupported dataset_type={dataset_type}. "
            "Expected auto, text, or clean_latent_lmdb."
        )

    def _build_dataloader(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=getattr(self.config, "dataloader_num_workers", 8),
        )
        return cycle(dataloader)

    def _requires_real_latents(self) -> bool:
        return self.model.use_gan_branch

    def _step_needs_real_batch(self, *, train_generator: bool, train_discriminator: bool) -> bool:
        if train_generator:
            return self.model.relativistic_discriminator and self.model.gan_g_weight > 0.0
        if train_discriminator:
            return (
                self.model.gan_d_weight > 0.0
                or self.model.r1_weight > 0.0
                or self.model.r2_weight > 0.0
            )
        return (
            not self.separate_gan_discriminator_optimizer
            and (
                self.model.gan_d_weight > 0.0
                or self.model.r1_weight > 0.0
                or self.model.r2_weight > 0.0
            )
        )

    def _next_real_batch_if_needed(self, *, train_generator: bool, train_discriminator: bool):
        if self.real_dataloader is None:
            return None
        if not self._step_needs_real_batch(
            train_generator=train_generator,
            train_discriminator=train_discriminator,
        ):
            return None
        return next(self.real_dataloader)

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
        real_batch=None,
    ):
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        clean_latent = batch.get("clean_latent")
        if clean_latent is not None:
            clean_latent = clean_latent.to(device=self.device, dtype=self.dtype)
        image_latent = None
        if self.config.i2v:
            if "ode_latent" in batch:
                image_latent = batch["ode_latent"][:, -1][:, 0:1].to(device=self.device, dtype=self.dtype)
            elif clean_latent is not None:
                image_latent = clean_latent[:, 0:1]
            else:
                raise ValueError("i2v training requires ode_latent or clean_latent in the main batch")

        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = len(text_prompts)
        conditional_dict, unconditional_dict = self._get_conditioning(text_prompts)
        real_latent = clean_latent
        real_conditional_dict = conditional_dict if clean_latent is not None else None
        if real_batch is not None:
            real_latent = real_batch["clean_latent"].to(device=self.device, dtype=self.dtype)
            real_conditional_dict, _ = self._get_conditioning(real_batch["prompts"])

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
                            clean_latent=real_latent,
                            initial_latent=image_latent,
                            real_conditional_dict=real_conditional_dict,
                        )
                else:
                    generator_log_dict = self.model.generator_loss_and_backward(
                        image_or_video_shape=image_or_video_shape,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        clean_latent=real_latent,
                        initial_latent=image_latent,
                        real_conditional_dict=real_conditional_dict,
                    )
            elif getattr(self.config, "generator_activation_cpu_offload", False):
                with torch.autograd.graph.save_on_cpu(pin_memory=False):
                    generator_loss, generator_log_dict = self.model.generator_loss(
                        image_or_video_shape=image_or_video_shape,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        clean_latent=real_latent,
                        initial_latent=image_latent,
                        real_conditional_dict=real_conditional_dict,
                    )
            else:
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=real_latent,
                    initial_latent=image_latent,
                    real_conditional_dict=real_conditional_dict,
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
                clean_latent=real_latent,
                initial_latent=image_latent,
                real_conditional_dict=real_conditional_dict,
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
            clean_latent=real_latent,
            initial_latent=image_latent,
            real_conditional_dict=real_conditional_dict,
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
                real_batch = self._next_real_batch_if_needed(
                    train_generator=True,
                    train_discriminator=False,
                )
                generator_log_dict = self.fwdbwd_one_step(
                    batch,
                    train_generator=True,
                    real_batch=real_batch,
                )
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
                    real_batch = self._next_real_batch_if_needed(
                        train_generator=False,
                        train_discriminator=True,
                    )
                    critic_log_dict = self.fwdbwd_one_step(
                        batch,
                        train_generator=False,
                        train_discriminator=True,
                        real_batch=real_batch,
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
                    real_batch = self._next_real_batch_if_needed(
                        train_generator=False,
                        train_discriminator=False,
                    )
                    critic_log_dict = self.fwdbwd_one_step(
                        batch,
                        train_generator=False,
                        real_batch=real_batch,
                    )
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            else:
                self.critic_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                real_batch = self._next_real_batch_if_needed(
                    train_generator=False,
                    train_discriminator=False,
                )
                critic_log_dict = self.fwdbwd_one_step(
                    batch,
                    train_generator=False,
                    real_batch=real_batch,
                )
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
