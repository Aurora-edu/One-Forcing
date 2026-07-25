import argparse
import faulthandler
import json
import os
import shlex
import signal
import subprocess
import sys


def main():
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--generator_ckpt", type=str, default="", help="Override generator checkpoint in config")
    parser.add_argument("--teacher_model_path", type=str, default="", help="Override teacher model path in config")
    parser.add_argument("--data_path", type=str, default="", help="Override data path in config")
    parser.add_argument(
        "--prompt_embedding_cache_path",
        type=str,
        default="",
        help="Override prompt-embedding LMDB path in config",
    )
    parser.add_argument("--dataset_type", type=str, default="", help="Override dataset type in config")
    parser.add_argument("--extended_prompt_path", type=str, default="", help="Override extended prompt path in config")
    parser.add_argument("--real_data_path", type=str, default="", help="Override real latent data path in config")
    parser.add_argument("--real_dataset_type", type=str, default="", help="Override real dataset type in config")
    parser.add_argument("--resume_ckpt", type=str, default="", help="Override resume checkpoint in config")
    parser.add_argument("--seed", type=int, default=None, help="Fixed base seed; rank is added per process")
    parser.add_argument("--max_steps", type=int, default=None, help="Override the total optimizer iterations")
    parser.add_argument("--log_iters", type=int, default=None, help="Override checkpoint interval")
    parser.add_argument(
        "--randomize_seed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw and broadcast a random base seed instead of using --seed/config.seed",
    )
    parser.add_argument(
        "--text_encoder_cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override FSDP CPU offload for the frozen text encoder (hardware-only setting)",
    )
    parser.add_argument(
        "--fake_score_cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override FSDP CPU offload for the fake-score/critic model (hardware-only setting)",
    )
    parser.add_argument(
        "--real_score_cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override FSDP CPU offload for the frozen real-score teacher (hardware-only setting)",
    )
    parser.add_argument(
        "--manual_generator_backward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the mathematically equivalent low-peak generator backward path",
    )
    parser.add_argument(
        "--generator_optimizer_state_cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep AdamW generator optimizer state on CPU between optimizer steps",
    )
    parser.add_argument(
        "--rank0_preload_generator_ckpt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load the ODE generator checkpoint only on rank 0 before FSDP broadcasts it",
    )

    args = parser.parse_args()

    from omegaconf import OmegaConf
    import wandb

    from trainer import OneForcingTrainer
    from utils.config import load_config

    config = load_config(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    if args.generator_ckpt:
        config.generator_ckpt = args.generator_ckpt
    if args.teacher_model_path:
        config.teacher_model_path = args.teacher_model_path
    if args.data_path:
        config.data_path = args.data_path
    if args.prompt_embedding_cache_path:
        config.prompt_embedding_cache_path = args.prompt_embedding_cache_path
    if args.dataset_type:
        config.dataset_type = args.dataset_type
    if args.extended_prompt_path:
        config.extended_prompt_path = args.extended_prompt_path
    if args.real_data_path:
        config.real_data_path = args.real_data_path
    if args.real_dataset_type:
        config.real_dataset_type = args.real_dataset_type
    if args.resume_ckpt:
        config.resume_ckpt = args.resume_ckpt
    if args.seed is not None:
        config.seed = args.seed
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max_steps must be positive")
        config.max_steps = args.max_steps
    if args.log_iters is not None:
        if args.log_iters <= 0:
            raise ValueError("--log_iters must be positive")
        config.log_iters = args.log_iters
    if args.randomize_seed is not None:
        config.randomize_seed = args.randomize_seed
    elif not hasattr(config, "randomize_seed"):
        config.randomize_seed = False
    if args.text_encoder_cpu_offload is not None:
        config.text_encoder_cpu_offload = args.text_encoder_cpu_offload
    if args.fake_score_cpu_offload is not None:
        config.fake_score_cpu_offload = args.fake_score_cpu_offload
    if args.real_score_cpu_offload is not None:
        config.real_score_cpu_offload = args.real_score_cpu_offload
    if args.manual_generator_backward is not None:
        config.manual_generator_backward = args.manual_generator_backward
    if args.generator_optimizer_state_cpu_offload is not None:
        config.generator_optimizer_state_cpu_offload = (
            args.generator_optimizer_state_cpu_offload
        )
    if args.rank0_preload_generator_ckpt is not None:
        config.rank0_preload_generator_ckpt = args.rank0_preload_generator_ckpt

    if int(os.environ.get("RANK", "0")) == 0 and config.logdir:
        os.makedirs(config.logdir, exist_ok=True)
        OmegaConf.save(config, os.path.join(config.logdir, "resolved_config.yaml"))
        with open(os.path.join(config.logdir, "launch_command.txt"), "w", encoding="utf-8") as fp:
            fp.write(shlex.join([sys.executable, *sys.argv]) + "\n")
        metadata = {
            "config_path": os.path.abspath(args.config_path),
            "cwd": os.getcwd(),
            "python": sys.version,
        }
        try:
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.getcwd(),
                text=True,
            ).strip()
            metadata["git_status_porcelain"] = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=os.getcwd(),
                text=True,
            ).splitlines()
        except (OSError, subprocess.CalledProcessError):
            metadata["git_commit"] = None
            metadata["git_status_porcelain"] = None
        with open(os.path.join(config.logdir, "run_metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)
            fp.write("\n")

    if config.trainer == "one_forcing":
        trainer = OneForcingTrainer(config)
    else:
        raise ValueError(f"Unsupported trainer: {config.trainer}")
    trainer.train()

    wandb.finish()
    if int(os.environ.get("RANK", "0")) == 0 and config.logdir:
        completion = {
            "final_step": int(trainer.step),
            "max_steps": int(config.max_steps),
        }
        with open(os.path.join(config.logdir, "training.done"), "w", encoding="utf-8") as fp:
            json.dump(completion, fp, sort_keys=True)
            fp.write("\n")


if __name__ == "__main__":
    main()
