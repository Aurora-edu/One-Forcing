import argparse
import faulthandler
import os
import signal


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

    args = parser.parse_args()

    from omegaconf import OmegaConf
    import wandb

    from trainer import OneForcingTrainer

    config = OmegaConf.load(args.config_path)
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

    if config.trainer == "one_forcing":
        trainer = OneForcingTrainer(config)
    else:
        raise ValueError(f"Unsupported trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
