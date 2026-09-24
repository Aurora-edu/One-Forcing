# Official GitHub-main-code DMD+GAN versus DMD-only

This experiment trains **both** arms with the unmodified official
`Aurora-edu/One-Forcing` training implementation at
`main@c9a2350e8740531562011fc9618e1a928d911ae0`. The DMD+GAN arm uses
the official `config.yaml` values and official README training entry point.
The DMD-only arm changes **only** `gan_g_weight` and `gan_d_weight` from
`0.03` to `0.0`. No FFE rollout is used during training; both checkpoints
are evaluated with identical FFE inference afterward.

Official main interprets `seed: 0` as a new random runtime seed in each run.
For a paired ablation, `official_main_pair.py prepare` draws **one nonzero**
seed from the official runtime-seed range and writes it into both
configurations (zero is excluded because main treats it as a redraw sentinel). Thus the
effective seed is shared and recorded; this is the sole controlled deviation
from the literal main YAML for the full arm. The training code, one-step
rollout, optimizer, model, 200-step budget, and README CLI flags are otherwise
official main. The published `one_forcing.pt` is an independent reference,
not a substitute for this newly trained DMD+GAN arm.

The earlier `fc3b475` FFE-trained pair, the `a5cdc8e` pair that uses rebuttal
branch training code, and the old 80.58/77.21 checkpoints are **not** inputs
to this experiment. Do not interrupt their running processes or overwrite
their output directories.

## Training-machine instructions

Keep weights and videos local. Commit only the final audited metric report
and genuine visual observations after review. The previously specified
48-hour limit applies to the entire new pair and evaluation; if sufficient
idle GPUs or time are unavailable, report that rather than changing the
batch size, training steps, prompt count, or sample count.

1. Make a **separate** HTTPS clone for this experiment. From its rebuttal
   checkout, add an official-main worktree at the pinned commit. Do not pull
   inside a checkout used by another live training session.

   ```bash
   git clone --branch iclr2027-rebuttal-gan-ablation-audit \
     https://github.com/Aurora-edu/One-Forcing.git One-Forcing-official-pair
   cd One-Forcing-official-pair
   git fetch https://github.com/Aurora-edu/One-Forcing.git main

   export REBUTTAL_ROOT="$PWD"
   export OFFICIAL_ROOT=/LOCAL/DISK/One-Forcing-main-c9a2350
   export PAIR_DIR=/LOCAL/DISK/one_forcing_official_pair/configs
   export FULL_RUN_DIR=/LOCAL/DISK/one_forcing_official_pair/runs/full
   export DMD_RUN_DIR=/LOCAL/DISK/one_forcing_official_pair/runs/dmd
   export FULL_EVAL_DIR=/LOCAL/DISK/one_forcing_official_pair/eval/full
   export DMD_EVAL_DIR=/LOCAL/DISK/one_forcing_official_pair/eval/dmd
   export TRAIN_PYTHON=/ABS/PATH/train-env/bin/python
   export VBENCH_PYTHON=/ABS/PATH/vbench-env/bin/python
   export ODE_CKPT=/ABS/PATH/causal_ode.pt
   export TEACHER_DIR=/ABS/PATH/Wan2.1-T2V-14B
   export WAN13_DIR=/ABS/PATH/Wan2.1-T2V-1.3B
   export CLEAN_DATA=/ABS/PATH/clean_data
   export GPU_IDS=0,1,2,3,4,5,6,7

   git worktree add --detach "$OFFICIAL_ROOT" \
     c9a2350e8740531562011fc9618e1a928d911ae0
   "$TRAIN_PYTHON" experiments/rebuttal/official_main_pair.py prepare \
     --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR"
   "$TRAIN_PYTHON" experiments/rebuttal/official_main_pair.py check \
     --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR"
   ```

   Use eight **idle** GPUs; model type need not be A100. Use the same ODE
   initialization, teacher, Wan 1.3B assets, and clean-latent LMDB in both
   arms. The launcher refuses occupied GPUs and existing run directories,
   records paths and file sizes, and only creates an ignored Wan-asset
   symlink in the official worktree. It never edits official tracked code.
   Do not pass a prompt cache, CPU-offload, manual-backward, resume, or
   alternative training-schedule flag: those were differences in the old
   80.58 retrain, not the official main README command.

2. Train the official DMD+GAN arm in tmux. Monitor the actual log through
   **at least step 10**, checking finite critic/generator losses and GPU
   usage; then wait for the step-200 checkpoint. The session name is printed
   by the launcher. Do not infer success merely from session creation.

   ```bash
   "$TRAIN_PYTHON" experiments/rebuttal/launch_official_main_pair.py \
     --arm full --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR" \
     --run_dir "$FULL_RUN_DIR" --generator_ckpt "$ODE_CKPT" \
     --teacher_model_path "$TEACHER_DIR" --wan_1_3b_path "$WAN13_DIR" \
     --data_path "$CLEAN_DATA" --gpus "$GPU_IDS" --python "$TRAIN_PYTHON"

   bash experiments/rebuttal/run_paper_aligned_vbench.sh \
     --name one_forcing_official_main \
     --checkpoint_path "$FULL_RUN_DIR/checkpoint_model_000200/model.pt" \
     --schedule ffe --output_root "$FULL_EVAL_DIR" --gpus all \
     --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"
   ```

   Check representative MP4s at beginning, middle, and end, including
   dynamic and multiple-object prompts. Record visible failures, if any.
   Compare the full-arm total with 83.76 as a diagnostic, but do not tune
   one arm or substitute the published checkpoint to force that score.

3. Once the GPUs are idle, train and evaluate DMD-only with the **same**
   inputs, GPUs, seed, official source, 200 steps, and Qwen/FFE manifest.
   Monitor this tmux job through at least step 10 and completion too.

   ```bash
   "$TRAIN_PYTHON" experiments/rebuttal/launch_official_main_pair.py \
     --arm dmd --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR" \
     --run_dir "$DMD_RUN_DIR" --generator_ckpt "$ODE_CKPT" \
     --teacher_model_path "$TEACHER_DIR" --wan_1_3b_path "$WAN13_DIR" \
     --data_path "$CLEAN_DATA" --gpus "$GPU_IDS" --python "$TRAIN_PYTHON"

   bash experiments/rebuttal/run_paper_aligned_vbench.sh \
     --name dmd_only_official_main \
     --checkpoint_path "$DMD_RUN_DIR/checkpoint_model_000200/model.pt" \
     --schedule ffe --output_root "$DMD_EVAL_DIR" --gpus all \
     --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"

   "$TRAIN_PYTHON" experiments/rebuttal/official_main_pair.py report \
     --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR" \
     --full_run_dir "$FULL_RUN_DIR" --dmd_run_dir "$DMD_RUN_DIR" \
     --full_eval_root "$FULL_EVAL_DIR" --dmd_eval_root "$DMD_EVAL_DIR" \
     --output_prefix /LOCAL/DISK/one_forcing_official_pair/official_main_gan_ablation
   ```

The report refuses a changed source or configuration, non-finite/incomplete
training logs, a wrong step or checkpoint, different assets/GPU list, or
different evaluation manifest hashes. Inspect matching videos from both
arms before citing the metrics. Do not merge in the older FFE-trained pair
or the 80.58/77.21 historical outputs.
