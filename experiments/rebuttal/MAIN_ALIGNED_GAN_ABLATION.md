# GitHub-main-aligned One-Forcing GAN ablation

GitHub `main@c9a2350` documents `torchrun --nproc_per_node=8 train.py
--config_path config.yaml` with the clean-latent LMDB. Its `config.yaml` has
`max_steps: 200`, `denoising_step_list: [1000]`, one latent frame per block,
and GAN weights 0.03/0.03. It has **no first-four-step FFE training rollout**.
FFE is an **inference** setting. The separate `ffe_config.yaml` was added
later; it is not the training recipe used by current GitHub `main`.

The two configs in this plan inherit the local `config.yaml`, which the
`check_recipe --recipe main` audit compares against the pinned GitHub main
commit. The only objective difference is `gan_g_weight` and `gan_d_weight`:

| Arm | Config | GAN weights |
|---|---|---:|
| One-Forcing / DMD+GAN | `configs/train_one_forcing_main_aligned.yaml` | 0.03 / 0.03 |
| DMD-only | `configs/train_dmd_only_main_aligned.yaml` | 0 / 0 |

Both train with **fixed one-step framewise rollout** for 200 iterations,
then evaluate using the same FFE first-four-step inference schedule, pinned
Qwen rewrites, 944 prompts × 5 generated samples, all 16 VBench dimensions,
and raw generator weights. The two newly measured scores—not the manuscript's
83.76 reference—define the paired GAN gain.

Important limits: `main`'s older trainer interpreted YAML `seed: 0` as a
request to draw a random runtime seed. This paired run deliberately fixes
runtime seed 0 in **both** arms so the ablation is reproducible; report that
controlled deviation. The rebuttal branch's training code has additional
resource-handling and validation changes relative to `main`; matching the
configuration alone does not prove bitwise reproduction of the released
`one_forcing.pt`, whose checkpoint does not contain a resolved training config.

## Instructions for the training-machine AI

1. Use a clean checkout of `iclr2027-rebuttal-gan-ablation-audit` and fetch
   GitHub `main` over HTTPS so the pinned `c9a2350` object is available.
   Verify that the ODE initialization, Wan teacher, and clean-latent LMDB are
   the intended assets; record their exact paths and fingerprints. Use eight
   idle GPUs to keep the published batch size (one sample per GPU). GPU model
   need not be A100. Never stop another session's process. In particular,
   the already-started **FFE-trained pair** is a different experiment: do not
   alter its running process, checkpoints, or directories and do not reuse its
   weights here. If 48 hours cannot accommodate this entire new pair and
   evaluation, report that before starting; do not shrink the protocol.

2. From the repository root, configure machine-specific absolute paths.
   Use the **same** checkpoint, teacher, data, optional prompt cache, GPU IDs,
   offload flags, and seed in both runs. The launcher runs preflight, refuses
   occupied GPUs and existing run directories, and starts training in tmux.
   On B200/sm_100 only, append `--allow_b200_torch_deviation` to **both**
   launches if using torch 2.11.0/torchvision 0.26.0. Any hardware-only
   offload option must also match in both arms.

   ```bash
   git fetch https://github.com/Aurora-edu/One-Forcing.git main
   git status --short

   export TRAIN_PYTHON=/ABS/PATH/train-env/bin/python
   export VBENCH_PYTHON=/ABS/PATH/vbench-env/bin/python
   export ODE_CKPT=/ABS/PATH/causal_ode.pt
   export TEACHER_DIR=/ABS/PATH/Wan2.1-T2V-14B
   export CLEAN_DATA=/ABS/PATH/clean_data
   export GPU_IDS=0,1,2,3,4,5,6,7
   export FULL_RUN_DIR="$PWD/runs/rebuttal/one_forcing_paired_main/seed_0"
   export DMD_RUN_DIR="$PWD/runs/rebuttal/dmd_only_paired_main/seed_0"
   export FULL_EVAL_DIR="$PWD/eval/paper_aligned/one_forcing_paired_main"
   export DMD_EVAL_DIR="$PWD/eval/paper_aligned/dmd_only_paired_main"

   "$TRAIN_PYTHON" experiments/rebuttal/published_ffe_dmd_ablation.py \
     check_recipe --recipe main

   bash experiments/rebuttal/launch_train.sh \
     --config_path experiments/rebuttal/configs/train_one_forcing_main_aligned.yaml \
     --run_name one_forcing_paired_main --seed 0 --gpus "$GPU_IDS" \
     --generator_ckpt "$ODE_CKPT" --teacher_model_path "$TEACHER_DIR" \
     --data_path "$CLEAN_DATA" --python "$TRAIN_PYTHON"
   ```

3. Watch the DMD+GAN tmux run through **at least step 10**, checking actual
   losses and GPU usage, then to `training.done` and the step-200 checkpoint.
   Do not infer completion merely because tmux exists. Evaluate this arm
   first; actually open dynamic and multi-object MP4s and inspect their
   beginning, middle, and end. Compare its measured score with 83.76 as a
   reproduction diagnostic, **not** as a score target to tune toward. If
   materially different, report the real score and inspect provenance before
   spending resources on the DMD-only run.

   ```bash
   bash experiments/rebuttal/run_paper_aligned_vbench.sh \
     --name one_forcing_paired_main \
     --checkpoint_path "$FULL_RUN_DIR/checkpoint_model_000200/model.pt" \
     --schedule ffe --output_root "$FULL_EVAL_DIR" --gpus all \
     --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"
   ```

4. When the GPUs are idle and the full arm's setup is confirmed, launch the
   DMD-only arm with the same options, monitor it through step 10 and through
   completion, then evaluate under the **same** pinned manifest. Do not
   substitute the older fixed-rollout rebuttal checkpoint or FFE-trained
   pair, change a single arm after seeing its result, or add extra seeds.

   ```bash
   bash experiments/rebuttal/launch_train.sh \
     --config_path experiments/rebuttal/configs/train_dmd_only_main_aligned.yaml \
     --run_name dmd_only_paired_main --seed 0 --gpus "$GPU_IDS" \
     --generator_ckpt "$ODE_CKPT" --teacher_model_path "$TEACHER_DIR" \
     --data_path "$CLEAN_DATA" --python "$TRAIN_PYTHON"

   bash experiments/rebuttal/run_paper_aligned_vbench.sh \
     --name dmd_only_paired_main \
     --checkpoint_path "$DMD_RUN_DIR/checkpoint_model_000200/model.pt" \
     --schedule ffe --output_root "$DMD_EVAL_DIR" --gpus all \
     --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"

   "$TRAIN_PYTHON" experiments/rebuttal/published_ffe_dmd_ablation.py \
     report --recipe main \
     --full_run_dir "$FULL_RUN_DIR" --dmd_run_dir "$DMD_RUN_DIR" \
     --full_eval_root "$FULL_EVAL_DIR" --dmd_eval_root "$DMD_EVAL_DIR" \
     --output_prefix "$PWD/eval/paper_aligned/paired_main_gan_ablation"
   ```

5. Open matching MP4s from **both** arms at beginning/middle/end; document
   subject identity, motion and any collapse. The report audit requires the
   same executed training settings (except GAN weights and run paths), clean
   code revision, eight ranks, step-200 raw checkpoints, identical evaluation
   prompt/rewrite/seed hashes, and complete scores. Leave raw videos and
   weights on the training machine. Publish only the audited small metrics
   report and genuine visual-inspection notes after author review.
