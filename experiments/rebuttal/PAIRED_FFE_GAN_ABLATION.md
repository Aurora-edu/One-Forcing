# Paired FFE GAN ablation for One-Forcing

This replaces the earlier one-arm comparison to the manuscript's 83.76
headline. **Newly train and evaluate both arms** under the repository's
documented `ffe_config.yaml` One-Forcing recipe:

| Arm | Training config | Objective |
|---|---|---|
| DMD+GAN / full One-Forcing | `configs/train_one_forcing_published_ffe.yaml` | `gan_g_weight=gan_d_weight=0.03` |
| DMD-only | `configs/train_dmd_only_published_ffe.yaml` | `gan_g_weight=gan_d_weight=0` |

Both configs inherit `ffe_config.yaml`; the only arm-to-arm configuration
difference is the two GAN weights. Each run uses the **same** framewise causal
ODE initialization, clean-latent LMDB, Wan teacher, fixed training seed 0,
eight ranks with batch size 1, 200 iterations, 5:1 critic/generator schedule,
and FFE first-4-then-1 self-rollout. Keep all hardware-only options and the
prompt cache setting equal. The pair is one controlled training seed, not a
cross-training-seed variance estimate. Do not substitute the older
fixed-rollout `full200_ffe`/`dmd200_ffe` pair or use 83.76 as one arm.

The manuscript's 83.76 / 85.22 / 77.91 is an **external reproduction check**
for the newly trained full arm; it is never used to calculate the GAN gain.
The exact original headline training metadata and generation manifest are not
archived in this repository. Matching the documented recipe cannot guarantee
reproduction of that particular numerical score. If the full score differs,
investigate and disclose the difference rather than changing the result or
tuning one arm only.

## Training-machine AI instructions

1. On a clean checkout of branch `iclr2027-rebuttal-gan-ablation-audit`, read
   this whole runbook and run `check_recipe`. The new machine need not have
   GitHub SSH; clone/fetch over HTTPS. Confirm that the ODE checkpoint and
   clean-latent LMDB are the intended ones for the One-Forcing experiment.
   Record their paths and hashes/fingerprints. Do not silently replace them.
   Use eight **idle** GPUs for the published global batch size; GPU type may
   differ. If eight GPUs are unavailable, stop and report that matched
   training cannot be run as configured. Never stop another session's process.

2. Set machine-specific paths. Use the same values for **both** arms. The
   training launcher creates tmux sessions, checks GPU occupancy and inputs,
   and refuses existing output directories. `--allow_b200_torch_deviation`
   may be appended on B200/sm_100 only with exact torch 2.11.0 and
   torchvision 0.26.0; omit it elsewhere. If a prompt-embedding cache is
   used, append its **same** `--prompt_embedding_cache_path` to both runs.

   ```bash
   git clone --branch iclr2027-rebuttal-gan-ablation-audit \
     https://github.com/Aurora-edu/One-Forcing.git One-Forcing
   cd One-Forcing
   git status --short

   export TRAIN_PYTHON=/ABS/PATH/train-env/bin/python
   export VBENCH_PYTHON=/ABS/PATH/vbench-env/bin/python
   export ODE_CKPT=/ABS/PATH/causal_ode.pt
   export TEACHER_DIR=/ABS/PATH/Wan2.1-T2V-14B
   export CLEAN_DATA=/ABS/PATH/clean_data
   export GPU_IDS=0,1,2,3,4,5,6,7
   export FULL_RUN_DIR="$PWD/runs/rebuttal/one_forcing_paired_ffe/seed_0"
   export DMD_RUN_DIR="$PWD/runs/rebuttal/dmd_only_paired_ffe/seed_0"
   export FULL_EVAL_DIR="$PWD/eval/paper_aligned/one_forcing_paired_ffe"
   export DMD_EVAL_DIR="$PWD/eval/paper_aligned/dmd_only_paired_ffe"

   "$TRAIN_PYTHON" experiments/rebuttal/published_ffe_dmd_ablation.py check_recipe

   bash experiments/rebuttal/launch_train.sh \
     --config_path experiments/rebuttal/configs/train_one_forcing_published_ffe.yaml \
     --run_name one_forcing_paired_ffe --seed 0 --gpus "$GPU_IDS" \
     --generator_ckpt "$ODE_CKPT" --teacher_model_path "$TEACHER_DIR" \
     --data_path "$CLEAN_DATA" --python "$TRAIN_PYTHON" \
     --rank0_preload_generator_ckpt
   ```

3. Watch the full run **through at least step 10** in the tmux session and
   `"$FULL_RUN_DIR/train.log"`; check finite critic/generator losses, actual
   FFE rollout, and GPU activity. Keep monitoring to `training.done` and
   step-200 checkpoint. Evaluate and visually inspect the newly trained full
   One-Forcing arm first, so its actual score can be checked against the
   manuscript headline before spending resources on DMD-only. If it differs
   materially, inspect configuration, initialization, data, Qwen prompts,
   scorer, and checkpoint provenance; report the discrepancy to the author
   instead of tuning either arm to reach a target number.

   ```bash
   bash experiments/rebuttal/run_paper_aligned_vbench.sh \
     --name one_forcing_paired_ffe \
     --checkpoint_path "$FULL_RUN_DIR/checkpoint_model_000200/model.pt" \
     --schedule ffe --output_root "$FULL_EVAL_DIR" --gpus all \
     --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"
   ```

   **Do not start the second run while the first training or evaluation still
   occupies its GPUs.** Once the full arm's setup is confirmed, launch
   DMD-only with identical paths, GPUs, cache, and hardware-only flags:

   ```bash
   bash experiments/rebuttal/launch_train.sh \
     --config_path experiments/rebuttal/configs/train_dmd_only_published_ffe.yaml \
     --run_name dmd_only_paired_ffe --seed 0 --gpus "$GPU_IDS" \
     --generator_ckpt "$ODE_CKPT" --teacher_model_path "$TEACHER_DIR" \
     --data_path "$CLEAN_DATA" --python "$TRAIN_PYTHON" \
     --rank0_preload_generator_ckpt
   ```

   Watch this run through at least step 10 as well, then to
   `training.done` and its step-200 checkpoint. Do not extend either run to
   600/1200 steps, add extra training seeds, reuse old weights, or modify only
   one arm after seeing its score. Diagnose actual failures rather than
   substituting a different experiment.

4. Run the DMD-only full Qwen-conditioned VBench evaluation. Both arms use the same
   pinned 944-prompt × 5-sample generation manifest and all 16 dimensions,
   FFE, 21 latent/81 RGB frames, 16 fps, no attention sink, and raw generator
   weights. The evaluator checks idle GPUs and audits prompt/rewrite hashes,
   video count, and output settings. Keep videos and checkpoints locally.

   ```bash
   bash experiments/rebuttal/run_paper_aligned_vbench.sh \
     --name dmd_only_paired_ffe \
     --checkpoint_path "$DMD_RUN_DIR/checkpoint_model_000200/model.pt" \
     --schedule ffe --output_root "$DMD_EVAL_DIR" --gpus all \
     --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"

   "$TRAIN_PYTHON" experiments/rebuttal/published_ffe_dmd_ablation.py report \
     --full_run_dir "$FULL_RUN_DIR" --dmd_run_dir "$DMD_RUN_DIR" \
     --full_eval_root "$FULL_EVAL_DIR" --dmd_eval_root "$DMD_EVAL_DIR" \
     --output_prefix "$PWD/eval/paper_aligned/paired_ffe_gan_ablation"
   ```

5. **Open actual MP4s from both arms** for the same representative prompts
   and seeds. Inspect beginning/middle/end of dynamic, human/animal, and
   multi-object clips; record concrete failures or their absence. Do not
   infer video integrity from VBench scores alone. Submit the audited small
   Markdown/JSON report and visual-inspection notes for author review; raw
   videos and checkpoints remain on the training machine. If the new full
   One-Forcing result diverges from 83.76, explicitly report its value and
   inspect configuration/protocol provenance before using the pair to revise
   the manuscript.

The audit script requires identical executed training settings except GAN
weights and run names/paths, the same clean code revision, eight ranks,
step-200 raw checkpoints, identical Qwen/seed manifest hashes, and complete
16-dimensional scores. It calculates `GAN gain = new full − new DMD-only`.
The current paper appendix describes the older GAN comparison as using
original prompts; if this Qwen-conditioned pair replaces that table, update
the table and protocol text together rather than mixing their numbers.
