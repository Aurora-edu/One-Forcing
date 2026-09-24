# Published-recipe DMD-only ablation (ICLR 2027)

This is the replacement GAN ablation for the manuscript's framewise One-Forcing
result. **Train and evaluate only DMD-only.** Use the manuscript's reported
One-Forcing 83.76 total / 85.22 quality / 77.91 semantic as the DMD+GAN
reference. Do not use the older `full200_ffe` result (80.58): that run was
trained with a fixed rollout, whereas the manuscript's `ffe_config.yaml`
uses first-4-then-1 rollout during training.

`configs/train_dmd_only_published_ffe.yaml` inherits `ffe_config.yaml`, fixes
the clean-latent LMDB data type explicitly, and changes only `gan_g_weight`
and `gan_d_weight` from 0.03 to 0. The data type is also explicitly passed by
the training launcher. The remaining settings include the same ODE initial
checkpoint, 21 latent frames, seed 0, 200 optimization iterations, 5:1
critic/generator schedule, first block of 4 latent frames at four steps, and
subsequent one-step framewise blocks. `check_recipe` fails if this parity
drifts. A single seed is enough for this ablation; do not launch the old
600-step or multi-seed sweep.

## Training-machine AI instructions

1. Pull the `iclr2027-rebuttal-gan-ablation-audit` branch and work from a
   **clean checkout**. Do not stop, kill, or occupy GPUs belonging to other
   sessions. Preserve prior checkpoints and video data. The new run name and
   output paths below must not already exist. Verify the **same** framewise
   causal ODE checkpoint and clean-latent LMDB used for the published FFE run;
   do not silently substitute a different initialization or dataset. Record
   their exact paths and fingerprints in the report. Use eight idle GPUs to
   reproduce the published 8-rank batch-1 setting; GPU *type* may differ.
   The report audit refuses other world sizes because the effective batch
   would otherwise differ. If eight are unavailable, stop and tell the author.

2. Set paths below for this machine. Use the repository's Python 3.10 training
   environment and a separate working VBench environment. Do not patch the
   model or lower prompt/sample counts to make the run pass. The documented
   `--allow_b200_torch_deviation` flag is **only** for sm_100 B200 machines
   with exact `torch==2.11.0` / `torchvision==0.26.0`; omit it elsewhere.

   ```bash
   git fetch origin iclr2027-rebuttal-gan-ablation-audit
   git switch iclr2027-rebuttal-gan-ablation-audit
   git pull --ff-only origin iclr2027-rebuttal-gan-ablation-audit
   git status --short

   export TRAIN_PYTHON=/ABS/PATH/train-env/bin/python
   export VBENCH_PYTHON=/ABS/PATH/vbench-env/bin/python
   export ODE_CKPT=/ABS/PATH/causal_ode.pt
   export TEACHER_DIR=/ABS/PATH/Wan2.1-T2V-14B
   export CLEAN_DATA=/ABS/PATH/clean_data
   export GPU_IDS=0,1,2,3,4,5,6,7
   export RUN_DIR="$PWD/runs/rebuttal/dmd_only_published_ffe/seed_0"
   export EVAL_DIR="$PWD/eval/paper_aligned/dmd_only_published_ffe"

   "$TRAIN_PYTHON" experiments/rebuttal/published_ffe_dmd_ablation.py check_recipe
   bash experiments/rebuttal/launch_train.sh \
     --config_path experiments/rebuttal/configs/train_dmd_only_published_ffe.yaml \
     --run_name dmd_only_published_ffe --seed 0 --gpus "$GPU_IDS" \
     --generator_ckpt "$ODE_CKPT" --teacher_model_path "$TEACHER_DIR" \
     --data_path "$CLEAN_DATA" --python "$TRAIN_PYTHON" \
     --rank0_preload_generator_ckpt
   ```

   If the published run used the prompt-embedding cache and the same cache is
   available, pass `--prompt_embedding_cache_path /ABS/PATH/cache`; otherwise
   leave it unset and use the normal text encoder. On B200 only, append
   `--allow_b200_torch_deviation`. Hardware-only offload flags may be added
   if needed for memory, but disclose them; do not change optimizer, training
   steps, frame schedule, data, or effective batch size. The launcher starts
   a tmux session, refuses occupied GPUs, and runs asset/version preflight.

3. **Watch the training through at least step 10.** For example, inspect
   `tail -f "$RUN_DIR/train.log"`, periodic `nvidia-smi`, and the tmux
   session printed by the launcher. Verify non-NaN losses and the actual
   first-4-then-1 schedule; do not mistake an initialized process for a
   successful run. Continue monitoring to `training.done` with
   `final_step=max_steps=200` and
   `checkpoint_model_000200/model.pt`. Do not run a second seed or extend to
   600/1200 steps. If training fails, diagnose and report it; do not swap in
   the old fixed-rollout DMD-only checkpoint.

4. Run the full, pinned Qwen-rewritten 944-prompt × 5-sample, 16-dimension
   VBench evaluation with FFE. This exports the **raw generator** from the
   new DMD-only checkpoint, 21 latent/81 RGB frames, 16 fps, no sink, then
   audits the prompt hashes and generated-video count. Evaluate on idle GPUs
   only; `--gpus all` checks occupancy. Keep videos and intermediate data
   on the training machine.

   ```bash
   bash experiments/rebuttal/run_paper_aligned_vbench.sh \
     --name dmd_only_published_ffe \
     --checkpoint_path "$RUN_DIR/checkpoint_model_000200/model.pt" \
     --schedule ffe --output_root "$EVAL_DIR" --gpus all \
     --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"

   "$TRAIN_PYTHON" experiments/rebuttal/published_ffe_dmd_ablation.py report \
     --run_dir "$RUN_DIR" --eval_root "$EVAL_DIR" \
     --output_prefix "$EVAL_DIR/published_ffe_dmd_ablation"
   ```

5. Do not rely on VBench alone: **open and visually inspect representative
   MP4s** (beginning, middle, and end) from the DMD-only output, including
   moving subjects/camera, people/animals, and multi-object prompts. Note
   concrete collapse/identity drift or confirm absence in inspected samples.
   Submit the small audited Markdown/JSON result and visual-inspection notes
   for author review; leave raw videos, checkpoint, and large scorer assets
   local. Never claim visual inspection was done unless the actual clips were
   opened.

## Interpretation

The report prints 83.76/85.22/77.91 alongside the newly measured DMD-only
scores and their arithmetic differences. The published headline's exact
generation seed/noise manifest is not archived in this repository, so the
numbers are a **published-reference comparison**, not paired per-video
estimates. The Qwen rewrite file is pinned and the DMD-only side is fully
audited. If the training machine has the original headline run's manifest,
first verify it against the new protocol before making a paired-seed claim.
