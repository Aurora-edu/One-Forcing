# Curvature controlled experiment: small adjacent-state CD rerun

This rerun fixes the causal exposure problem in the earlier endpoint-regression
experiment.  It reuses the audited paired raw/no-EMA trajectory LMDB, so no new
teacher ODE generation is required.

## Fixed design

- Two arms differ only in the stored ODE path: original curved states versus
  time-linearized interior states. Prompt, initial noise, ODE endpoint, clean
  conditioning, timesteps, initialization, sample order, and seed are shared.
  Every global row index is recorded, and evaluation refuses to start unless
  the completed arms have the same sample-order digest and all controls match.
- Train the online generator with adjacent-state consistency:
  `f_theta(x_t, t) = stopgrad(f_target(x_s, s))` for `t > s > 0`, and the
  identity boundary target `x_0` when `s = 0`.
- The four pairs are swept from low noise to high noise. With 300 updates, every
  pair receives exactly 75 updates in each arm.
- Only the first 9 latent frames are used during training. The generator remains
  framewise (`num_frame_per_block: 1`).
- The target network is an algorithmic EMA used only inside the stop-gradient
  training target (`decay: 0.95`). It is not checkpointed. Every saved and
  evaluated checkpoint contains only the raw online `generator`; VBench runs
  reject EMA inference with `--require_no_ema`.
- Evaluation uses one fixed seed and all 16 VBench dimensions. The existing
  warped `all4` schedule resolves exactly to the four trajectory inputs:
  `[1000, 937.5, 833.3333, 625]`.

The pre-registered primary statistic is `rectified_all1 - curved_all1`. The
aligned four-step difference is the negative control. The reported
difference-in-differences is:

`(rectified_all1 - curved_all1) - (rectified_all4 - curved_all4)`.

Do not select checkpoints, seeds, dimensions, or training length after looking
at VBench. The fixed final checkpoint is step 300.

## Run order

All training must run in tmux. Start with the fail-closed preflight, then train
the two arms, then evaluate. The `all` phase follows that order automatically:

```bash
tmux new -s curvature_cd
cd /path/to/One-Forcing
bash experiments/rebuttal/run_curvature_cd_small.sh all \
  --data_path /path/to/shared_curvature_lmdb \
  --raw_checkpoint /path/to/raw_ar_model.pt \
  --output_root /path/to/curvature_cd_small \
  --gpus 0,1,2,3,4,5,6,7 \
  --full_info_path /path/to/VBench_full_info.json \
  --vbench_python /path/to/vbench/environment/bin/python \
  --python /path/to/causal_forcing/environment/bin/python
```

Watch the live log and do not leave the training unattended before both arms
have printed steps 1 through 10. Each step reports the exact adjacent pair,
target type, loss, gradient norm, and duration.
