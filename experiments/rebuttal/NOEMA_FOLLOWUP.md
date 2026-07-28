# Ordered raw/no-EMA follow-up experiments

All commands below fail closed on EMA provenance. VBench uses one generated
sample (`sample_index=0`) per official prompt and scores all 16 dimensions. The
result is deliberately labeled as a one-sample protocol, not an official
five-sample leaderboard submission.

Use the same Python environment that runs One-Forcing for `--python`, and the
separate audited VBench 0.1.5 environment for `--vbench_python`.

## 1. Controlled curvature intervention

This experiment constructs a paired intervention from the same raw AR teacher
trajectories. The rectified arm replaces only the three intermediate teacher
states with exact time-aware linear interpolation. Prompt, initial noise, ODE
endpoint, clean conditioning latent, timestep grid, initialization checkpoint,
sample order, optimizer, seed, and training length are identical between arms.

Run the phases in order. Curvature training is intentionally rejected outside
tmux. Keep the pane attached (or tail its output) through at least step 10; the
trainer prints every one of the first ten steps.

```bash
tmux new -s curvature_noema

bash experiments/rebuttal/run_curvature_causal_experiment.sh prepare \
  --ar_checkpoint /path/to/raw_ar/model.pt \
  --clean_lmdb /path/to/clean_latents_lmdb \
  --output_root /path/to/rebuttal/curvature_control \
  --gpus 0,1,2,3,4,5,6,7

bash experiments/rebuttal/run_curvature_causal_experiment.sh train_curved \
  --ar_checkpoint /path/to/raw_ar/model.pt \
  --clean_lmdb /path/to/clean_latents_lmdb \
  --output_root /path/to/rebuttal/curvature_control \
  --gpus 0,1,2,3,4,5,6,7

bash experiments/rebuttal/run_curvature_causal_experiment.sh train_rectified \
  --ar_checkpoint /path/to/raw_ar/model.pt \
  --clean_lmdb /path/to/clean_latents_lmdb \
  --output_root /path/to/rebuttal/curvature_control \
  --gpus 0,1,2,3,4,5,6,7

bash experiments/rebuttal/run_curvature_causal_experiment.sh evaluate \
  --ar_checkpoint /path/to/raw_ar/model.pt \
  --clean_lmdb /path/to/clean_latents_lmdb \
  --output_root /path/to/rebuttal/curvature_control \
  --gpus 0,1,2,3,4,5,6,7 \
  --full_info_path /path/to/VBench_full_info.json \
  --vbench_python /path/to/vbench/bin/python

bash experiments/rebuttal/run_curvature_causal_experiment.sh summarize \
  --ar_checkpoint /path/to/raw_ar/model.pt \
  --clean_lmdb /path/to/clean_latents_lmdb \
  --output_root /path/to/rebuttal/curvature_control \
  --gpus 0,1,2,3,4,5,6,7
```

The primary causal readout is
`curvature_causal_effect = (rectified_all1 - curved_all1) -
(rectified_all4 - curved_all4)`. A positive effect means straightening helps
one-step inference more than four-step inference, matching the curvature
mechanism. Inspect `curvature_causal_summary.json` and the per-dimension deltas.

## 2. Step-200 one-step versus four-step inference

Both conditions use the exact same raw step-200 checkpoint and manifest. `all1`
is framewise one-step; `all4` is framewise four-step at every generated frame.

```bash
bash experiments/rebuttal/run_step200_fourstep_comparison.sh \
  --checkpoint /path/to/one_forcing/checkpoint_model_000200/model.pt \
  --full_info_path /path/to/VBench_full_info.json \
  --output_root /path/to/rebuttal/step200_1v4 \
  --gpus 0,1,2,3,4,5,6,7 \
  --vbench_python /path/to/vbench/bin/python
```

The audited comparison is written to `step200_fourstep_comparison.json`.

## 3. Complete VBench for raw step-200 and step-400

Both checkpoints use the paper's FFE schedule: four denoising steps for the
first four-frame rollout, then framewise one-step rollout.

```bash
bash experiments/rebuttal/run_full_vbench_noema_single_seed.sh \
  --step200_checkpoint /path/to/one_forcing/checkpoint_model_000200/model.pt \
  --step400_checkpoint /path/to/one_forcing/checkpoint_model_000400/model.pt \
  --full_info_path /path/to/VBench_full_info.json \
  --output_root /path/to/rebuttal/raw_step200_step400 \
  --gpus 0,1,2,3,4,5,6,7 \
  --vbench_python /path/to/vbench/bin/python
```

The final audited summary is `raw_step200_step400_full_vbench.json`. Each run
also contains `videos/export.done` (must say `weight_source: generator` and
`use_ema: false`) and a VBench protocol JSON recording one sample per prompt.
