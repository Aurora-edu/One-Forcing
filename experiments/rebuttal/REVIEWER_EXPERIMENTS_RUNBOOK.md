# Reviewer experiments: strict execution order

Run these three experiments in order. Do not substitute EMA checkpoints, prompt
files, seeds, sample counts, VBench environments, or a different training
budget. Do not stop or overlap processes belonging to another session.

Before starting, pull the `review` branch and locate the real inputs. Every
placeholder below must be replaced with an existing path; do not create a
fallback input merely to make a command start.

## 1. Controlled curvature experiment

The two arms start from the same raw checkpoint and use the same audited
trajectory LMDB, initialization, shuffled row order, 300-step budget, adjacent
timestep schedule, prompt conditioning, and framewise model. The only treatment
is whether the three interior ODE states remain curved or are time-linearized.
The initial state, ODE endpoint, and clean conditioning latent are bitwise fixed.

Run the full pipeline in tmux:

```bash
tmux new-session -d -s reviewer_curvature "cd /path/to/One-Forcing && \
  bash experiments/rebuttal/run_curvature_cd_small.sh all \
    --data_path /path/to/audited_curvature_lmdb \
    --raw_checkpoint /path/to/the_same_raw_source_model.pt \
    --output_root /path/to/results/curvature_cd_small \
    --gpus 0,1,2,3,4,5,6,7 \
    --full_info_path /path/to/VBench_full_info.json \
    --vbench_python /path/to/pinned_vbench/bin/python \
    --python /path/to/one_forcing/bin/python"
```

Immediately monitor the tmux pane and the JSONL metrics. Do not leave either
training arm unattended before it has completed steps 1 through 10:

```bash
tmux capture-pane -pt reviewer_curvature -S -120
tail -n 10 /path/to/results/curvature_cd_small/training/curved/metrics.jsonl
tail -n 10 /path/to/results/curvature_cd_small/training/rectified/metrics.jsonl
```

The rectified arm begins only after the curved arm completes. Continue polling
until the first arm has `training.done`, then explicitly watch the second arm to
step 10. The final result is:

```text
/path/to/results/curvature_cd_small/curvature_cd_small_summary.json
```

Use `rectification_gain_all1` as the pre-registered primary result, `all4` as
the aligned negative control, and `curvature_causal_effect` as the
difference-in-differences. All four VBench runs are one-sample, all-16-dimension,
raw/no-EMA evaluations.

## 2. DMD-only step-200 complete VBench

Use the DMD-only raw step-200 checkpoint. The reference must be the already
completed full-method raw step-200, one-sample, all-16-dimension FFE result.
Pass the exact prompt and manifest files recorded by that reference run's
`videos/export.intent.json`; do not regenerate a merely similar manifest.

```bash
bash experiments/rebuttal/run_dmd_only_step200_full_vbench.sh \
  --dmd_checkpoint /path/to/dmd_only/checkpoint_model_000200/model.pt \
  --reference_full_result /path/to/full_step200/vbench/full_step200_eval_results.json \
  --prompt_path /exact/path/from/full_step200/export.intent.json \
  --manifest_path /exact/path/from/full_step200/export.intent.json \
  --full_info_path /path/to/VBench_full_info.json \
  --output_root /path/to/results/dmd_only_step200_full16 \
  --gpus 0,1,2,3,4,5,6,7 \
  --vbench_python /path/to/pinned_vbench/bin/python \
  --python /path/to/one_forcing/bin/python
```

For `--prompt_path`, use the intent's `prompt_path`; for `--manifest_path`, use
its `manifest_path`. The final audit rejects EMA weights, incomplete dimensions,
non-one-sample scoring, or any manifest/resolved-config mismatch. Result:

```text
/path/to/results/dmd_only_step200_full16/dmd_only_step200_full16_summary.json
```

## 3. Qwen-matched raw One-Forcing versus Self-Forcing

This is the step-300 four-step One-Forcing model used by the existing all4
comparison, evaluated from its raw `generator` weights. The runner reconstructs
the exact Qwen rewrites from the two historical `{prompt,rewrite}` shards in
official VBench prompt order. It then reproduces the historical Self-Forcing
generation protocol: two even/odd prompt processes, each seeded once with 0,
one sample for every one of the 944 prompts. It scores both video sets in the
same pinned VBench 0.1.5 environment.

Exactly two generation GPUs are required to preserve the historical RNG
protocol:

```bash
bash experiments/rebuttal/run_one_forcing_qwen_4step_vbench.sh \
  --one_forcing_checkpoint /path/to/of4/checkpoint_model_000300/model.pt \
  --qwen_pair_shard0 /path/to/qwen/shard00_pairs.jsonl \
  --qwen_pair_shard1 /path/to/qwen/shard01_pairs.jsonl \
  --self_forcing_videos_path /path/to/self_forcing_ema_all4_qwen_videos \
  --full_info_path /path/to/VBench_full_info.json \
  --output_root /path/to/results/qwen_matched_4step \
  --gpus 0,1 \
  --vbench_python /path/to/pinned_vbench/bin/python \
  --python /path/to/one_forcing/bin/python
```

The One-Forcing video-generation directory must be new and empty. A partial
sequential-RNG export cannot be resumed without changing later random draws; if
generation fails, diagnose the cause and rerun into a new output root. Do not
delete the partial directory or interrupt unrelated sessions.

The final result is:

```text
/path/to/results/qwen_matched_4step/qwen_matched_4step_summary.json
```

The summary explicitly labels One-Forcing as raw/no-EMA and the released
Self-Forcing checkpoint as EMA. It also records the exact prompt, rewrite,
manifest, video-set, generation-protocol, and VBench provenance audits.
