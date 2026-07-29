# Reviewer experiments: execute prior items 3 → 2 → 1

Run the experiments in exactly this order:

1. Prior item 3: Qwen-matched One-Forcing versus Self-Forcing.
2. Prior item 2: DMD-only step-200 complete 16-dimension VBench.
3. Prior item 1: controlled curvature experiment.

Every GPU phase passes `--gpus all`. The runner queries `nvidia-smi`, records
the host/GPU inventory, and requires the selected set to equal every physical
GPU detected on that host. It also refuses to start if any card has a compute
process, because processes from other sessions must not be interrupted or
overlapped. On an 8×H200 host, this therefore resolves to `0,1,2,3,4,5,6,7`.

Before starting, pull the latest `review` branch. If local changes conflict,
do not reset, stash, or delete them; report the conflict.

## 1. Prior item 3: all-GPU Qwen-matched four-step comparison

This reruns both video sets; it does not reuse historical Self-Forcing videos.
One-Forcing loads the raw `generator` from the four-step step-300 checkpoint.
Self-Forcing loads `generator_ema` from the released `self_forcing_dmd.pt`.
Both methods share one manifest with 944 official prompts, the exact historical
Qwen rewrites, one sample per prompt, and seed `prompt_index` (`0...943`). Every
record resets both initial-noise and intermediate-re-noising RNGs, so results
remain exactly paired regardless of the number of GPU shards.

```bash
bash experiments/rebuttal/run_one_forcing_qwen_4step_vbench.sh \
  --one_forcing_checkpoint /path/to/of4/checkpoint_model_000300/model.pt \
  --self_forcing_checkpoint /path/to/self_forcing_dmd.pt \
  --qwen_pair_shard0 /path/to/qwen/shard00_pairs.jsonl \
  --qwen_pair_shard1 /path/to/qwen/shard01_pairs.jsonl \
  --full_info_path /path/to/VBench_full_info.json \
  --output_root /path/to/results/qwen_matched_4step_all_gpu \
  --gpus all \
  --vbench_python /path/to/pinned_vbench/bin/python \
  --python /path/to/one_forcing/bin/python
```

Known shared paths, if mounted on the experiment host:

```text
/nfs/data/Causal_forcing/vbench/prompts/cf_dmd2v_r1w5_step1000_qwen_rewrite/shard00_pairs.jsonl
/nfs/data/Causal_forcing/vbench/prompts/cf_dmd2v_r1w5_step1000_qwen_rewrite/shard01_pairs.jsonl
/data/fengjiaqi/tools/VBench_full_info.json
```

The released checkpoint must contain a non-empty `generator_ema`; the audit
records its complete SHA256. Both conditions run framewise `all4`, all 16
dimensions, and the repository-pinned VBench 0.1.5 environment. Result:

```text
/path/to/results/qwen_matched_4step_all_gpu/qwen_matched_4step_summary.json
```

## 2. Prior item 2: DMD-only step-200 complete VBench

Use the raw/no-EMA DMD-only step-200 checkpoint. The reference must be the
already completed full-method raw/no-EMA step-200, one-sample, all-16-dimension
FFE result. Read `prompt_path` and `manifest_path` from that reference run's
`videos/export.intent.json`; do not generate a merely similar manifest.

```bash
bash experiments/rebuttal/run_dmd_only_step200_full_vbench.sh \
  --dmd_checkpoint /path/to/dmd_only/checkpoint_model_000200/model.pt \
  --reference_full_result /path/to/full_step200/vbench/full_step200_eval_results.json \
  --prompt_path /exact/prompt_path/from/reference/export.intent.json \
  --manifest_path /exact/manifest_path/from/reference/export.intent.json \
  --full_info_path /path/to/VBench_full_info.json \
  --output_root /path/to/results/dmd_only_step200_full16 \
  --gpus all \
  --vbench_python /path/to/pinned_vbench/bin/python \
  --python /path/to/one_forcing/bin/python
```

The final audit rejects EMA weights, incomplete dimensions, non-one-sample
scoring, manifest/config mismatch, or failure to use every detected GPU.
Result:

```text
/path/to/results/dmd_only_step200_full16/dmd_only_step200_full16_summary.json
```

## 3. Prior item 1: controlled curvature experiment

The paired arms use the same audited trajectory LMDB, raw initialization,
prompt/noise/endpoint/clean conditioning, sample order, adjacent-state schedule,
300-step budget, and framewise model. The only treatment is whether the three
interior ODE states remain curved or are time-linearized.

Training must run in tmux:

```bash
tmux new-session -d -s reviewer_curvature "cd /path/to/One-Forcing && \
  bash experiments/rebuttal/run_curvature_cd_small.sh all \
    --data_path /path/to/audited_curvature_lmdb \
    --raw_checkpoint /path/to/the_same_raw_source_model.pt \
    --output_root /path/to/results/curvature_cd_small \
    --gpus all \
    --full_info_path /path/to/VBench_full_info.json \
    --vbench_python /path/to/pinned_vbench/bin/python \
    --python /path/to/one_forcing/bin/python"
```

Immediately monitor the pane and metrics. Do not leave either arm unattended
before it prints steps 1 through 10:

```bash
tmux capture-pane -pt reviewer_curvature -S -120
tail -n 10 /path/to/results/curvature_cd_small/training/curved/metrics.jsonl
tail -n 10 /path/to/results/curvature_cd_small/training/rectified/metrics.jsonl
```

The rectified arm starts only after curved reaches step 300. Confirm the curved
`training.done`, then explicitly watch rectified through step 10. Both arms must
record `world_size == detected_gpu_count`. Evaluation is raw/no-EMA, one sample,
and all 16 dimensions for all1 and all4. Result:

```text
/path/to/results/curvature_cd_small/curvature_cd_small_summary.json
```

Use `rectification_gain_all1` as the pre-registered primary statistic, all4 as
the aligned negative control, and `curvature_causal_effect` as the
difference-in-differences. Do not change checkpoints, seeds, dimensions,
training length, or prompt files after observing results.
