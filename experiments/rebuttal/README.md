# NeurIPS 2026 rebuttal：48 小时 / 8×A100 执行手册

本手册只覆盖这轮 48 小时内实际要跑的实验。Long-video 按当前决定暂不执行；它会作为未回答的
限制明确写进 rebuttal，不会用短视频或 smoke 结果替代。

## 1. 最小实验矩阵

审稿意见没有要求每个实验跑 3 个训练 seed，也没有要求训练到 1200 step。Reviewer dGCl 要求
说明结果是 single-seed 还是 multi-seed，并建议提供少量 runs **或 checkpoints** 的方差；
Reviewer dP5 明确要求的是 200 step 后的多个 VBench checkpoints。因此本轮全部训练使用
`seed=0`，如实报告 single-seed，不报告虚假的跨 seed 均值、标准差或置信区间。

| 问题 | 训练 | 评测 | 为什么已经够回答问题 |
|---|---:|---|---|
| 200 step 后是否稳定 | 完整 One-Forcing 跑到 600 | step 100/200/400/600 | 600 是论文所称 200-step 收敛点的 3 倍；审稿人未要求 1000/1200 |
| GAN 的贡献 | DMD-only 跑到 200 | 对完整方法的 step 200，同 schedule/manifest | 两边恰好同初始化、同数据、同 200-step 预算 |
| FFE 的贡献 | 不训练 | 完整方法 step 600，FFE on/off | 必须使用同一个 checkpoint |
| 4-step 能否工作 | 4-step One-Forcing 跑到 300 | 对 4-step Self-Forcing，双方 `all4` | 该 run 只回答 few-step 泛化，不承担长期稳定性证明 |
| 多样性/分布覆盖 | 不增加训练 | step-200 GAN 配对 + 1-step SF；LPIPS、FVD、P/R/coverage | 直接回答 mode-seeking concern |
| 延迟 | 不增加训练 | `all1`、FFE、`all4` | 报 first block、steady block、端到端 |

因此正式训练总量是：

```text
完整 One-Forcing:       600 step
DMD-only:               200 step
4-step One-Forcing:     300 step
合计:                  1100 step，均为 seed 0
```

不运行 `GAN-only`。审稿人要求的是 `DMD-only` 对 `DMD+GAN`；GAN-only 既不是所要求的消融，
也无法替代 fake-score DMD critic。

本轮不执行：

- long rollout / error accumulation：按当前决定延期，rebuttal 中标为未覆盖；
- curvature 的 rectification/reflow 因果干预：仓库没有对应的受控 rectified checkpoint，
  因此只收窄论文中的因果表述，不声称实验已完成；
- action-conditioned generation：不属于本文 text-to-video 设置；
- 新的人评：不占用这 48 小时 GPU 计划；原人评和 checklist 的矛盾需要文字修正。

## 2. 48 小时预算

本机 4×RTX A6000 的完整 10-step 实测按 `1 次 generator + 4 次 critic` 更新周期校正后为：

| 配置 | 秒/step | 本轮 step | 4×A6000 外推 |
|---|---:|---:|---:|
| 完整 1-step | 76.16 | 600 | 12.69 h |
| DMD-only | 42.99 | 200 | 2.39 h |
| 4-step | 85.68 | 300 | 7.14 h |

用相对保守的 8×A100 仅 `1.5×` 加速估计，三段正式训练合计约 14.8 h；计划上限记为
**16 h**。不能把这个估计当作实测：每个正式 tmux run 都必须盯到 step 10，并用目标机器的
`per_iteration_time` 更新 ETA。

评测规模固定为：

| 部分 | 视频数 |
|---|---:|
| 4 个 final official VBench 条件，944 prompts × 5 | 18,880 |
| GAN 配对的 7-dimension subset，326 prompts × 5 × 2 | 3,260 |
| stability 额外 step 100/400，233 prompts × 5 × 2 | 2,330 |
| diversity，100 prompts × 4 × 3 methods | 1,200 |
| FVD fake，256 × 2 methods | 512 |

共生成 26,182 个 81-frame 视频，另解码 256 个真实视频。若目标 A100 实测不超过
15 s/video，8 卡分片生成约 13.6 h；VBench/FVD/LPIPS/latency 预留 6 h，环境预检 2 h，
总预算约 37.6 h，留下约 10.4 h 的硬件波动与 I/O 余量。

现有发布 checkpoint 在单张 A6000 上的 21-latent 端到端实测为：`all1=12.10 s`、
`FFE=13.17 s`、`all4=21.10 s`。按同一个保守 `1.5×` A100 加速折算分别为 8.07、8.78、
14.07 s，因此上面的 A100 `15 s/video` 预算也覆盖两个较慢的 `all4` official 条件；目标机
仍需先实测确认。把目标机 §7.3 profiler 的三个 `total_ms.mean / 1000` 填入：

```bash
python experiments/rebuttal/estimate_evaluation_eta.py \
  --all1_seconds TARGET_ALL1_SECONDS \
  --ffe_seconds TARGET_FFE_SECONDS \
  --all4_seconds TARGET_ALL4_SECONDS \
  --num_gpus 8
```

输出按本手册固定的 `all1=5,120`、`FFE=11,622`、`all4=9,440` 个视频分别计时，不能再用
单一 schedule 的速度外推。

存储至少预留 **350 GiB**：main/DMD/4-step 分别保存 6/2/3 个 checkpoint（约 11×17 GiB），
其余空间用于约 2.6 万个视频、真实视频、VBench 模型缓存和结果。不要在正式 run 中途删除
checkpoint 来腾空间。

硬规则：

1. 48 小时从目标机器环境预检开始计时。
2. 三个训练 run 串行使用同一组 8 卡；不能同时抢卡。
3. 训练必须由 `launch_train.sh` 放进 tmux，且人工确认至少到 step 10、loss 有限、ETA 在预算内。
4. 推理用 `run_sharded_inference.py` 将同一 manifest 无重叠地切到 8 卡；禁止单卡串行跑正式集。
5. 不能减少 official VBench 的每 prompt 5 samples，并把结果仍称为 official score。

## 3. 环境与资产

训练环境：

```bash
conda create -n one_forcing python=3.10 -y
conda activate one_forcing
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

VBench 依赖与训练环境冲突，必须单独安装：

```bash
conda create -n one_forcing_vbench python=3.10 -y
conda activate one_forcing_vbench
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-vbench.txt
export VBENCH_PYTHON="$(which python)"
export VBENCH_FULL_INFO="$("$VBENCH_PYTHON" -c \
  'import pathlib, vbench; print(pathlib.Path(vbench.__file__).parent / "VBench_full_info.json")')"
conda activate one_forcing
```

准备：

```bash
export ODE_CKPT=checkpoints/framewise/causal_ode.pt
export TEACHER_DIR=wan_models/Wan2.1-T2V-14B
export CLEAN_LMDB=clean_data
export SF4_CKPT=PATH_TO_PAPER_4STEP_SELF_FORCING_CHECKPOINT
export SF1_CKPT=PATH_TO_PAPER_1STEP_SELF_FORCING_CHECKPOINT
```

还需要 `wan_models/Wan2.1-T2V-1.3B/` 下的 VAE、1.3B generator、T5 权重/tokenizer，以及
`Wan2.1-T2V-14B/config.json` 和全部 teacher safetensor shards。`SF4_CKPT` 是 E1 的必要外部
对照；缺少它时不能声称 4-step reviewer concern 已回答，并且需要核对它使用论文所述的同一
Wan2.1-14B teacher/ODE 数据来源。`SF1_CKPT` 用于 diversity baseline。

静态预检：

```bash
python experiments/rebuttal/validate_configs.py
python -m unittest -v tests/test_rebuttal_tools.py
python -m compileall -q train.py trainer pipeline scripts experiments/rebuttal utils
```

`validate_configs.py` 会确认：

- 完整方法和 DMD-only 的训练配置只差实验名与两个 GAN 权重；
- 1-step 和 4-step 配置只差实验名与 denoising schedule；
- 固定 seed、framewise rollout、clean-latent LMDB 和 checkpoint 间隔一致。

先建立完整 prompt embedding cache，三个 run 共享：

```bash
python experiments/rebuttal/build_prompt_cache.py \
  --config_path experiments/rebuttal/configs/train_1step_one_forcing.yaml \
  --data_path "$CLEAN_LMDB" \
  --output_path prompt_cache/clean_data_umt5_bf16 \
  --device cuda:0 \
  --batch_size 4
export PROMPT_CACHE=prompt_cache/clean_data_umt5_bf16
```

## 4. 三个正式训练 run

### 4.1 完整 One-Forcing：600 step

```bash
bash experiments/rebuttal/launch_train.sh \
  --config_path experiments/rebuttal/configs/train_1step_one_forcing.yaml \
  --run_name train_1step_one_forcing_600 \
  --seed 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --generator_ckpt "$ODE_CKPT" \
  --teacher_model_path "$TEACHER_DIR" \
  --data_path "$CLEAN_LMDB" \
  --prompt_embedding_cache_path "$PROMPT_CACHE" \
  --max_steps 600 \
  --fake_score_cpu_offload \
  --manual_generator_backward \
  --generator_optimizer_state_cpu_offload \
  --rank0_preload_generator_ckpt
```

启动后立即监控，至少确认到 step 10：

```bash
tmux attach -t of_train_1step_one_forcing_600_s0
tail -f runs/rebuttal/train_1step_one_forcing_600/seed_0/train.log
```

step 10 后按 generator/critic 的真实更新周期估 ETA（该数不含 checkpoint I/O 和评测）：

```bash
python experiments/rebuttal/estimate_training_eta.py \
  --run_dir runs/rebuttal/train_1step_one_forcing_600/seed_0
```

结束后：

```bash
python experiments/rebuttal/check_run.py \
  --run_dir runs/rebuttal/train_1step_one_forcing_600/seed_0 \
  --min_step 600 \
  --expected_seed 0
```

### 4.2 DMD-only：200 step

```bash
bash experiments/rebuttal/launch_train.sh \
  --config_path experiments/rebuttal/configs/train_1step_dmd_only.yaml \
  --run_name train_1step_dmd_only_200 \
  --seed 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --generator_ckpt "$ODE_CKPT" \
  --teacher_model_path "$TEACHER_DIR" \
  --data_path "$CLEAN_LMDB" \
  --prompt_embedding_cache_path "$PROMPT_CACHE" \
  --max_steps 200 \
  --fake_score_cpu_offload \
  --manual_generator_backward \
  --generator_optimizer_state_cpu_offload \
  --rank0_preload_generator_ckpt
```

必须先等 4.1 完全结束和验收。该 run 也要在 tmux 中盯到 step 10。结束后：

```bash
python experiments/rebuttal/check_run.py \
  --run_dir runs/rebuttal/train_1step_dmd_only_200/seed_0 \
  --min_step 200 \
  --expected_seed 0
```

GAN 主比较必须使用：

```text
完整方法: runs/rebuttal/train_1step_one_forcing_600/seed_0/checkpoint_model_000200/model.pt
DMD-only: runs/rebuttal/train_1step_dmd_only_200/seed_0/checkpoint_model_000200/model.pt
```

不能拿完整方法 step 600 对 DMD-only step 200。

### 4.3 4-step One-Forcing：300 step

```bash
bash experiments/rebuttal/launch_train.sh \
  --config_path experiments/rebuttal/configs/train_4step_one_forcing.yaml \
  --run_name train_4step_one_forcing_300 \
  --seed 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --generator_ckpt "$ODE_CKPT" \
  --teacher_model_path "$TEACHER_DIR" \
  --data_path "$CLEAN_LMDB" \
  --prompt_embedding_cache_path "$PROMPT_CACHE" \
  --max_steps 300 \
  --fake_score_cpu_offload \
  --manual_generator_backward \
  --generator_optimizer_state_cpu_offload \
  --rank0_preload_generator_ckpt
```

同样串行运行、tmux 监控到 step 10。结束后：

```bash
python experiments/rebuttal/check_run.py \
  --run_dir runs/rebuttal/train_4step_one_forcing_300/seed_0 \
  --min_step 300 \
  --expected_seed 0
```

当前 `--resume_ckpt` 不恢复 AdamW 状态。正式 run 中断后不能用 weights-only resume 拼接稳定性
曲线；`check_run.py` 默认拒绝这种结果。

## 5. 固定评测 manifests

完整 official prompt 和 manifest：

```bash
python experiments/rebuttal/prepare_vbench_prompts.py \
  --full_info_path "$VBENCH_FULL_INFO" \
  --output_path eval/manifests/vbench_official_prompts.txt
python experiments/rebuttal/make_eval_manifest.py \
  --prompt_path eval/manifests/vbench_official_prompts.txt \
  --output_path eval/manifests/vbench_official_seed0.jsonl \
  --base_seed 0 \
  --num_samples_per_prompt 5 \
  --naming vbench
```

GAN quality 配对使用 7 个直接相关维度：5 个时序/动态维度加 aesthetic/imaging quality。当前
VBench 0.1.5 full-info 中是 326 个 unique prompts：

```bash
python experiments/rebuttal/prepare_vbench_subset.py \
  --full_info_path "$VBENCH_FULL_INFO" \
  --full_prompt_path eval/manifests/vbench_official_prompts.txt \
  --full_manifest_path eval/manifests/vbench_official_seed0.jsonl \
  --output_prompt_path eval/manifests/vbench_gan7_prompts.txt \
  --output_manifest_path eval/manifests/vbench_gan7_seed0.jsonl \
  --dimensions \
    subject_consistency background_consistency temporal_flickering \
    motion_smoothness dynamic_degree aesthetic_quality imaging_quality
```

稳定性曲线使用 5 个时序/动态维度，233 个 unique prompts：

```bash
python experiments/rebuttal/prepare_vbench_subset.py \
  --full_info_path "$VBENCH_FULL_INFO" \
  --full_prompt_path eval/manifests/vbench_official_prompts.txt \
  --full_manifest_path eval/manifests/vbench_official_seed0.jsonl \
  --output_prompt_path eval/manifests/vbench_stability5_prompts.txt \
  --output_manifest_path eval/manifests/vbench_stability5_seed0.jsonl \
  --dimensions \
    subject_consistency background_consistency temporal_flickering \
    motion_smoothness dynamic_degree
```

多样性固定前 100 个 official prompts、每 prompt 4 个生成样本：

```bash
python experiments/rebuttal/make_eval_manifest.py \
  --prompt_path eval/manifests/vbench_official_prompts.txt \
  --output_path eval/manifests/diversity100_seed0.jsonl \
  --base_seed 0 \
  --num_samples_per_prompt 4 \
  --limit 100 \
  --naming vbench
```

所有方法必须读取对应的同一个 JSONL。训练 seed 只有 0；VBench 的 5 个生成样本和 diversity
的 4 个生成 seeds 不是训练 seeds。两个 VBench subset 是从 official manifest 抽取的，因此
同一 prompt 在 full、GAN7、stability5 中保留完全相同的生成 seed；这使 step 200/600 结果可以
无偏复用。

## 6. VBench

先固定 checkpoint：

```bash
export MAIN600=runs/rebuttal/train_1step_one_forcing_600/seed_0/checkpoint_model_000600/model.pt
export MAIN200=runs/rebuttal/train_1step_one_forcing_600/seed_0/checkpoint_model_000200/model.pt
export DMD200=runs/rebuttal/train_1step_dmd_only_200/seed_0/checkpoint_model_000200/model.pt
export OF4_300=runs/rebuttal/train_4step_one_forcing_300/seed_0/checkpoint_model_000300/model.pt
```

`run_vbench_condition.sh` 会先将 manifest 分成 8 个不重叠 shard，全部视频通过帧数/FPS/来源
校验后，再用 8 卡 VBench。正式条件应在 tmux 中逐个串行运行。

### 6.1 Final full official VBench

完整方法 FFE：

```bash
bash experiments/rebuttal/run_vbench_condition.sh \
  --name main_step600_ffe \
  --config_path experiments/rebuttal/configs/eval_ffe.yaml \
  --checkpoint_path "$MAIN600" \
  --schedule ffe \
  --prompt_path eval/manifests/vbench_official_prompts.txt \
  --manifest_path eval/manifests/vbench_official_seed0.jsonl \
  --full_info_path "$VBENCH_FULL_INFO" \
  --output_root eval/final/main_step600_ffe \
  --gpus 0,1,2,3,4,5,6,7 \
  --vbench_python "$VBENCH_PYTHON" \
  --use_ema
```

FFE off：使用同一个 `$MAIN600`，仅把 config/schedule/output/name 换为
`eval_all1.yaml`、`all1`、`eval/final/main_step600_all1`、`main_step600_all1`。

4-step One-Forcing：使用 `$OF4_300`、`eval_all4.yaml`、`all4`，输出
`eval/final/of4_step300_all4`。

4-step Self-Forcing：使用 `$SF4_CKPT`、`eval_all4.yaml`、`all4`，输出
`eval/final/sf4_all4`。双方必须共享 official manifest、帧数和生成 seed。

这四个目录的结果可以报告 official VBench total。不能把下面的 subset score 写成 official total。

### 6.2 GAN clean ablation

完整方法 step 200 和 DMD-only step 200 各运行一次相同的 7-dimension condition：

```bash
bash experiments/rebuttal/run_vbench_condition.sh \
  --name main_step200_ffe_gan7 \
  --config_path experiments/rebuttal/configs/eval_ffe.yaml \
  --checkpoint_path "$MAIN200" \
  --schedule ffe \
  --prompt_path eval/manifests/vbench_gan7_prompts.txt \
  --manifest_path eval/manifests/vbench_gan7_seed0.jsonl \
  --full_info_path "$VBENCH_FULL_INFO" \
  --output_root eval/gan/main_step200_ffe_gan7 \
  --gpus 0,1,2,3,4,5,6,7 \
  --vbench_python "$VBENCH_PYTHON" \
  --dimensions subject_consistency,background_consistency,temporal_flickering,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality \
  --use_ema
```

DMD-only 命令只替换：

```text
name:            dmd_step200_ffe_gan7
checkpoint:      $DMD200
output_root:     eval/gan/dmd_step200_ffe_gan7
```

其余参数禁止改变。

### 6.3 Stability curve

step 200 复用 GAN subset 的主方法结果；step 600 复用 final full 结果，只额外生成 step 100 和
400：

```bash
python experiments/rebuttal/run_checkpoint_sweep.py \
  --run_dir runs/rebuttal/train_1step_one_forcing_600/seed_0 \
  --config_path experiments/rebuttal/configs/eval_ffe.yaml \
  --prompt_path eval/manifests/vbench_stability5_prompts.txt \
  --manifest_path eval/manifests/vbench_stability5_seed0.jsonl \
  --full_info_path "$VBENCH_FULL_INFO" \
  --output_root eval/stability/main_ffe \
  --model_name one_forcing \
  --seed 0 \
  --steps 100 400 \
  --existing_result 200=eval/gan/main_step200_ffe_gan7/vbench/main_step200_ffe_gan7_eval_results.json \
  --existing_result 600=eval/final/main_step600_ffe/vbench/main_step600_ffe_eval_results.json \
  --schedule ffe \
  --gpus 0,1,2,3,4,5,6,7 \
  --dimensions \
    subject_consistency background_consistency temporal_flickering \
    motion_smoothness dynamic_degree \
  --vbench_python "$VBENCH_PYTHON" \
  --use_ema
```

最终曲线是同一个训练 run 的四个 checkpoints；它证明随训练步数的稳定性，不是独立训练 seed
方差，正文必须明确这个区别。

## 7. Diversity、FVD/coverage 和 latency

### 7.1 LPIPS diversity

分别用 `$MAIN200`、`$DMD200`、`$SF1_CKPT` 和同一个
`diversity100_seed0.jsonl` 生成 400 个视频。前两者都用 `ffe`，1-step Self-Forcing 按论文原
设置使用 `all1`；每个条件用：

```bash
python experiments/rebuttal/run_sharded_inference.py \
  --config_path experiments/rebuttal/configs/eval_ffe.yaml \
  --checkpoint_path "$MAIN200" \
  --prompt_path eval/manifests/vbench_official_prompts.txt \
  --manifest_path eval/manifests/diversity100_seed0.jsonl \
  --output_folder eval/diversity/main_step200 \
  --gpus 0,1,2,3,4,5,6,7 \
  --schedule ffe \
  --use_ema
```

替换 checkpoint/config/schedule/output 后生成另两个条件。然后逐个运行：

```bash
python experiments/rebuttal/evaluate_diversity.py \
  --videos_dir eval/diversity/main_step200 \
  --manifest_path eval/manifests/diversity100_seed0.jsonl \
  --output_json eval/diversity/main_step200.json \
  --frames_per_video 8 \
  --metric lpips-vgg \
  --bootstrap_samples 2000
```

LPIPS 越高只表示感知差异越大，必须与 VBench/FVD 一起解释，不能把噪声当作多样性。

### 7.2 FVD、precision、recall、density、coverage

```bash
hf download flateon/FVD-I3D-torchscript i3d_torchscript.pt \
  --local-dir third_party/fvd_i3d
python experiments/rebuttal/decode_lmdb_references.py \
  --lmdb_path "$CLEAN_LMDB" \
  --output_dir eval/fvd/real_seed0 \
  --num_videos 256 \
  --seed 0 \
  --streaming_decode
```

使用脚本产生的匹配 prompt/manifest，分别由 `$MAIN200` 和 `$DMD200` 生成 fake 集：

```text
eval/fvd/real_seed0/reference_prompts.txt
eval/fvd/real_seed0/generation_manifest.jsonl
```

主方法：

```bash
python experiments/rebuttal/run_sharded_inference.py \
  --config_path experiments/rebuttal/configs/eval_ffe.yaml \
  --checkpoint_path "$MAIN200" \
  --prompt_path eval/fvd/real_seed0/reference_prompts.txt \
  --manifest_path eval/fvd/real_seed0/generation_manifest.jsonl \
  --output_folder eval/fvd/main_step200 \
  --gpus 0,1,2,3,4,5,6,7 \
  --schedule ffe \
  --use_ema
```

DMD-only 只替换 checkpoint 和 output folder 为 `$DMD200`、`eval/fvd/dmd_step200`；其余参数
禁止改变。

然后：

```bash
python experiments/rebuttal/evaluate_fvd.py \
  --real_videos_dir eval/fvd/real_seed0 \
  --fake_videos_dir eval/fvd/main_step200 \
  --i3d_path third_party/fvd_i3d/i3d_torchscript.pt \
  --output_json eval/fvd/main_step200.json \
  --real_manifest_path eval/fvd/real_seed0/reference_manifest.jsonl \
  --fake_manifest_path eval/fvd/real_seed0/generation_manifest.jsonl \
  --num_frames 16 \
  --batch_size 4 \
  --min_videos 256 \
  --nearest_k 5
```

对 DMD-only 再跑一次。两个 fake 集必须都严格对应同一 real manifest。

### 7.3 Latency

在同一张 A100、相同 dtype 和 21 latent frames 下分别测 `all1`、FFE、`all4`：

```bash
python experiments/rebuttal/profile_latency.py \
  --config_path experiments/rebuttal/configs/eval_ffe.yaml \
  --checkpoint_path "$MAIN600" \
  --prompt_path prompts/smoke_one.txt \
  --output_json eval/latency/main600_ffe.json \
  --num_output_frames 21 \
  --warmup 3 \
  --trials 20 \
  --seed 0 \
  --use_ema \
  --include_vae
```

输出必须报告 first block、steady block、diffusion、VAE、total、NFE/context updates、GPU 型号和
trial 方差，不能只报吞吐。

## 8. 最终验收

- [ ] 完整方法 seed 0 到 600；DMD-only seed 0 到 200；4-step seed 0 到 300。
- [ ] 三个 run 都由 tmux 启动，并人工检查到至少 step 10。
- [ ] GAN 对比是 `$MAIN200` 对 `$DMD200`，相同 FFE 和 manifest。
- [ ] FFE on/off 都是 `$MAIN600`。
- [ ] 4-step 双方都是 `all4`，并有真实 Self-Forcing checkpoint。
- [ ] stability 是 step 100/200/400/600 的同一 run，未冒充跨 seed 方差。
- [ ] official VBench 每个 prompt 恰有 5 个样本；subset 没有被称为 official total。
- [ ] diversity 每 prompt 4 samples；FVD/coverage 使用 256 个匹配 real/fake。
- [ ] checklist 改为 single-seed，并正确披露 3 名人类 annotators。
- [ ] long-video 和 curvature causal intervention 明确标为本轮未完成。
- [ ] 未运行的结果没有写成已完成。

## 9. 已完成的代码 smoke

以下仅证明路径跑通，不是论文正式结果：

- 完整 1-step、DMD-only、4-step 都已在 4×RTX A6000 完整配置跑过 10 step；
- DMD-only 的 GAN/R1/R2 项严格为 0；
- 发布 checkpoint 的逐 block FSDP key 已规范化后以 `strict=True` 加载；`all1`、FFE、`all4`
  均实际导出并校验视频；
- latency、4-sample LPIPS、真实 LMDB 解码、I3D-FVD/P/R/density/coverage、VBench
  custom-input 接口均做过 smoke；
- standard VBench 会拒绝少于 5 samples、NaN、空结果和不完整目录；
- 正式大规模训练、official VBench、256-sample FVD 和外部 Self-Forcing 对照仍需在目标机器运行。
