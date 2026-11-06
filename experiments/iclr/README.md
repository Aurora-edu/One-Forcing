# ICLR 2027：受控稳定性与轨迹实验

分支 `ICLR` 基于 `origin/review` (`bfd5e2d`)。此目录实现这次审稿意见对应的实验；旧 `experiments/rebuttal` 的其他任务不会自动启动。

用户已确认：**本机完成 CPU 验证，真实 GPU 训练、FSDP 恢复与视频检查在训练机完成。** 本机记录见 [ICLR_LOCAL_VALIDATION.md](../../docs/ICLR_LOCAL_VALIDATION.md)。CPU toy 测试不是 Wan 训练结果。

## 1. 迁移与资产准备

```bash
git clone --branch ICLR https://github.com/Aurora-edu/One-Forcing.git
cd One-Forcing
conda create -n oneforcing_iclr python=3.10
conda activate oneforcing_iclr
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
python -m pip install pytest
python setup.py develop
python -m pytest -q
```

VBench 使用单独环境，安装 `requirements-vbench.txt`；不要与训练环境混装不同版本的 transformers。记录训练机的 `pip freeze`、GPU 型号和 Git commit，所有实验组使用同一环境、同一组 8 张卡。

按实际路径设置以下变量；`ODE_INIT` 必须是有来源记录的**同一份 raw ODE 初始化**，不能以文件名推断身份，也不能用来源未确认的 `/data/fengjiaqi/model.pt` 替代：

```bash
export OF_PY=/path/to/oneforcing_iclr/bin/python
export VB_PY=/path/to/vbench/bin/python
export ICLR_ROOT=/data/fengjiaqi/iclr2027
export ODE_INIT=/data/fengjiaqi/checkpoints/verified_causal_ode.pt
export TEACHER=/data/fengjiaqi/wan_models/Wan2.1-T2V-14B
export DATA=/data/fengjiaqi/One-Forcing/mixkit_latents_lmdb
```

仓库代码仍从相对路径 `wan_models/Wan2.1-T2V-1.3B` 加载 generator/fake-score、T5、tokenizer、VAE。将完整模型目录放到该位置，或在仓库尚无 `wan_models` 时建立软链接：

```bash
ln -s /data/fengjiaqi/wan_models wan_models
```

不要覆盖已有目录或软链接。当前机器的 `/data/fengjiaqi/Causal_forcing` 是断链，不用作默认路径。数据 LMDB 应有 `data.mdb`，shape 为 `[N,21,16,60,104]`。

所有模型/数据/输出均不进入 Git。完整恢复 checkpoint 包含两个模型和 AdamW 状态，每个约数十 GB；12 组 × 5 时间点应预留约 2 TB 量级空间，视频另计。迁移代码不等于迁移权重和 LMDB。

## 2. 四组 × 三训练种子的稳定性对照

| 运行前缀 | 每 block latent 帧数 | R1/R2 权重 | seeds |
|---|---:|---|---|
| `frame_roff` | 1 | 0 / 0 | 0,1,2 |
| `frame_ron` | 1 | 1 / 0.001 | 0,1,2 |
| `chunk_roff` | 3 | 0 / 0 | 0,1,2 |
| `chunk_ron` | 3 | 1 / 0.001 | 0,1,2 |

开启正则时保留旧 chunkwise 的定义：`r1_loss_type=apt, r1_sigma=0.1, r2_sigma=0.01`。不要仅复制权重而换成不同归一化的 R1。

所有组统一 `[21,29]` 判别器特征层、2 registers、2048 FFN、21 个梯度帧、21 个 score 帧、raw 初始化、优化器、batch、数据顺序算法、G:D=1:5。所有运行的判别器初始化 seed 固定为 12345；训练 seed 0/1/2 控制数据顺序和训练随机性。数据实际采样量仍会随单位时间完成的 updates 不同而变化。

本实验关闭 EMA、训练和主评测均为 all1；它是新的受控对照，不是论文不同超参数下 83.76 分结果的直接复现。即使初始权重完全相同，改变 attention block 粒度也可能导致 step-0 输出不同，因此保留初始评测点。

先生成四组 smoke 配置（生成器同时输出全部 12 个组合）：

```bash
"$OF_PY" -m experiments.iclr.make_configs \
  --generator-ckpt "$ODE_INIT" --teacher "$TEACHER" --data "$DATA" \
  --budget-seconds 7200 --smoke --output "$ICLR_ROOT/configs_smoke"

# 完整资产/依赖版本检查；本机共享环境的 numpy/transformers 与锁定版本不同，
# 正式训练使用新环境，不能通过放宽检查把差异隐藏掉。
"$OF_PY" experiments/rebuttal/preflight.py \
  --config_path "$ICLR_ROOT/configs_smoke/frame_roff_s0.yaml" \
  --generator_ckpt "$ODE_INIT" --teacher_model_path "$TEACHER" \
  --data_path "$DATA" --gpus 0,1,2,3,4,5,6,7 --seed 0 --max_steps 10

# 不带 --execute 只打印命令；带上后自动创建独立 tmux。
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$ICLR_ROOT/configs_smoke/matrix.json" \
  --output "$ICLR_ROOT/smoke" --run frame_roff_s0 \
  --session iclr_smoke_frame_roff --execute
```

smoke 严格跑到 10 iterations（2 次 G update、10 次 critic update），即使时间预算很短也不会提前正常结束。分别检查 `frame_roff_s0`、`frame_ron_s0`、`chunk_roff_s0`、`chunk_ron_s0`，每次使用不同 session 名，顺序执行。没有结束前不要启动下一组。

```bash
tmux capture-pane -pt iclr_smoke_frame_roff -S -100
tail -n 10 "$ICLR_ROOT/smoke/frame_roff_s0/metrics.jsonl"
cat "$ICLR_ROOT/smoke/frame_roff_s0/training.done"
```

至少观察到第 10 步，确认 losses/grad norms 有限；接着生成并实际播放视频，检查花屏、结构崩溃、闪烁和冻结。`training.done` 只代表训练代码结束，不代表视觉质量合格。GPU 忙时启动器会拒绝执行，不会停止其他会话。

用 smoke/pilot 的耗时预先选定正式预算 T，例如让最慢组覆盖约 750 iterations；示例 7200 秒仅为占位，**不是已经测得的合适预算**。估算时包含每五步一次 G update 的完整周期，不能只取 critic-only 步的速度。

```bash
"$OF_PY" -m experiments.iclr.make_configs \
  --generator-ckpt "$ODE_INIT" --teacher "$TEACHER" --data "$DATA" \
  --budget-seconds 7200 --output "$ICLR_ROOT/configs_formal"
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$ICLR_ROOT/configs_formal/matrix.json" \
  --output "$ICLR_ROOT/runs" --run all --session iclr_formal --dry-run
# 核对后同一命令将 --dry-run 改为 --execute。
```

正式训练按累计同步训练时间停止，没有 200/750 的 step cap。计时包含取 batch、前后向、参数更新和必要同步，取各 rank 的最大耗时；模型初始化、checkpoint I/O、日志/GC 分别记录，不混入训练时间。于 0/.25/.5/.75/1.0 T 后第一个 iteration 边界保存，记录实际超出时间；不假装精确落在 T。全程不在线运行 VBench。每个新运行仍应观察头 10 步。

每组输出：`resolved_config.yaml`、`source.json`、`initialization.json`、`metrics.jsonl`、`training_clock.jsonl`、`checkpoints.jsonl`、完整 checkpoint 目录和 `training.done`。非有限损失/梯度在所有 ranks 同步失败，并输出 `nonfinite_failure.json`；不要把失败 seed 从报告中删掉。

```bash
"$OF_PY" -m experiments.iclr.audit_initialization --runs "$ICLR_ROOT/runs"
```

该检查验证实际加载的 generator/critic tensor hash 相同，不证明其 ODE 训练来源。ODE 来源仍需原训练记录。

## 3. 完整恢复与迁移约束

checkpoint 目录中的 `model.pt` 用于评测；恢复还必须有 `complete.json` 和 8 个 `rank_*.pt`（各 rank 的 optimizer、Python/NumPy/Torch/CUDA RNG、数据 epoch/batch 位置）。旧 review 分支的仅权重 checkpoint 不支持无缝恢复。

```bash
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$ICLR_ROOT/configs_formal/matrix.json" \
  --output "$ICLR_ROOT/resumed_runs" --run frame_roff_s0 \
  --resume "$ICLR_ROOT/runs/frame_roff_s0/checkpoint_model_000123" \
  --session iclr_resume_frame --execute
```

要求相同 config、GPU 类型、world size、FSDP topology 和软件环境。只更换输出路径可以恢复；改变训练预算/超参数应另开实验。checkpoint 的原子完成标记最后写入；中断留下的半写目录不会被覆盖。推荐从最近完整 checkpoint 恢复到新输出根目录。

新目录的 `checkpoints.jsonl` 仅包含恢复后保存的点。最终做曲线时，还需评测原运行的有效历史点；合并评测目录时每个 seed/timepoint 只保留一次。不能将中断期间的无 checkpoint 指标与重跑后的点重复计入。

CPU 测试只替换了 CUDA/FSDP collectives，用真实训练循环验证优化器/RNG/数据续接。训练机还要专门做一次真实 FSDP checkpoint 保存/恢复 smoke，确认加载及后续更新正常，才启动长实验。

## 4. 视频检查、固定评测与时间曲线

先在训练机导出少量视频并实际播放。示例使用 smoke 的 step-10 checkpoint：

```bash
PYTHON_BIN="$OF_PY" bash scripts/infer.sh \
  --method framewise --schedule all1 \
  --config_path "$ICLR_ROOT/configs_smoke/frame_roff_s0.yaml" \
  --checkpoint_path "$ICLR_ROOT/smoke/frame_roff_s0/checkpoint_model_000010/model.pt" \
  --prompt_path prompts/demos.txt --limit 4 --num_samples_per_prompt 1 \
  --output_folder "$ICLR_ROOT/visual_smoke/frame_roff_s0" --gpu_id 0
```

四组用相同 prompts 和生成 seed，chunk 组改 `--method chunkwise`。可用本地播放器播放或将 MP4 下载播放，记录逐视频的检查结论；指标和静态拼图不能代替动态画面检查。不要在训练占卡时重叠启动导出。

固定原始 prompt 清单与每 prompt 一个生成样本（不是五样本 leaderboard 协议）：

```bash
INFO=experiments/rebuttal/assets/qwen_vbench/VBench_full_info.json
"$OF_PY" experiments/rebuttal/prepare_vbench_prompts.py \
  --full_info_path "$INFO" --output_path "$ICLR_ROOT/manifests/original.txt"
"$OF_PY" experiments/rebuttal/make_eval_manifest.py \
  --prompt_path "$ICLR_ROOT/manifests/original.txt" \
  --output_path "$ICLR_ROOT/manifests/original_seed0.jsonl" \
  --base_seed 0 --num_samples_per_prompt 1
```

这里目录虽名为 qwen_vbench，其 `VBench_full_info.json` 的 `prompt_en` 是原始评测 prompts；不要传入旁边的 rewritten pairs。

```bash
"$OF_PY" -m experiments.iclr.evaluate curve \
  --index "$ICLR_ROOT/runs/frame_roff_s0/checkpoints.jsonl" \
  --config "$ICLR_ROOT/configs_formal/frame_roff_s0.yaml" \
  --prompts "$ICLR_ROOT/manifests/original.txt" \
  --manifest "$ICLR_ROOT/manifests/original_seed0.jsonl" --full-info "$INFO" \
  --output "$ICLR_ROOT/eval/frame_roff_s0" --vbench-python "$VB_PY" --dry-run
# 核对后改 --execute；逐一对 12 个 run 执行，不能并发抢占同一批卡。
"$OF_PY" -m experiments.iclr.summarize \
  --evaluations "$ICLR_ROOT/eval" --output "$ICLR_ROOT/summary"
```

评测从 checkpoint index 读取真实 framewise/chunkwise 模式，统一 raw/all1，禁用 FFE/EMA。汇总输出 `points.csv`、`matched_time_summary.csv`、`endpoints.csv`、`curves.pdf/png`；图展示各 seed 的真实训练时间，表报告 3 个训练 seeds 的均值、样本 SD 和实际时间范围。少组、少 seed、少时间点或评测协议不一致会报错。若某 seed 发散，先报告失败及已有曲线，不要补选“好 seed”来通过完整表检查。

## 5. Original/rectified CD：一步与四步

优先用论文原始两个 300-update checkpoint，确认其初始化、训练长度、数据和 raw 权重来源一致：

```bash
"$OF_PY" -m experiments.iclr.evaluate cd \
  --original /data/fengjiaqi/checkpoints/original_cd_300/model.pt \
  --rectified /data/fengjiaqi/checkpoints/rectified_cd_300/model.pt \
  --prompts "$ICLR_ROOT/manifests/original.txt" \
  --manifest "$ICLR_ROOT/manifests/original_seed0.jsonl" --full-info "$INFO" \
  --output "$ICLR_ROOT/cd_eval/step300" --vbench-python "$VB_PY" --dry-run
```

核对后加 `--execute`；生成四个条件：original/rectified × all1/all4，所有 block 都使用对应步数，不开 FFE。四份完整 16 维 VBench JSON 位于各 cell 的 `vbench/` 下。比较 total/quality/semantic 及 dynamic degree 等维度，实际播放配对视频。单一训练 seed 的结果不能解释为训练稳定性的置信区间。

```bash
"$OF_PY" -m experiments.iclr.summarize_cd \
  --evaluations "$ICLR_ROOT/cd_eval/step300" --output "$ICLR_ROOT/cd_summary/step300"
```

输出 `cd_scores.csv`、`cd_gains.csv` 和 JSON：分别列出四个条件及一步/四步的 rectified-minus-original 差值。总分使用 0–1 单位，论文百分制表格乘 100；这是配对聚合点估计，不是训练 seed 的误差条。

需要新训练预算曲线时：

```bash
"$OF_PY" -m experiments.iclr.cd \
  --data /data/fengjiaqi/paired_cd_trajectory_lmdb \
  --generator-ckpt "$ODE_INIT" --steps 900 --seed 0 \
  --output "$ICLR_ROOT/cd_train/seed0" --session iclr_cd_seed0 --dry-run
```

此工具复用 review 的 adjacent-consistency trainer，**从相同初始化重新训练到 900，并保存 300/600/900**，不假装从旧 300 checkpoint 恢复优化器/EMA target。默认沿用旧 small-CD 的 9 latent frames；要复现论文必须先核对原 config，必要时通过 `--config` 传入已确认配置。`target_ema_decay` 是训练目标网络 EMA，评测仍只用 raw generator。执行模式自动用 tmux，curved/rectified 顺序训练，各自都要观察到第 10 步。若复跑 seeds 1、2，使用独立输出目录和 session。

CD 输入是含 `curvature_dataset_manifest.json` 的稀疏轨迹 LMDB；准备/审计流程见 [旧 CD 说明](../rebuttal/CURVATURE_CD_SMALL.md)。下面的双向 teacher 完整轨迹不能直接冒充该 clean-conditioned CD 数据。

## 6. 完整 teacher 轨迹与参数化分析

`collect_teacher` 使用 **bidirectional Wan2.1-T2V-1.3B**、CFG=6、shift=8，保存完整 50/200 步 Euler 轨迹与真实 timestep，不传入 clean video。它不同于旧 causal ODE 脚本保存的少数中间状态。默认 negative prompt 与仓库 `config.yaml` 一致，metadata 明确记录求解器；原论文 solver 未核实，不能声称精确复现其轨迹。

预先选定含多种场景/运动的 100 个 prompts，固定顺序；200-step 分辨率复核用同一文件的前 20 个。以下在训练机的独立 tmux 中运行：

```bash
tmux new-session -s iclr_teacher
# 在该 tmux 中执行（先不带 --execute 检查配置）：
"$OF_PY" -m experiments.iclr.collect_teacher \
  --prompts "$ICLR_ROOT/manifests/teacher_prompts100.txt" \
  --model-path /data/fengjiaqi/wan_models/Wan2.1-T2V-1.3B \
  --output "$ICLR_ROOT/teacher50" --steps 50 --limit 100 --gpu 0 --execute
# 第二次改 --steps 200 --limit 20 --output "$ICLR_ROOT/teacher200"。
```

可用 `--num-shards 8 --shard-index 0..7 --gpu 0..7` 在各卡上采不同 prompts，所有 shards 共享同一 output root；同一轮必须使用相同参数、prompt 文件和 num-shards。种子为 base seed + 原 prompt index，不随分片改变。每个 prompt 立即将状态保存到 CPU mmap，成功发布 `.pt` 后只删除自己创建的临时 backing 文件。

```bash
"$OF_PY" -m experiments.iclr.trajectory \
  --trajectories "$ICLR_ROOT/teacher50" --output "$ICLR_ROOT/geometry50"
"$OF_PY" -m experiments.iclr.trajectory \
  --trajectories "$ICLR_ROOT/teacher200" --output "$ICLR_ROOT/geometry200"
```

输出 per-trajectory JSONL、CSV 与 bootstrap 汇总。分析同时覆盖 latent 和 temporal-difference latent；比较 flow time、t² 和弧长坐标，固定物理噪声区域 t≥0.9，按非均匀区间宽度积分，跨阈值段裁切。几何指标为路径/弦长、到弦线的正交距离和转角；零运动、闭合路径、重复点不会被任意 epsilon 伪造为有效比例。底层 loader 会排除代表 dataset clean latent 的末尾 null；正式 CLI 则要求完整双向 teacher 来源、权重 hash 和一致采样协议，拒绝把稀疏 CD 数据混入结果。

合成单元测试若走 CLI，必须显式将输入标为 `trajectory_kind=synthetic_test` 并传 `--synthetic-check`；输出会标为 `synthetic_validation`，不能作为论文 teacher 证据。

50/200 对比应只比较共同的 20 个 prompts；对 50-step 目录加 `--limit 20` 另出一份结果。重新生成更密轨迹与重标记同一条路径的时间是两种独立检查。高噪声局部 rectification 干预未纳入这次必做实现；现有实验不能独自建立“高噪声弯曲导致一步失败”的因果结论。

## 7. 验证边界

```bash
"$OF_PY" -m pytest -q
git diff --check
```

CPU 测试覆盖：配置混杂项、相同初始化审计、10-step 更新计数、真实循环的保存/恢复（tiny CPU models）、损失非有限失败、数据跨 epoch 续接、纯一步/四步评测命令、原始 prompts、teacher Euler 日程、非均匀积分、直线/变速直线/重复点/闭合路径、完整多 seed 时间汇总和图表导出。所有 synthetic fixture 均明确属于测试，不进入论文结果。

训练机验收仍需完成：四种设置各 10 步、实际 FSDP 保存/恢复、原始/rectified CD 模型加载与采样、teacher 全路径采集，以及真实 MP4 动态画面检查。
