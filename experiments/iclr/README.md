# ICLR 2027：受控稳定性与轨迹实验

分支 `ICLR` 基于 `origin/review` (`bfd5e2d`)，但实验 A 的科学设置以 **`main@c9a2350`** 为准。用户确认主表使用 EMA；对齐依据、自然联动和实现边界见 [ICLR_MAIN_ALIGNMENT.md](../../docs/ICLR_MAIN_ALIGNMENT.md)。旧 `experiments/rebuttal` 的其他任务不会自动启动。

训练机 Codex 的完整执行任务见 [ICLR_TRAINING_MACHINE_TASK.md](../../docs/ICLR_TRAINING_MACHINE_TASK.md)。先读任务说明，再按本手册执行命令。

三位审稿人的问题合并和新增评测见 [ICLR_REVIEWER_COVERAGE.md](../../docs/ICLR_REVIEWER_COVERAGE.md)：nPEg 的受控训练、CD 和几何分析优先；随后做主表协议核对与 original/Qwen 评测。训练总数为 6 个：4 个 frame/chunk run 与 2 个 CD run；A 的 frame 两个 seeds 同时提供标准配置第 200 步 EMA 重复性，不另加训练。

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
export ICLR_ROOT=/data/fengjiaqi/iclr2027_main_200
export ODE_INIT=/data/fengjiaqi/checkpoints/verified_causal_ode.pt
export TEACHER=/data/fengjiaqi/wan_models/Wan2.1-T2V-14B
export DATA=/data/fengjiaqi/One-Forcing/mixkit_latents_lmdb
```

仓库代码仍从相对路径 `wan_models/Wan2.1-T2V-1.3B` 加载 generator/fake-score、T5、tokenizer、VAE。将完整模型目录放到该位置，或在仓库尚无 `wan_models` 时建立软链接：

```bash
ln -s /data/fengjiaqi/wan_models wan_models
```

不要覆盖已有目录或软链接。当前机器的 `/data/fengjiaqi/Causal_forcing` 是断链，不用作默认路径。数据 LMDB 应有 `data.mdb`，shape 为 `[N,21,16,60,104]`。

所有模型/数据/输出均不进入 Git。完整恢复 checkpoint 包含 raw/EMA 模型和 AdamW 状态；按 4 runs × 5 个保存点的实测大小预留空间，另计 smoke、CD、轨迹和视频。迁移代码不等于迁移权重和 LMDB。

## 2. 四个 run：关闭正则的粒度对照与两 seed 重复性

| 运行顺序与名称 | 每 block latent 帧数 | R1/R2 权重 | training seed |
|---|---:|---|---|
| 1. `frame_roff_s0` | 1 | 0 / 0 | 0 |
| 2. `chunk_roff_s0` | 3 | 0 / 0 | 0 |
| 3. `frame_roff_s1` | 1 | 0 / 0 | 1 |
| 4. `chunk_roff_s1` | 3 | 0 / 0 | 1 |

R1/R2 全部关闭。分别在 seeds 0、1 内比较 frame/chunk，保留两组配对差值，再汇总差值的均值和样本 SD。这不是正则开/关消融；每种粒度 n=2，只提供小样本重复性证据。

`base.yaml` 继承逐字保存的 main `config.yaml`。所有组统一 `gradient_num_blocks=17`、21 个训练/score 帧、EMA=0.99（第 50 步开始）、判别器 layers [21,29]、2 registers、FFN 2048、raw 初始化、优化器、batch、G:D=1:5。同 seed 内只改 `num_frame_per_block` 1→3；按 main 的 block 计数规则，实际梯度窗口为 frame 17 帧 / chunk 21 帧，明确报告该联动。同 seed 的判别器初始化应一致；跨 seed 保留 main 的模型随机初始化方式。sampler 使用 main 默认 seed 0，两种粒度按相同更新次数取数据，实际耗时分别记录。

旧的 3-run / 12-run `matrix.json` 会被拒绝。请在新的 `ICLR_ROOT` 下重新生成配置，保留旧输出；`--run all` 只运行表中四项，CD 两组通过第 5 节的单独入口执行，不追加其他 seed 或正则设置。

旧的 all-21 梯度窗口、EMA 关闭的训练不能复用为本次 main 对照；新目录必须重新生成配置。代码校验 `main_block_ablation_200_v2` profile 和参考超参数，避免混用旧产物。

训练 all1；主评测使用 EMA、`main_ffe`、pinned Qwen、944×5 samples。`main_ffe` 保持固定块，只给第一块四步，后续一步；不要用 review 的 first4then1 `ffe`。第 0 步尚无 EMA，单独标记为 raw 初始化参考；所有后续主评测点必须包含 EMA。改变 block 粒度可能使 step-0 输出不同，因此保留此参考点。

先生成 smoke 配置（生成器输出表中 4 个组合，先分别验证 seed 0 的两种粒度）：

```bash
"$OF_PY" -m experiments.iclr.make_configs \
  --generator-ckpt "$ODE_INIT" --teacher "$TEACHER" --data "$DATA" \
  --smoke --output "$ICLR_ROOT/configs_smoke"

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

smoke 严格跑到 10 iterations（2 次 G update、10 次 critic update），保存第 0、5、10 步。先分别用 `--run frame_roff_s0` 和 `--run chunk_roff_s0` 检查两种粒度，每次使用不同 session 名，顺序执行。没有结束前不要启动下一组。两次 smoke 不计入 4 个正式 frame/chunk run；正式 frame/chunk seed 1 均须观察前 10 步。

```bash
tmux capture-pane -pt iclr_smoke_frame_roff -S -100
tail -n 10 "$ICLR_ROOT/smoke/frame_roff_s0/metrics.jsonl"
cat "$ICLR_ROOT/smoke/frame_roff_s0/training.done"
```

至少观察到第 10 步，确认 losses/grad norms 有限；接着生成并实际播放视频，检查花屏、结构崩溃、闪烁和冻结。`training.done` 只代表训练代码结束，不代表视觉质量合格。GPU 忙时启动器会拒绝执行，不会停止其他会话。

正式 frame/chunk 四组全部固定 **200 iterations**，每组 40 次 G update、200 次 critic update。smoke/pilot 只用于估算耗时和验证实现，不决定停止预算。估算需包含每五步一次 G update 的完整周期，不能只取 critic-only 步速度。旧时间预算配置不再适用，请使用新的 `ICLR_ROOT` 重新生成。

```bash
"$OF_PY" -m experiments.iclr.make_configs \
  --generator-ckpt "$ODE_INIT" --teacher "$TEACHER" --data "$DATA" \
  --output "$ICLR_ROOT/configs_formal"
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$ICLR_ROOT/configs_formal/matrix.json" \
  --output "$ICLR_ROOT/runs" --run all --session iclr_formal --dry-run
# 核对后同一命令将 --dry-run 改为 --execute。
```

正式训练在第 200 步结束，保存 **0、50、100、150、200** 步，不因实际耗时较短而延长。计时包含取 batch、前后向、参数更新、EMA 和必要同步，取各 rank 最大耗时；初始化、checkpoint I/O、日志/GC 分列。曲线按相同步数比较，实际训练耗时作为成本单独报告；不称为严格的 matched-wall-clock 对照。全程不在线运行 VBench，每个新运行仍应观察头 10 步。

每组输出：`resolved_config.yaml`、`source.json`、`initialization.json`、`metrics.jsonl`、`training_clock.jsonl`、`checkpoints.jsonl`、完整 checkpoint 目录和 `training.done`。非有限损失/梯度在所有 ranks 同步失败，并输出 `nonfinite_failure.json`；不要把失败 seed 从报告中删掉。

```bash
"$OF_PY" -m experiments.iclr.audit_initialization --runs "$ICLR_ROOT/runs"
```

该检查验证四组 generator hash 相同、同 seed 的 frame/chunk critic hash 相同，不要求跨 seed 的 critic 相同，也不证明 ODE 训练来源。ODE 来源仍需原训练记录。

## 3. 完整恢复与迁移约束

checkpoint 目录中的 `model.pt` 保存 raw generator、critic，以及启用后的 `generator_ema`；恢复还必须有 `complete.json` 和 8 个 `rank_*.pt`（各 rank 的 optimizer、EMA、Python/NumPy/Torch/CUDA RNG、数据 epoch/batch）。当前 schema=3；旧 schema=1/2 或仅权重 checkpoint 不支持无缝恢复。

```bash
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$ICLR_ROOT/configs_formal/matrix.json" \
  --output "$ICLR_ROOT/resumed_runs" --run frame_roff_s0 \
  --resume "$ICLR_ROOT/runs/frame_roff_s0/checkpoint_model_000150" \
  --session iclr_resume_frame --execute
```

要求相同 config、GPU 类型、world size、FSDP topology 和软件环境。只更换输出路径可以恢复；改变训练预算/超参数应另开实验。checkpoint 的原子完成标记最后写入；中断留下的半写目录不会被覆盖。推荐从最近完整 checkpoint 恢复到新输出根目录。

新目录的 `checkpoints.jsonl` 仅包含恢复后保存的点。最终做曲线时，还需评测原运行的有效历史点；合并评测目录时每个 seed/step 只保留一次。不能将中断期间的无 checkpoint 指标与重跑后的点重复计入。

CPU 测试替换 CUDA/FSDP collectives，用真实训练循环验证优化器/EMA/RNG/数据续接。训练机需验证真实 FSDP 保存、恢复和后续更新；10 步 smoke 尚无 EMA，另需覆盖第 50 步后的 EMA 恢复，可在正式运行中按计划完成。

## 4. 视频检查、固定评测与训练步数曲线

先在训练机导出少量视频并实际播放。示例使用 smoke 的 step-10 checkpoint：

```bash
PYTHON_BIN="$OF_PY" bash scripts/infer.sh \
  --method framewise --schedule main_ffe \
  --config_path "$ICLR_ROOT/configs_smoke/frame_roff_s0.yaml" \
  --checkpoint_path "$ICLR_ROOT/smoke/frame_roff_s0/checkpoint_model_000010/model.pt" \
  --prompt_path prompts/demos.txt --limit 4 --num_samples_per_prompt 1 \
  --output_folder "$ICLR_ROOT/visual_smoke/frame_roff_s0" --gpu_id 0
```

四个 run 用相同 prompts 和生成 seed，chunk 组改 `--method chunkwise`。可用本地播放器播放或将 MP4 下载播放，记录逐视频的检查结论；指标和静态拼图不能代替动态画面检查。不要在训练占卡时重叠启动导出。

主评测先固定 Qwen rewrites 与五样本 manifest；另准备原始 prompts 的单样本 manifest 供 CD 使用：

```bash
INFO=experiments/rebuttal/assets/qwen_vbench/VBench_full_info.json
"$OF_PY" experiments/rebuttal/paper_vbench_protocol.py prepare \
  --output_dir "$ICLR_ROOT/manifests/main" --samples_per_prompt 5
"$OF_PY" experiments/rebuttal/prepare_vbench_prompts.py \
  --full_info_path "$INFO" --output_path "$ICLR_ROOT/manifests/original.txt"
"$OF_PY" experiments/rebuttal/make_eval_manifest.py \
  --prompt_path "$ICLR_ROOT/manifests/original.txt" \
  --output_path "$ICLR_ROOT/manifests/original_seed0.jsonl" \
  --base_seed 0 --num_samples_per_prompt 1
```

`VBench_full_info.json` 的 `prompt_en` 是原始评分 prompts；主评测只将 rewrites 用作模型 conditioning。CD 使用 original.txt 和 original_seed0.jsonl，不使用 rewrites。

```bash
"$OF_PY" -m experiments.iclr.evaluate curve \
  --index "$ICLR_ROOT/runs/frame_roff_s0/checkpoints.jsonl" \
  --config "$ICLR_ROOT/configs_formal/frame_roff_s0.yaml" \
  --prompts "$ICLR_ROOT/manifests/main/vbench_prompts.txt" \
  --extended-prompts "$ICLR_ROOT/manifests/main/vbench_qwen_rewrites.txt" \
  --manifest "$ICLR_ROOT/manifests/main/vbench_qwen_5sample_seed0.jsonl" \
  --samples-per-prompt 5 --full-info "$INFO" \
  --output "$ICLR_ROOT/eval_steps/frame_roff_s0" --vbench-python "$VB_PY" --dry-run
# 核对后改 --execute；逐一对 4 个 run 执行，不能并发抢占同一批卡。
"$OF_PY" -m experiments.iclr.summarize \
  --evaluations "$ICLR_ROOT/eval_steps" --output "$ICLR_ROOT/summary_steps"
```

评测从 index 读取真实粒度，固定 EMA/main_ffe/Qwen×5，第 0 步是明确标注的 raw 参考。汇总输出 `points.csv`、`matched_step_summary.csv`、`endpoints.csv`、`same_seed_comparison.csv`、`paired_step_summary.csv`、`step200_seed_summary.csv`、`curves.pdf/png`。图展示 4 个 run 在 0/50/100/150/200 步的质量；frame/chunk 各 n=2，报告均值、样本 SD、实际耗时范围和两组 seed 内配对差值。

完整曲线汇总已经包含第 200 步各 seed 的 EMA 分数，不必重复生成视频或评分。若只需要终点而不做完整曲线，可给 `evaluate curve` 加 `--reference-step 200`，输出到独立 `$ICLR_ROOT/eval_step200/<run>`；然后用 `summarize --reference-step 200` 汇总。不要把这套可选终点评测与完整曲线目录混合。

`step200_seed_summary.csv` 给出标准训练设置下的两 seed 单次分数、均值和样本 SD。原主表完整 rewrites/随机流未归档，不保证新评测恰好得到 83.76。

缺少指定 run、额外 seed、正则开启、步数点覆盖或评测协议不一致会报错。若某 seed 发散，先报告失败及已有曲线，不要补选“好 seed”来通过完整表检查。两个 seeds 的样本量仍有限，不据此宣称统计显著或普遍稳定。

## 5. 重训 original/rectified CD，再评测一步与四步

原实验流程、历史一步/四步结果及公开资产入口见 [CD 来源与重建说明](../../docs/ICLR_CD_PROVENANCE.md)。历史配置的 training seed 是 2026，本节 ICLR 命令使用 seed 0，因此是新的受控重跑；不能用本节默认值声称原实验完全同设置。

用户已确认旧 CD 权重找不到，本轮必须重新训练两个 arm：original（代码名 `curved`）与 `rectified`。两组 training seed 均为 0，每组 300 updates；与第 2 节的 4 个 run 合计 **6 个正式训练 run**。all1/all4 共 4 个评测条件，不增加训练次数。

重新训练仍需要配对 ODE 轨迹 LMDB 和生成这份数据的共同 raw 初始化。`CD_INIT` 是这份初始化，不是丢失的 CD 成品权重；它可以与实验 A 的 `ODE_INIT` 相同，但必须先核对数据 manifest。不要因找不到旧 CD 权重而跳过本实验，也不能用来源未知的 checkpoint 替代初始化。

```bash
export CD_DATA=/data/fengjiaqi/paired_cd_trajectory_lmdb
export CD_INIT=/data/fengjiaqi/checkpoints/verified_cd_ode_source.pt
export CD_CONFIG=experiments/rebuttal/configs/train_curvature_cd_small.yaml
export CD_OUT="$ICLR_ROOT/cd_train/seed0"

"$OF_PY" -m experiments.iclr.cd \
  --config "$CD_CONFIG" --data "$CD_DATA" \
  --generator-ckpt "$CD_INIT" --steps 300 --seed 0 \
  --output "$CD_OUT" --session iclr_cd_seed0 --dry-run
# 核对两条命令后，将 --dry-run 改成 --execute；自动创建 tmux 并顺序训练两组。
```

该工具复用 review 的 adjacent-consistency trainer，默认已改为 **300 updates/arm**。共享初始化、数据顺序、prompts、初始噪声、ODE 端点、timesteps 和 clean conditioning，只改变轨迹干预。两组各自观察到第 10 步，保存训练日志、配置、数据 manifest 和 sample-order digest；固定使用 step-300，不能看评测结果后挑 checkpoint。

默认 `CD_CONFIG` 是旧 small-CD 的 **9 latent frames** 配置，先核对原始实验设置；若不能确认与论文一致，就明确标为新的受控重跑。两个 arm 使用同一配置。`target_ema_decay` 是训练目标网络 EMA，评测仅使用 raw generator；当前 trainer 不保存 optimizer/EMA-target 恢复状态，中断后须从共享初始化重新开始，不支持无缝续训。

输入 LMDB 必须包含 `curvature_dataset_manifest.json`。训练前用 `analyze_curvature_intervention.py` 和 `audit_curvature_cd_design.py` 核对配对干预和 timestep alignment，流程见[旧 CD 说明](../rebuttal/CURVATURE_CD_SMALL.md)。如果 CD 数据也缺失，使用[数据准备入口](../rebuttal/build_curvature_dataset.py)按已核实设置重新生成并审计；普通 clean latent LMDB 和第 6 节的完整双向 teacher 轨迹都不能直接充当此数据。训练后核对两份 `training.done` 的配对字段和 sample-order digest，参考 `audit_curvature_cd_arms.py`。

生成的两个权重路径固定如下。实际检查配对视频后，使用同一评测 manifest 完成四条件 VBench；若调整了训练模型配置，评测配置也要保持对应设置一致。

```bash
"$OF_PY" -m experiments.iclr.evaluate cd \
  --original "$CD_OUT/curved/checkpoint_model_000300/model.pt" \
  --rectified "$CD_OUT/rectified/checkpoint_model_000300/model.pt" \
  --config experiments/rebuttal/configs/eval_all1.yaml \
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

本轮不自动增加训练 seeds 或 600/900 步预算。入口仍保留显式 `--steps 600/900`，供以后另定计划时使用；这些运行需新输出目录，并从共同初始化开始训练。所有新结果标明“本轮重训”，不能用仓库历史分数代替本轮评测。

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

## 7. 三位审稿人共享结果与 prompts 补充

A/B/C 完成后，按[合并任务](../../docs/ICLR_REVIEWER_COVERAGE.md)执行 `experiments.iclr.prompt_ablation`。该入口默认 dry-run，可通过 `--prepare-only` 在 CPU 上准备严格配对的 original/Qwen manifests；`--execute` 要求 tmux 和空闲 GPU，只评测已有 checkpoint，不启动训练。

`--conditioning paired` 比较同一 checkpoint 的两种文本条件；`--conditioning original` 只补原始 prompts。默认 main_reference 配置、EMA、main_ffe、每 prompt 1 个样本，支持显式指定 5 个并用新输出目录。汇总验证实际导出、配置、权重来源、样本数量和全 16 维评分协议；输出 `prompt_scores.csv`、配对模式的 `prompt_gains.csv` 和 `prompt_comparison.json`。prompt 增益与第 200 步训练 seed 方差分别报告，不保证复现历史 83.76。

基线要先核实模型兼容性和原生采样方案；详细来源表见 [ICLR_BASELINE_PROTOCOL_AUDIT.csv](../../docs/ICLR_BASELINE_PROTOCOL_AUDIT.csv)。未知项保持未知，缺失模型不能用其它 checkpoint 冒充。所有用于论文的 MP4 仍须实际播放检查。

## 8. 验证边界

```bash
"$OF_PY" -m pytest -q
git diff --check
```

CPU 测试覆盖：四-run 范围与执行顺序、配置混杂项、相同初始化审计、10-step 更新计数、真实循环的保存/恢复（tiny CPU models）、损失非有限失败、数据跨 epoch 续接、纯一步/四步评测命令、原始 prompts、teacher Euler 日程、非均匀积分、直线/变速直线/重复点/闭合路径、seeds 0/1 配对比较、两组配对差值的均值/样本 SD、frame/chunk 各两 seed 的固定步数汇总和图表导出。所有 synthetic fixture 均明确属于测试，不进入论文结果。

训练机验收仍需完成：关闭正则的两种粒度各 10 步 smoke、六个正式 run 的前 10 步监控、实际 FSDP 保存/恢复、原始/rectified CD 模型加载与采样、teacher 全路径采集，以及真实 MP4 动态画面检查。
