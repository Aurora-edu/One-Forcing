# ICLR 2027：受控稳定性与轨迹实验

**当前一步实验更新：** 以[高噪声状态局部渐进拉直任务](../../docs/ICLR_CD_LOCAL_ALL1_TASK.md)为准，只评测 Original/Rectified 两个 all1 条件，首帧也一步；不运行四步生成或四步评测。下文旧四条件方案保留作历史记录，不是本次执行范围。

**2026-11-08 最新任务：** 已完成实验保留，只补 [原配置 CD 对照](../../docs/ICLR_CD_ORIGINAL_TASK.md)。使用 `python -m experiments.iclr.cd_original`：9 帧、纯 CD、seed=2026、各 300 步；同源轨迹可复用。第 5 节的 `cd_one_step.yaml` 是上一轮已完成的 21 帧端点监督扩展，不能用于本次原配置对照。

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

训练 all1；主评测只在第 200 步使用 EMA、`main_ffe`、pinned Qwen、944×5 samples。`main_ffe` 保持固定块，只给第一块四步，后续一步；不要用 review 的 first4then1 `ffe`。保留 0/50/100/150/200 的训练 checkpoint，但 0/50/100/150 不做任何 VBench（包括子集），也不生成 step-0 VBench 参考。

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

正式训练在第 200 步结束，保存 **0、50、100、150、200** 步，不因实际耗时较短而延长。计时包含取 batch、前后向、参数更新、EMA 和必要同步，取各 rank 最大耗时；初始化、checkpoint I/O、日志/GC 分列。仅比较第 200 步的 VBench 终点，实际训练耗时作为成本单独报告；不提供 VBench 学习曲线，也不称为严格的 matched-wall-clock 对照。全程不在线运行 VBench，每个新运行仍应观察头 10 步。

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

新目录的 `checkpoints.jsonl` 仅包含恢复后保存的点。只评测最终完成的第 200 步，不回补原运行的中间 VBench；每个 seed 只保留一个有效终点，累计训练耗时使用恢复后的记录。

CPU 测试替换 CUDA/FSDP collectives，用真实训练循环验证优化器/EMA/RNG/数据续接。训练机需验证真实 FSDP 保存、恢复和后续更新；10 步 smoke 尚无 EMA，另需覆盖第 50 步后的 EMA 恢复，可在正式运行中按计划完成。

## 4. 视频检查与第 200 步终点评测

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
  --index "$ICLR_ROOT/runs/frame_roff_s0/checkpoints.jsonl" --reference-step 200 \
  --config "$ICLR_ROOT/configs_formal/frame_roff_s0.yaml" \
  --prompts "$ICLR_ROOT/manifests/main/vbench_prompts.txt" \
  --extended-prompts "$ICLR_ROOT/manifests/main/vbench_qwen_rewrites.txt" \
  --manifest "$ICLR_ROOT/manifests/main/vbench_qwen_5sample_seed0.jsonl" \
  --samples-per-prompt 5 --full-info "$INFO" \
  --output "$ICLR_ROOT/eval_step200/frame_roff_s0" --vbench-python "$VB_PY" --dry-run
# 核对后改 --execute；逐一对 4 个 run 执行，不能并发抢占同一批卡。
"$OF_PY" -m experiments.iclr.summarize \
  --evaluations "$ICLR_ROOT/eval_step200" --output "$ICLR_ROOT/summary_step200" --reference-step 200
```

评测从 index 读取真实粒度，仅选择第 200 步，固定 EMA/main_ffe/Qwen×5。`curve` 只是保留的旧命令名，省略 `--reference-step` 也只评测 200；若没有最终 checkpoint 就报错，不回退到中间点。A 共 4×944×5=18,880 条 VBench 视频。汇总输出 `points.csv`、`same_seed_comparison.csv`、`paired_step_summary.csv`、`step200_seed_summary.csv` 和 `protocol.json`；frame/chunk 各 n=2，报告单次分数、均值、样本 SD、各 seed 实际耗时和配对差值，不输出 VBench 质量曲线。

终点结果同时用于粒度比较和主模型训练 seed 重复性，不重复生成或评分。使用独立 `$ICLR_ROOT/eval_step200/<run>` 目录；汇总拒绝混入旧中间点。中间 checkpoint 继续保存用于恢复，训练 loss/梯度日志保留；不补跑中间的全量或子集 VBench。

`step200_seed_summary.csv` 给出标准训练设置下的两 seed 单次分数、均值和样本 SD。原主表完整 rewrites/随机流未归档，不保证新评测恰好得到 83.76。

缺少指定 run、第 200 步终点、额外 seed、正则开启或评测协议不一致会报错。若某 seed 发散，先报告失败及已有训练日志，不要补选“好 seed”来通过完整表检查。两个 seeds 的样本量仍有限，不据此宣称统计显著或普遍稳定。

## 5. 重训 original/rectified CD，再评测一步与四步

**本轮新设置已确定，执行前阅读 [CD 一步实验详细设计](../../docs/ICLR_CD_ONE_STEP_DESIGN.md)。** 两组各 300 updates、seed=0、21 latent frames，从用户指定的 Causal-Forcing `checkpoints/framewise/ar_diffusion.pt` raw 权重初始化，并用相同权重重新生成共享轨迹。与实验 A 的 ODE 初始化区分，不再使用来源待定的路径或旧 9 帧配置。

每次更新使用 `(L_adjacent_CD + L_high_noise_endpoint)/2`，保留严格 original/rectified 配对和相同 teacher-forcing 历史。前 64 个轨迹行留出诊断，其余用于训练。正式配置固定为 `endpoint_anchored_cd_v1`；总训练仍为 6 个 run。历史设置、旧负结果及证据限制保留在 [CD 来源记录](../../docs/ICLR_CD_PROVENANCE.md)。

```bash
export CD_DATA="$ICLR_ROOT/cd_ar_data/data/shared_curvature_lmdb"
export CD_INIT="$PWD/checkpoints/framewise/ar_diffusion.pt"
export CD_CONFIG=experiments/iclr/configs/cd_one_step.yaml
export CD_OUT="$ICLR_ROOT/cd_train/seed0"

"$OF_PY" -m experiments.iclr.cd \
  --config "$CD_CONFIG" --data "$CD_DATA" --generator-ckpt "$CD_INIT" \
  --output "$ICLR_ROOT/cd_smoke" --smoke --session iclr_cd_smoke --dry-run
# smoke 核对后改为 --execute；两组各观察到第 10 步，并按下文运行 --smoke 诊断、检查视频。

"$OF_PY" -m experiments.iclr.cd \
  --config "$CD_CONFIG" --data "$CD_DATA" --generator-ckpt "$CD_INIT" \
  --steps 300 --seed 0 --output "$CD_OUT" --session iclr_cd_seed0 --dry-run
# 核对后改为 --execute；自动 tmux，顺序训练两个 arm。
```

使用新数据目录，旧 LMDB 没有源权重 SHA256 时须重建，不能补填未经验证的 hash。具体 `prepare` 命令见详细设计；不要运行旧数据脚本的 `all`（会额外执行另一套 1,000-step 实验）。两组训练只改变三个轨迹中间状态，共享噪声、端点、clean conditioning、prompts、初始化、holdout、配置和样本顺序。

每步写 `metrics.jsonl`，保存 **0、50、100、150、200、250、300** 步 raw generator，训练完成自动审计配对配置和 sample-order digest。固定 step-300 为主结果，不根据中间分数选 checkpoint。target EMA=0.95 仅用于训练目标；评测不使用它。当前 checkpoint 未保存 optimizer/target EMA 恢复状态，中断后需新目录从共同初始化重跑。

先完成两个 arm 的 all1，再做 all4。评测入口拒绝混入旧配置、smoke 或非 step-300 权重：

```bash
"$OF_PY" -m experiments.iclr.evaluate cd \
  --original "$CD_OUT/curved/checkpoint_model_000300/model.pt" \
  --rectified "$CD_OUT/rectified/checkpoint_model_000300/model.pt" \
  --config experiments/rebuttal/configs/eval_all1.yaml \
  --prompts "$ICLR_ROOT/manifests/original.txt" \
  --manifest "$ICLR_ROOT/manifests/original_seed0.jsonl" --full-info "$INFO" \
  --output "$ICLR_ROOT/cd_eval/step300" --vbench-python "$VB_PY" --dry-run
# 核对后改为 --execute，在独立 tmux 中执行。

"$OF_PY" -m experiments.iclr.summarize_cd \
  --evaluations "$ICLR_ROOT/cd_eval/step300" --output "$ICLR_ROOT/cd_summary/step300"
```

四条件均为原始 944 prompts × 1、相同生成 seeds、raw 权重、全 16 维、无 FFE。输出 `cd_scores.csv`、`cd_gains.csv`、JSON；分数为 0–1，论文百分制乘 100。单一训练 seed 的聚合增益不是训练 seed 置信区间。历史一步 −1.44 / 四步 +9.95 保留；不能保证新一步结果为正。

补充诊断复用相同留出样本、噪声和 KV sampler，对比真实历史与自由生成历史，导出逐帧误差和 learning curves：

```bash
"$OF_PY" -m experiments.iclr.cd_diagnostics \
  --training-root "$CD_OUT" --data "$CD_DATA" \
  --output "$ICLR_ROOT/cd_history_diagnostics" --dry-run
# 核对后改为 --execute；自动 tmux。smoke 指向 cd_smoke，另加 --smoke，使用独立输出目录。
```

输出 `per_frame.csv`、`history_curves.csv`、`rectified_minus_original.csv`、`learning_curves.png/pdf`，图包含两项训练 loss、梯度范数和两种历史的 MSE。正式固定前 4 个留出行导出 step-300 配对视频；smoke 前 2 行导出 step-10 视频。必须实际播放检查动态画面；文件可解码不等于视觉正常。history gap 是固定 teacher-forced target 下的历史敏感性，不是总误差的严格因果分解；MSE 不能替代 VBench。

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

CPU 测试覆盖：四-run 范围与执行顺序、配置混杂项、相同初始化审计、10-step 更新计数、真实循环的保存/恢复（tiny CPU models）、损失非有限失败、数据跨 epoch 续接、纯一步/四步评测命令、原始 prompts、teacher Euler 日程、非均匀积分、直线/变速直线/重复点/闭合路径、seeds 0/1 配对比较、两组配对差值的均值/样本 SD、frame/chunk 各两 seed 的第 200 步终点汇总。所有 synthetic fixture 均明确属于测试，不进入论文结果。

训练机验收仍需完成：关闭正则的两种粒度各 10 步 smoke、六个正式 run 的前 10 步监控、实际 FSDP 保存/恢复、原始/rectified CD 模型加载与采样、teacher 全路径采集，以及真实 MP4 动态画面检查。
