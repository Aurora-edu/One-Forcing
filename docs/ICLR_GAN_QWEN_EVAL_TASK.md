# seed=2026 的 DMD-only / DMD+GAN 配对训练与 Qwen 评测

用户已确认：旧论文 Table 4a 两份 step200 权重未找到，改做一对新的 seed=2026 训练。复用本轮 `frame_roff_s2026` 的 step200 **raw generator** 作为 DMD+GAN，仅新增一个 `dmd_only_s2026` 训练，不重复训练 GAN 模型。

本轮合计三组200步训练：frame、chunk、DMD-only。CD/轨迹实验独立，由原任务管理。本文件替代之前“找旧权重、只换 prompt”的执行方案。

## 固定实验设置

| 项目 | DMD+GAN | DMD-only |
|---|---|---|
| 训练 run | `frame_roff_s2026`，复用 | `dmd_only_s2026`，新增 |
| Profile | `main_block_ablation_s2026_v1` | `main_dmd_only_s2026_v1` |
| 实际 training seed | 2026，禁止随机化 | 相同 |
| 训练预算 | 200 iterations：40次G、200次critic更新 | 相同 |
| GAN G / D 权重 | 0.03 / 0.03 | 0 / 0 |
| 训练 block / 反传 | block=1、gradient_num_blocks=17，实际17帧 | 相同 |
| 初始化、数据、学习率、EMA、R1/R2 | 沿用 seed2026 framewise 配方 | 相同 |
| 本项推理权重 | step200 `generator` raw | 相同 |
| 本项推理协议 | 旧版 `ffe`、固定Qwen、944×1、21 latent / 81 RGB frames、16fps | 相同 |

训练均使用 `experiments.iclr.train` / `ICLRTrainer` 和同一份代码。DMD-only沿用现有实现：两个GAN权重及R1/R2为零时，不创建GAN判别头、不计算GAN损失，保留DMD和fake-score去噪训练。只改变目标相关开关；其它训练配置逐项核验。固定sampler seed仍为0。

训练器新增 `critic_backbone_tensor_sha256`，对比两组**共有主干**的初始化。完整critic hash不应强求相同，因为DMD-only没有GAN头。生成器文件hash、实际加载tensor hash和共有主干hash均须相同。相同training seed和数据顺序不意味着训练噪声逐次相同：GAN分支会消耗额外随机数。保留原行为，不另改训练RNG流程。

**这是一对新训练的目标消融。** 不能将旧75.27/80.36和新分数之差解释为仅由prompt rewriting造成；旧权重身份无法复现。无需为本任务额外跑original prompts。

**保留旧版 `ffe`：** `eval_ffe.yaml`对应`first4then1`，首块联合生成4个latent frames、4次去噪，之后每块1帧、一步。同一framewise权重在frame/chunk任务中另按 **EMA / main_ffe / Qwen944×5** 评测，两个输出目录和协议分开。

## 生成配置并启动新增训练

先按 [本轮训练与seed0日志上传总入口](ICLR_SEED2026_TRAINING_TASK.md) 接入GitHub `ICLR` 分支交接prompt指定的提交，再完成 [seed2026 frame/chunk 任务](ICLR_SEED2026_PAIR_TASK.md) 的配置核验。两任务必须使用同一代码快照，保留commit及完整diff/未跟踪代码副本。

下列变量沿用已验证B200环境；路径不同时只映射真实位置。`ODE_INIT`、`TEACHER`、`CLEAN_LMDB`必须与新framewise组完全相同。不要从旧resolved_config继承旧logdir/resume。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export SEED2026_ROOT="$R/iclr2027_main_s2026"
export DMD_ROOT="$R/iclr2027_dmd_only_s2026"
export GAN_QWEN_ROOT="$R/iclr2027_gan_qwen_s2026"
export VB_RUNNER="$R/tools/vbpy"
export GAN_RUN="$SEED2026_ROOT/runs/frame_roff_s2026"
export DMD_RUN="$DMD_ROOT/runs/dmd_only_s2026"
export GAN200="$GAN_RUN/checkpoint_model_000200/model.pt"
export DMD200="$DMD_RUN/checkpoint_model_000200/model.pt"

"$OF_PY" -m experiments.iclr.make_configs \
  --profile main_dmd_only_s2026_v1 \
  --base experiments/iclr/configs/base.yaml \
  --generator-ckpt "$ODE_INIT" --teacher "$TEACHER" --data "$CLEAN_LMDB" \
  --output "$DMD_ROOT/configs"

"$OF_PY" -m experiments.iclr.launch \
  --matrix "$DMD_ROOT/configs/matrix.json" --output "$DMD_ROOT/runs" \
  --python "$OF_PY" --dry-run
```

dry-run应只有一个8卡训练命令。确认8卡空闲后启动，不停止其它session进程：

```bash
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$DMD_ROOT/configs/matrix.json" --output "$DMD_ROOT/runs" \
  --run dmd_only_s2026 --python "$OF_PY" --session iclr_dmd_s2026 --execute
```

入口创建独立tmux。至少盯到第10步：runtime seed=2026，loss/梯度有限，`gan_g_loss`、`gan_d_loss`、`r1_loss`、`r2_loss`恒零，DMD和critic仍有更新。训练异常保留日志，不自行改seed/学习率/预算。

两组已初始化时先跑初始化核验；两组完成后再次做终点核验。新版训练器必须在两组启动前部署，旧记录缺少主干hash时不能补写假值。

```bash
"$OF_PY" -m experiments.iclr.audit_gan_pair \
  --gan-run "$GAN_RUN" --dmd-run "$DMD_RUN" --initialization-only \
  --output "$GAN_QWEN_ROOT/audit/initialization.json"

# 两组训练均结束后执行；仅此命令成功才进入正式GAN消融评测。
"$OF_PY" -m experiments.iclr.audit_gan_pair \
  --gan-run "$GAN_RUN" --dmd-run "$DMD_RUN" \
  --output "$GAN_QWEN_ROOT/audit/step200_pair.json"
```

终点核验读取真实raw权重、profile、完整配置、初始化、200步/40G/200critic和逐步loss记录。若恢复训练产生多个目录，保留并按实际步号整理完整1..200日志及来源链；不要把缺日志的训练直接当完整配对结果。

## 两组Qwen评测

复用仓库固定Qwen rewrites，不重新调用Qwen。原始prompt/sample/noise配对不变；评分文本仍是原始VBench文本。两组新视频共1888条。

```bash
"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint "$DMD200" --config experiments/rebuttal/configs/eval_ffe.yaml \
  --method framewise --schedule ffe --weight-source generator \
  --conditioning qwen --samples-per-prompt 1 \
  --output "$GAN_QWEN_ROOT/dmd_only" --python "$OF_PY" \
  --vbench-python "$VB_RUNNER" --dry-run

"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint "$GAN200" --config experiments/rebuttal/configs/eval_ffe.yaml \
  --method framewise --schedule ffe --weight-source generator \
  --conditioning qwen --samples-per-prompt 1 \
  --output "$GAN_QWEN_ROOT/dmd_gan" --python "$OF_PY" \
  --vbench-python "$VB_RUNNER" --dry-run
```

每组只有一个`--name qwen`条件，底层明确`--require_no_ema --schedule ffe --samples_per_prompt 1`。确认后在空闲8卡和独立tmux内，顺序将`--dry-run`换成`--execute`。tmux内重新设置变量，不覆盖其它实验目录。

固定输入核验值：

- Qwen单样本manifest SHA256：`182effc44e9d8ca9e4f388fa9c4d6d01ceae25369886a1fd3eccce64ab7cd764`。
- `eval_ffe.yaml`展开后SHA256：`3bd52f7233950e7304382df3cf62654fbc65d1ba7b21122dce8613ddb6584c8a`。
- 原始单样本manifest SHA256：`521d3e8f58bcb5e7c1574c3ab2cf34cdb604483b66a3b53b41942497558cabe5`，仅核对配对来源，不新增original评测。

两组完成后汇总：

```bash
"$OF_PY" -m experiments.rebuttal.summarize_single_seed_vbench \
  --result "dmd_only=$GAN_QWEN_ROOT/dmd_only/qwen/vbench/qwen_eval_results.json" \
  --result "dmd_gan=$GAN_QWEN_ROOT/dmd_gan/qwen/vbench/qwen_eval_results.json" \
  --comparison "gan_gain=dmd_gan,dmd_only" \
  --output_path "$GAN_QWEN_ROOT/gan_qwen_summary.json"
```

核对两组`prompt_comparison.json`中的manifest/config/rewrite hashes相同，checkpoint分别为本次核验的两份step200 raw。报告Total/Quality/Semantic、全部16维原始值和GAN−DMD差值，注明seed2026、单样本协议。结果无论方向如何均保留。

预先固定至少16个相同prompt/noise pairs，实际观看两组完整MP4，检查模糊、运动、冻结、闪烁和结构崩溃；不能只看指标。若只能抽帧，明确记录尚未做完整播放检查。交付训练配置和代码快照、初始化/终点audit、训练日志/耗时、权重位置、export/protocol、原始VBench JSON、汇总和视频检查记录。

本机只完成CPU验证；正式GPU训练、至少10步观察和视频检查由训练机执行。
