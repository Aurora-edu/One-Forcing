# CD 一步实验：已确认的新设置

本轮仍是 **2 个 CD 正式 run，各 300 updates，seed=0**；加上 frame/chunk 两种粒度 × 两个 seed（各 200 步），共 6 个正式 run。smoke、轨迹生成和诊断是验证/评测，不增加正式训练组。nPEg 优先。

用户选择：保留严格 original/rectified 配对，使用相同 teacher-forcing 历史，并增加诊断隔离历史变化的影响。不要改成两个模型各自产生历史、各自在线生成 teacher 端点的实验，那样会失去跨组固定端点。

## 初始化与数据

- 用户指定新实验初始化：Causal-Forcing 的 `checkpoints/framewise/ar_diffusion.pt`，raw 权重。
- 公开文件：`zhuhz22/Causal-Forcing` 仓库的 `framewise/ar_diffusion.pt`；SHA256 为 `721ea4d8f6d23099ccbefc6f5741b170402f509738f5a7b2db10e91b76857825`。可迁移路径，身份按内容 hash 核验。
- 两个学生和离线轨迹 teacher 均使用这份权重。它不同于 frame/chunk 实验 A 使用的 `causal_ode.pt`。
- 这项选择锁定**本轮新实验**，不代表已经证实历史 Table 1 使用了相同初始化。历史结果和仍缺失的证据见 [CD 来源记录](ICLR_CD_PROVENANCE.md)。
- 从 clean LMDB 生成一套 21 latent frames 的配对轨迹：48 步、CFG=6、shift=5，保存初始噪声、三个中间状态、teacher ODE 终点和独立 clean conditioning。后者仅用于历史条件，不能错当成 ODE 终点。
- 新采集代码把初始化 SHA256 写入每份轨迹及 LMDB manifest；缺少 hash 的旧数据不直接放行，须用指定 AR 权重重建。不要补填一个未经验证的 hash。
- 固定 LMDB 第 0–63 行作为留出诊断集，不参与两个 arm 的梯度更新；其余行顺序由相同 seed、sampler 和 8 卡 batch 决定。这是留出轨迹行，不保证与训练集 prompt 去重，也不是 VBench。

数据生成入口仅使用 `run_curvature_causal_experiment.sh prepare`；不要运行其 `all`，后者会训练另一套 1,000-step 实验。该阶段在独立 tmux 中运行，先确认 GPU 空闲：

```bash
export CD_INIT="$PWD/checkpoints/framewise/ar_diffusion.pt"
export CD_PREP="$ICLR_ROOT/cd_ar_data"
bash experiments/rebuttal/run_curvature_causal_experiment.sh prepare \
  --ar_checkpoint "$CD_INIT" --clean_lmdb "$CLEAN_LMDB" \
  --output_root "$CD_PREP" --gpus 0,1,2,3,4,5,6,7 --python "$OF_PY"
export CD_DATA="$CD_PREP/data/shared_curvature_lmdb"
```

`CLEAN_LMDB` 指实际 clean 数据目录，其它环境变量见[运行手册](../experiments/iclr/README.md)。所有输出都在 `/data/fengjiaqi/` 下，不提交权重/数据/视频到代码仓库。

## 训练目标与对照

[固定配置](../experiments/iclr/configs/cd_one_step.yaml) 使用 **21 latent frames、block=1、300 updates**，AdamW lr=2e-6、betas=(0.9,0.999)、weight decay=0.01、clip=10，训练目标 EMA decay=0.95。评测 raw generator，目标 EMA 仅用于 CD 的 stop-gradient target。

每次更新都计算：

`L = (L_adjacent_CD + L_high_noise_endpoint) / 2`

- `L_adjacent_CD`：保留原来的相邻状态一致性，四对状态按低噪声到高噪声循环，每对 75 次。
- `L_high_noise_endpoint`：在最高噪声 `t=1000` 输入上，一次网络预测直接监督到配对 teacher ODE 终点；每个正式 run 共 300 次。这明确训练 all1 推理使用的输入位置。
- 两项等权是预先固定的设置，不按 VBench 调权重或挑 checkpoint。分两次 backward 累积后再做一次 optimizer update，降低同时保留两张计算图的显存需求。
- Original/rectified 只改变三个中间状态；初始噪声、终点、clean conditioning、prompts、训练初始化、holdout、数据顺序及优化设置相同。相同模型参数下两组的 endpoint 项相同；轨迹区别通过 CD 项进入。不能仅保留 endpoint 回归，否则两个 arm 是同一个实验。

旧训练本来就不是“训练时串行生成四步”；它在四对相邻时间点上训练一致性。本次增加一步端点监督、由 9 帧改为 21 帧，但仍使用 teacher forcing，**不称为完全消除了训推历史差异**。新方法应标为 endpoint-anchored CD，不能与旧的纯 CD 设置混写。两组相互预算一致，但新增项增加了每步计算量，不能把本次 300 步成本等同于旧 300 步。

先跑两组 10 步 smoke（独立输出，观察到各自第 10 步），再跑正式组：

```bash
"$OF_PY" -m experiments.iclr.cd --data "$CD_DATA" --generator-ckpt "$CD_INIT" \
  --output "$ICLR_ROOT/cd_smoke" --smoke --session iclr_cd_smoke --dry-run
# 核对后改为 --execute；smoke 保存 0、5、10 步。

export CD_OUT="$ICLR_ROOT/cd_train/seed0"
"$OF_PY" -m experiments.iclr.cd --data "$CD_DATA" --generator-ckpt "$CD_INIT" \
  --output "$CD_OUT" --session iclr_cd_seed0 --dry-run
# 核对后改为 --execute；正式保存 0、50、100、150、200、250、300 步。
```

每个 run 的 `metrics.jsonl` 保存 CD/endpoint/总损失、梯度范数、pair index、样本行号、每步耗时和累计耗时。`training.done` 记录完整配置、源模型/数据 manifest hash 和样本顺序 digest；两组完成后输出 `paired_training_audit.json`。checkpoint 仍仅存 raw 模型，不含 optimizer 和 target EMA 恢复状态；中断需在新目录从共同初始化重跑。

## 历史条件诊断与 learning curves

`cd_diagnostics` 对两个 arm 的每个保存点使用相同留出行、初始噪声和 prompt；所有 latent frame 都只做一次去噪，无 FFE。通过同一个 KV sampler 比较：

1. **Forced history**：预测当前帧之后，用该行的真实 clean latent 更新历史 cache。
2. **Free history**：预测当前帧之后，用自身预测更新历史 cache。

两条路径首帧必须相同；当前/未来 clean latent 不得泄漏到当前预测。逐帧记录两条路径到固定 teacher ODE 终点的 MSE、`free_mse − forced_mse` 和两条预测之间的距离。

```bash
"$OF_PY" -m experiments.iclr.cd_diagnostics \
  --training-root "$CD_OUT" --data "$CD_DATA" \
  --output "$ICLR_ROOT/cd_history_diagnostics" --dry-run
# 核对后改为 --execute，自动 tmux。smoke 使用 cd_smoke 根目录，另加 --smoke。
```

正式诊断固定 64 行 × 7 个 checkpoint × 2 arms × 2 history modes；smoke 为前 8 行 × 3 个 checkpoint。输出 `per_frame.csv`、`history_curves.csv`、`rectified_minus_original.csv`、`learning_curves.png/pdf`，其中图同时展示 history MSE、CD/endpoint loss 和梯度范数。第 300 步固定前 4 行导出两种历史的配对 MP4（smoke 第 10 步前 2 行），必须实际播放检查运动、结构和画面，记录观察。程序能验证编码可读，不能据此宣称视觉正常。

`history_gap` 只能表示固定 teacher-forced target 下对历史替换的敏感性，**不是总误差的严格因果分解**；自由历史下理想目标也可能改变。MSE 不等同于视觉质量，变化可正可负，不据此选训练超参数。

## 主结果与结论边界

固定最终 step-300 raw generator，先完成两个 arm 的 **all1** VBench，再做 **all4** 辅助评测。使用相同原始 944 prompts × 1 sample、生成 seeds、全 16 维、无 FFE。命令见[运行手册第 5 节](../experiments/iclr/README.md#5-重训-originalrectified-cd再评测一步与四步)。入口会审计两组训练来源、配置、权重步数和 raw 权重；四条件汇总拒绝混入其它训练协议。

历史 all4 +9.95 / all1 −1.44 必须保留；四步正收益不保证一步正收益。本次目标是消除可修正的训练目标覆盖和序列长度问题、评估严格轨迹干预并诊断历史敏感性。即使 all1 仍负，也如实报告并收窄几何因果主张。不能以“涨分应为正”作为验收条件。
