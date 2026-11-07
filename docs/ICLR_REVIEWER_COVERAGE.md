# 三位审稿人的合并实验与回应

本文件补充[训练机执行任务](ICLR_TRAINING_MACHINE_TASK.md)。用户已确定：**nPEg 优先，共 6 个正式 run；framewise 与标准 main 设置对齐，chunkwise 仅改 block 大小，主表评测用 EMA。** 详见 [main 对齐记录](ICLR_MAIN_ALIGNMENT.md)。不运行旧 `run_iclr2027_priority.py` 的整套排程来替代本任务；它针对另一轮 Qwen 消融，且排除了本轮必做的 CD。

## 问题合并与执行优先级

| 优先级 / 工作 | nPEg | PkyQ | o6Jx | 新训练 |
|---|---|---|---|---:|
| P0-A：main 设置，frame/chunk 各 seeds 0、1，R1/R2 全关；固定 200 步，仅第 200 步 VBench EMA 表 | 相同步数/正则控制、两 seed 配对及耗时；不含学习曲线，未严格匹配 wall-clock | 粒度与实际成本 | 标准 frame 配置第 200 步的两 seed 重复性 | 4 |
| P0-B：配对端点监督 CD 各 300 updates、21 帧，再评 all1/all4 与历史诊断 | 一步 rectification | 一步 CD 与四步证据边界 | 一步几何假说的直接检验 | 2 |
| P0-C：时间参数化、弧长和路径几何复核 | 高噪声集中是否依赖坐标 | 收敛几何因果解释 | 收敛高噪声曲率的强结论 | 0 |
| P1-D：Table 2 各行协议与分数来源核对 | 辅助可信度 | 明确基线是否统一重评 | 解释主表和消融的 prompt 差异 | 0 |
| P1-E：主模型原始 / Qwen prompts 配对评测；可用基线补原始 prompts | 辅助公平比较 | 基线协议透明度 | 原始 prompts 下的主比较 | 0 |

同一个 CD 四条件表复用于三位审稿人的回答，不为每个审稿人重训一次。已有 GAN 配对消融已被三人认可，本轮无需重新训练 DMD-only 或添加 GAN 正则开/关消融。P0-A 的 frame 第二个 seed 也不额外重复。

执行顺序：先 P0-A，再 P0-B 和 P0-C，之后 P1-E；P1-D 的 CPU 文档核对可穿插进行。某个 P0 缺资产时先推进其他可执行 P0，只有没有可执行的 P0 GPU 工作时才插入 P1。所有长任务用独立 tmux，不占用其他 session 的 GPU。

## P0：六个训练 run 的共享交付

执行细节沿用主任务和[运行手册](../experiments/iclr/README.md)，不扩大训练矩阵：

1. `frame_roff_s0`、`chunk_roff_s0`、`frame_roff_s1`、`chunk_roff_s1`，全部固定 200 iterations 后结束，保存 0/50/100/150/200 步，VBench 只测第 200 步；中间点不做全量或子集 VBench。使用 main EMA、梯度 block 计数和 `main_ffe` 推理。实际梯度窗口随 block 大小变为 17/21 帧，要明确披露。
2. CD `curved`、`rectified`，指定 AR diffusion 初始化、seed 0、21 帧，各 300 updates。每步加入等权最高噪声端点监督，保留严格轨迹配对及 teacher forcing，并做留出历史诊断。详见 [新 CD 设计](ICLR_CD_ONE_STEP_DESIGN.md)。旧 CD 权重不再是前置条件。

第 200 步终点表包含 seeds 0、1 的 frame/chunk 配对差值，报告两组差值的均值和样本 SD；两种粒度也分别报告 n=2 的均值、样本 SD 与单次结果。不将两个 seeds 当成大样本显著性证据。报告训练时间、G/critic updates、实际保存时间、初始化与保存开销，并注明 GPU 型号和数量。8 卡训练的 GPU-hours 为 `8 × training_seconds / 3600`；端到端耗时单列，不把排队时间算进训练效率。相同步数不等于相同 wall-clock compute，本轮不声称完成严格等时长对照。用户已取消中间 VBench，因此不提供质量学习曲线；训练 loss/梯度曲线不能替代这一缺项。

PkyQ 还问原先“frame 无正则 / chunk 有正则”的实际成本：优先从历史日志补每步耗时、总训练时间、硬件和 R1/R2 配置。缺日志或硬件不一致就标明不能直接比较；不要把 200/750 的步数比写成 wall-clock 加速，也不要为此恢复已取消的正则开启训练。

CD 表固定列出 original/all1、rectified/all1、original/all4、rectified/all4 的全 16 维结果及 rectified−original 差值。正文优先回答 all1 的结果；四步改善只能支持相应条件下的学习改善。一步即使仍失败也应报告，不能据此删掉该实验。结合 P0-C，分别解释速度偏差、路径几何、坐标依赖与因果结论，不能仅凭一个正增益声称高噪声曲率是一步失败的唯一原因。

## P1-D：主表协议核对

[ICLR_BASELINE_PROTOCOL_AUDIT.csv](ICLR_BASELINE_PROTOCOL_AUDIT.csv)列出 PDF Table 2 的 13 行及待核实字段。其中分数只是论文记录，不是本轮测量；`unknown` 不能自动填成“相同协议”。训练机应逐行补充：

- 分数来自本项目重评、外部作者报告还是其他来源；具体文件/链接及版本。
- checkpoint、训练配置和 raw/EMA 权重来源。
- 原始或重写 prompts、rewrite 文件 hash、生成 seeds、每 prompt 样本数。
- 首 block 和后续 block 的 NFE、FFE、帧数、FPS、分辨率、CFG、VBench 版本及维度。
- 本轮是否可在共同协议下重新评测；不可用项列出确切缺失资产。

One-Forcing 主表在论文中声明使用 Qwen 和 FFE，但仓库 `PAPER_ALIGNED_VBENCH.md` 明确说原 83.76 的完整 rewrite 文件和随机流没有归档。仓库固定 Qwen assets 可用于新的配对评测，不能因此声称精确复现 83.76。历史 944×1 和 944×5、raw 和 EMA 结果要分表保留。

对 Self Forcing 1-step、Causal-Forcing 1-step 和 ASD 1-step 优先查找相应的真实权重与官方推理路径。现有通用导出器只适用于确认兼容的模型；不能仅因同属 Wan 骨干就直接加载。缺 1-step Self Forcing 权重时，不得把 4-step checkpoint 用 all1 采样后冒充论文 1-step 基线。不得用历史分数填充新的“共同原始 prompts”表。

完成核对后直接回答 PkyQ：哪些行确实统一重评、哪些沿用报告分数、哪些协议无法确认。没有证据时不要笼统写“所有基线采用完全相同设置”。

## P1-E：新增评测，不新增训练

先找来源已核实的历史主模型 checkpoint；若缺失，可预先指定 P0-A 的 frame seed 0 第 200 步 EMA checkpoint，明确标为本轮标准配置重训，不能称为历史 83.76 权重。不要事后挑得分最高的 seed 或其他 checkpoint；不新增训练。

主模型用同一个 checkpoint 的 EMA、同一 `main_ffe` 设置，在 944 个 original / pinned-Qwen prompts 上配对评测。用户已确认主表使用 EMA。默认每 prompt 1 个样本，明确标为补充协议；两个条件的 prompt_index、sample_index、噪声 seed 和评分 prompt 相同，仅模型看到的文本不同。

新入口 `experiments.iclr.prompt_ablation` 默认只打印命令，默认权重为 EMA、schedule 为 `main_ffe`：

```bash
export MAIN_CKPT=/data/fengjiaqi/checkpoints/verified_headline/model.pt
# 在独立 tmux 内执行；先 dry-run，确认来源和 P0 完成状态后改 --execute。
"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint "$MAIN_CKPT" \
  --config experiments/iclr/configs/main_reference.yaml \
  --method framewise --schedule main_ffe --weight-source generator_ema \
  --conditioning paired --samples-per-prompt 1 \
  --output "$ICLR_ROOT/followups/headline_prompts" \
  --vbench-python "$VB_PY" --dry-run
```

输出 `prompt_scores.csv`、`prompt_gains.csv` 和 `prompt_comparison.json`，同时保留每个条件的 export、VBench protocol 和原始结果。`prompt_gains.csv` 报告 Qwen−original；它不是训练 seed 方差。若正式扩展到每 prompt 5 个样本，所有比较方必须一致，使用新输出目录，不能仅替换一行与单样本表混比。

然后把可用的一步基线加入**原始 prompts 补充表**，统一评分 prompts、样本数、生成 seeds、时长/帧率/分辨率和指标版本；保留各方法本身经过确认的原生采样方案，并明确披露 FFE 与 NFE。若某方法有官方支持的 FFE 变体，可另作采样方案对照；不能把未验证的 FFE 强加给所有基线再称为原方法。

兼容本仓库 exporter 的模型可用同一入口 `--conditioning original`；同一 sampling manifest 可与官方 adapter 对接。如下仅是**四步 Self Forcing 参考模型**示例，不替代一步基线：

```bash
"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint /data/fengjiaqi/checkpoints/verified_self_forcing_dmd.pt \
  --config self_forcing_config.yaml --method chunkwise --schedule all4 \
  --weight-source generator_ema --conditioning original --samples-per-prompt 1 \
  --output "$ICLR_ROOT/followups/self_forcing4_original" \
  --vbench-python "$VB_PY" --dry-run
```

ASD 等需要原生 adapter 的模型仍通过其已验证推理代码执行，保留相同 manifest 的采样键与完整 provenance。只有实际完成并核实协议的行进入新表；缺少基线时明确称为“可用模型的补充比较”，不声称完整重评 Table 2。固定 prompts 的 MP4 必须实际播放检查，包括两侧的运动、结构崩溃、闪烁和冻结。

## 主表 seed 方差的回应边界

P0-A 已对齐 `main@c9a2350`：`gradient_num_blocks=17`、EMA=0.99（从 50 步开始维护），第 200 步结束并保存。两个 frame seeds 的第 200 步 EMA 结果可以提供标准配置的训练 seed 重复性，不需要再追加两次训练。评测使用 pinned Qwen × 5 samples 与 main 的首块增强。

报告两个独立训练 seed 的单次分数、均值和样本 SD，说明 n=2。原主表的完整 rewrite/随机流未归档，不能把新评测的波动直接说成对历史数值的逐位重现；step-400 和多生成几个 noise seeds 均不能替代第 200 步训练 seed 表。相对 Self Forcing 4-step 的 0.30 分差异仍不能据此声称统计显著。

## 最终回答组织

- 给 nPEg：优先展示 P0-A 第 200 步两 seed 配对终点、相同 200 iterations/正则控制、P0-B 的 all1 结果及 P0-C 参数化复核。明确未补学习曲线或 matched-wall-clock 对照，并披露每种粒度只有 2 个 seeds 的样本量限制。
- 给 PkyQ：复用 A/B，追加历史成本的可核实部分和逐行 Table 2 协议说明。
- 给 o6Jx：复用 B、原始 prompts 的补充比较，提供标准 frame 配置第 200 步 EMA 两 seed 结果，并披露小样本及历史评测资产限制。

每份回答列出对应真实结果路径、视频检查记录、失败/负结果和尚未覆盖的限制。不能把 dry-run、合成测试或历史分数计入本轮实验完成项。
