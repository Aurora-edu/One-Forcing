# 训练机任务：原配置 CD 对照（2026-11-08）

本次只补 **original/rectified 两个 9 帧纯 CD run**，以及同一对最终权重的一步/四步评测。已完成的 frame/chunk 四组、21 帧端点监督 CD、几何分析和 prompts 评测保留，不重跑。用户要求恢复论文原 CD 训练配方；不得再增加端点损失、改成 21 帧或更换 seed。

## 固定训练和评测设置

训练直接继承 `experiments/rebuttal/configs/train_curvature_cd_small.yaml`；参考文件 SHA256 固定为 `d4bdfeb74bfd466ba19ed766ddf0fd9ad7613596e72a8b114c0a90ebfa391a64`。新增入口 `experiments.iclr.cd_original` 会验证全部原配置字段，拒绝未声明的覆盖。

| 设置 | 本次两组完全相同 |
|---|---|
| Profile | `original_adjacent_cd_9f_v1` |
| 训练长度 / seed / global batch | 300 updates / **2026** / 8（每卡 1） |
| 训练窗口 | 仅现有轨迹的**前 9 latent frames**，block=1 |
| 损失 | **纯相邻状态 CD**，endpoint loss weight=0，无额外端点 forward/backward |
| 四个相邻对 | 从低噪声向高噪声循环，每对 75 次 |
| 数据 | 全部轨迹行；**holdout=0**，恢复使用之前留出的前 64 行 |
| 优化器 | AdamW lr=2e-6，betas=(0.9,0.999)，weight decay=0.01，clip=10 |
| Target EMA | 0.95，仅训练目标，评测用 raw generator |
| 正式保存点 | 第 300 步；不增加中间 VBench |
| 评测 | 两个 step-300 raw 权重各做 all1/all4，21 latent frames，944 原始 prompts×1，相同生成 seeds，全 16 维，无 FFE |

`all4` 的真实输入时刻必须与轨迹一致：1000 / 937.5 / 833.3333 / 625。不得使用 Qwen prompts、EMA 权重或首帧增强。

## 复用资产与历史复现边界

优先核查是否找回了历史 Table 1 的源权重和配对轨迹；具体原始初始化身份此前尚未确认。不能将文件名相同视为内容相同，也不能预设四步增益必须恢复到 +9.95。

训练机已有 `cd_ar_data/data/shared_curvature_lmdb`（6505 行，21 帧）可以复用：显式传入其真实 raw 源权重，例如上一轮的 `ar_diffusion.pt`。新入口先检查**初始化文件 SHA256 与 manifest 中源模型 SHA256 一致**、raw/CFG=6、真实 timesteps、数据形状和前 9 帧的配对干预，再只取前 9 帧训练。训练切片不需要重建 LMDB，也不需要重新生成 6505 条轨迹。

若 hash 缺失或不一致，先查真实来源，不能补填 hash、换一个权重继续训练或直接启动整套轨迹重建。确认同源可以复用现有轨迹；确认历史输入不同，才需要另外处理该输入。复用上一轮 AR 数据得到的是**原训练配方的新受控重跑**；在历史源模型/数据身份未闭环时，不宣称精确复现 Table 1。

## 执行顺序

先保留本机改动并更新 `ICLR` 分支，阅读适用的 AGENTS.md。训练代码已纳入上一轮 B200 修复 `255c65e` 的同一处 barrier；如训练机保留该本地提交，合并时保留一次 barrier 即可。继续使用已经跑通的 B200 环境和 VBench wrapper，记录实际版本和偏差，不覆盖公共环境。

上一轮训练机可沿用这些路径（其它机器按真实路径映射）：

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export CD9_ROOT="$ICLR_ROOT/cd_original_9f_seed2026"
export CD9_DATA="$CD_DATA"
export CD9_INIT="$CD_INIT"
export CD9_VB_PY="$R/tools/vbpy"
export CAUSAL_DISABLE_FLEX_ATTENTION=1
```

先 CPU 预检，检查输出的源模型 hash；它不会修改或重建数据。训练机 AI 应核对 `preflight.json` 的 `historical_initialization_match`，按实际证据披露未确认项。

```bash
"$OF_PY" -m experiments.iclr.cd_original \
  --data "$CD9_DATA" --generator-ckpt "$CD9_INIT" \
  --output "$CD9_ROOT" --preflight
```

在空闲的同一组 8 张 B200 上执行两个 10 步 smoke，使用独立 tmux；每组必须观察到第 10 步。核对 loss、梯度有限，`endpoint_loss` / `endpoint_updates` 均为 0，实际张量窗口为 9 帧。

```bash
"$OF_PY" -m experiments.iclr.cd_original \
  --data "$CD9_DATA" --generator-ckpt "$CD9_INIT" \
  --output "$CD9_ROOT/smoke" --session iclr_cd_original_smoke --smoke --execute
```

smoke 完成后，用 `scripts/export_videos.py`、`experiments/rebuttal/configs/eval_all1.yaml`、同一套 `prompts/demos.txt`、`--method framewise --limit 4 --num_samples_per_prompt 1`，对两组 step-10 raw 权重分别导出 `--schedule all1` 和 `--schedule all4` 的视频到独立目录。先确认空闲，再用 `--gpu_id 0`。实际播放检查视频；10 步模型质量差须如实记录，解码成功或静态拼图不等于动态验收完成。

随后启动两组正式训练，分别观察头 10 步。入口自动创建 tmux 并顺序训练 curved / rectified。省略 `--execute` 或使用 `--dry-run` 只打印计划，不访问模型或 GPU。

```bash
"$OF_PY" -m experiments.iclr.cd_original \
  --data "$CD9_DATA" --generator-ckpt "$CD9_INIT" \
  --output "$CD9_ROOT" --session iclr_cd_original_formal --execute
```

完成后自动生成 `paired_training_audit.json`：检查两组所有控制项、实际逐步日志、300 updates、每相邻对 75 次、零端点监督和相同 sample-order digest。旧 checkpoint 不含 optimizer/target EMA 恢复状态；中断须使用新输出根目录从同一初始化重新运行，不能续训旧的 21 帧权重。

正式训练结束后，在独立 tmux 中评测四个条件。它显式选择原配置 profile，拒绝混入上一轮 21 帧权重或 smoke 权重。复用上一轮原始 prompts 和单样本 seed manifest。

```bash
"$OF_PY" -m experiments.iclr.evaluate cd \
  --training-profile original_adjacent_cd_9f_v1 \
  --original "$CD9_ROOT/curved/checkpoint_model_000300/model.pt" \
  --rectified "$CD9_ROOT/rectified/checkpoint_model_000300/model.pt" \
  --config experiments/rebuttal/configs/eval_all1.yaml \
  --prompts "$ICLR_ROOT/manifests/original.txt" \
  --manifest "$ICLR_ROOT/manifests/original_seed0.jsonl" --full-info "$INFO" \
  --output "$CD9_ROOT/eval" --vbench-python "$CD9_VB_PY" --execute

"$OF_PY" -m experiments.iclr.summarize_cd \
  --evaluations "$CD9_ROOT/eval" --output "$CD9_ROOT/summary"
```

原始 944 prompts 的 manifest 路径不存在时，可按运行手册准备同协议文件，不改成 Qwen 或五样本。`VBENCH` 环境继续使用上一轮 B200 兼容 wrapper（含 color 空 shard 修复），保留每个条件的环境偏差 JSON。所有评测四条件共 3776 条视频。

## 验收和交付

实际播放并比较四条件同一批固定 prompts 的 MP4，检查模糊、运动、闪烁、后段崩溃；不能仅凭时序拼图标记动态检查完成。若执行代理不能播放，保留明确的 pending 状态并提供 MP4 给作者检查，继续完成可执行的评分和记录。

请打包以下轻量产物：`preflight.json`、`configs/cd_original.yaml`、两个 `run.intent.json` / `training.done` / `metrics.jsonl`、`paired_training_audit.json`、四条件原始 VBench JSON / protocol / 环境偏差 / `iclr_cell.json`、prompt/seed manifests、导出校验记录、同 prompts 的实际 MP4 样例、视频检查记录，以及 summary 中的 CSV/JSON。权重路径和 hash 单列，不把大权重和 LMDB 塞进轻量包。B200 环境包装脚本和新增代码一并归档。

汇总提供 `cd_scores.csv`、`cd_gains.csv`，还自动生成 `historical_comparison.csv`（0–1 量纲，论文百分制乘 100），逐行比较新旧 Total/Quality/Semantic。旧记录为一步 56.79/55.35、四步 61.16/71.11；这些是历史参考，不是本轮新测量。保留所有正负结果，不能按结果改 seed、训练步数、loss 或 checkpoint。
