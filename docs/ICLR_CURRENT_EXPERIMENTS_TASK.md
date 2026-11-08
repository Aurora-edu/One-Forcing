# 本轮训练机任务：18 帧 frame/chunk + 一步局部拉直 CD

这是本轮总任务入口。请执行实验并交付原始结果；先阅读适用的 AGENTS.md，保留已有修改和其它 session 的进程。使用最新 `ICLR` 分支和训练机已经跑通的 B200 环境、资产、VBench wrapper。两项实验分别按下面链接执行，不将训练或评测配置混用。

## 本轮范围

| 子任务 | 正式训练 run | training seed | 每组预算 | 最终 VBench |
|---|---|---:|---:|---|
| A：反传窗口一致的粒度消融 | `frame_w18_s42`、`chunk_w18_s42` | 42 | 200 iterations | step 200，EMA + main_ffe，pinned Qwen，944×5 |
| B：高噪声状态局部渐进拉直 CD | Original（`curved`）、Local Rectified（`rectified`） | **2026** | 300 updates | step 300，**raw + all1，无 FFE**，原始 prompts，944×1 |

**合计 4 个正式训练 run、4 个终点 VBench 条件。** A 共 9440 条评测视频，B 共 1888 条，共 11328 条。B 的两个 10-update smoke 是预检，不计入正式 run 或 VBench 条件。每组正式训练都要观察到第 10 步；长任务全部放独立 tmux，顺序使用同一组空闲 8 卡。

用户已确认 B 本轮只做 seed=2026 一对，不运行 seed=2027，也不对单 seed 报告训练方差。此前取消的是「公开 One-Forcing 配方独立训练 chunkwise，再测 original prompts」那一组；它继续取消，与本次 CD 无关。已完成的旧 seeds 0/1、端点监督 CD 和旧四步 CD 都保留，不重跑。

## A：先完成 frame/chunk 对照

执行 [ICLR_WINDOW18_SEED42_TASK.md](ICLR_WINDOW18_SEED42_TASK.md)。两组独立训练，实际反传窗口都是最后 18 latent 帧，训练/score 仍为 21 帧，chunk 为 6 个完整反传 block。R1/R2 都关闭，训练 seed=42；仅第 200 步 EMA/main_ffe 评测，不做中间 VBench。

配置必须显式使用 `--profile matched_window18_s42_v1`。该文档包括生成配置、8 卡 tmux 训练、初始化核验、终点评测和 `experiments.iclr.summarize_window_pair` 汇总命令。本轮 A 保持此前已经确认的设置。

## B：补做此前新增的一步 CD

执行 [ICLR_CD_LOCAL_ALL1_TASK.md](ICLR_CD_LOCAL_ALL1_TASK.md)，使用 `python -m experiments.iclr.cd_local`，profile=`local_progressive_cd_all1_v1`。这是已有代码的新干预方案，不是旧 `cd_original` 的全局拉直四步对照。

固定 seed=2026、前 9 latent 帧、纯相邻状态 CD、endpoint loss=0、holdout=0、各 300 updates。Original 不改轨迹；Local Rectified 仅对第 1 个高噪声中间状态局部拉直，`alpha=min(step/100,1)`。两组同源 raw 初始化、轨迹、样本顺序与优化设置；复用 hash 匹配的现有 LMDB，不重新生成整套轨迹。

CD 的 target EMA=0.95 仅用于训练目标；最终导出 **raw generator**，不能套用 A 的 EMA 权重选择。所有生成、smoke 样例、诊断和 VBench 均为 **all1，首帧也一步，无 FFE**；不得自动追加 all4 或 `main_ffe`。

沿用此前训练机路径，启动前确认路径存在：

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export CDLP_BASE="$ICLR_ROOT/cd_local_progressive_all1_v1"
export CDLP_DATA="$CD_DATA"
export CDLP_INIT="$CD_INIT"
export CDLP_VB_PY="$R/tools/vbpy"
export CAUSAL_DISABLE_FLEX_ATTENTION=1
export CDLP_SEED=2026
export CDLP_OUT="$CDLP_BASE/seed$CDLP_SEED"

"$OF_PY" -m experiments.iclr.cd_local \
  --data "$CDLP_DATA" --generator-ckpt "$CDLP_INIT" \
  --seed "$CDLP_SEED" --output "$CDLP_OUT" --dry-run
```

接着按 B 详细文档顺序执行 CPU preflight、两个 10-update smoke、同 prompts 的 all1 视频检查、两组正式训练、终点诊断与两个 all1 VBench。若输出目录已有内容，先核对实际 run.intent/training.done/profile/seed；已按本轮配置完成的工作保留，缺项接着做。不要覆盖旧目录、伪装 optimizer 恢复，或把旧 CD checkpoint 改标签作为本轮产物。

B 的评测必须显式传 `--training-profile local_progressive_cd_all1_v1`。评测 dry-run 应恰好打印两个 `--schedule all1` 命令；旧 `evaluate cd` 的默认四条件不适用。缺原始 prompts manifest 时，用现有 `experiments.iclr.prompt_ablation --prepare-only --samples-per-prompt 1` 准备同协议文件，再把 `manifests/prompts.txt` / `manifests/original.jsonl` 显式传入 CD 评测，不改成 Qwen 或五样本。

## 验证和交付

两项任务顺序执行，不抢占其它 session 的 GPU。每组检查前 10 步 loss/grad norm 与实际配置；失败时保留日志，不按分数调 seed、训练步数或干预强度。观察实际完整 MP4，保留固定同 prompts 的配对视频、观看记录和异常；指标或静态拼图不能代替动态视频检查。

A、B 各自使用独立输出根目录、manifests 和汇总。A 交付 frame−chunk 单次原始分数与配对差值；B 交付 Original/Rectified 的 Total/Quality/Semantic、16 维原始分数、Rectified−Original 差值与一步诊断。两项都只有一个 training seed，不估计 seed 方差。

最后提供 4 个正式 run 的配置、实际 seed、源码 commit/环境、初始化和数据 hashes、训练日志/实际耗时、最终权重路径、原始 VBench JSON、导出/评分 provenance、视频样例与检查记录。汇总表区分 0–1 和百分制，不混算两项不同评测协议；打包轻量结果时不放大权重或 LMDB。
