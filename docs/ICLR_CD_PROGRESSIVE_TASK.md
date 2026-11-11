# 独立实验：Original / Rectified 确定性渐进蒸馏

本文件是本轮训练机的完整任务。新增独立输出目录，保留之前 CD、frame/chunk、GAN 消融及全部结果。只做下面这轮；不触发旧合并任务，不追加 Qwen。

## 1. 先找到并核实四步起点

历史对照见 [来源审计](ICLR_CD_PROVENANCE.md)、[历史配置](../experiments/rebuttal/configs/train_curvature_cd_small.yaml)、[历史汇总](../experiments/rebuttal/results/final_metrics.json)。该实验是 **seed2026、300 updates、前9 latent frames、纯相邻状态 CD**，两组仅全局 trajectory rectification 干预不同；训练 target EMA=0.95，评测 raw。

归档四步 Total 为 Original **61.16** / Rectified **71.11**。这些是历史汇总记录，尚不能替代本次权重身份和生成质量核验。本机没有找到可验证的这对历史权重及其完整运行记录。训练机优先查 `eval/reviewer/curvature_cd_small`、旧训练归档、原配置9帧重跑目录，而非按 `model.pt` 文件名判断身份。

**不得直接采用本轮1000步端点监督权重，也不得采用300步局部高噪声拉直权重。** `original` 对应旧命名 `curved`；`rectified` 必须对应历史全局中间状态拉直。核对同一初始化 hash、trajectory manifest hash、300步 raw 参数、配套 `run.intent.json` 和配置。只有同名路径、汇总分数或无来源的导出文件不够；不补造历史 metadata。

CPU 盘点示例：

```bash
"$OF_PY" -m experiments.iclr.progressive inventory \
  --roots "$R/iclr2027_main_200" /实际历史训练归档目录
```

该命令只列候选路径，**不会认定权重已经核验，也不会恢复训练**。不要把不存在的示例路径原样传入。复制 [来源清单模板](../experiments/iclr/configs/cd_progressive_sources.example.json) 到新实验目录以外，填写实际绝对路径、实测 SHA256 和真实记录路径；恢复重跑的权重必须将 `origin` 写成 `recovered_four_step_cd`。

若历史记录缺少初始化 hash，或原权重/原数据找不到，先给用户交代缺口和下表成本。不要为了通过校验改写历史 checkpoint 或日志。可以提出恢复同一9帧配方的新四步对照，但不能称为找回了历史原权重。

| 情形 | 新增成本与下一步 |
|---|---|
| 找到可核验且生成正常的历史四步对照 | 直接做本轮4个阶段 run，共1200 updates |
| 缺失四步权重，但有 hash 匹配的初始化与共享轨迹 LMDB | 先恢复2个300步四步 teacher，**额外600 updates**；再检查四步视频，合计训练1800 updates |
| 初始化或配对轨迹也缺失 | 先确认源模型与数据；若需重建6505条48步轨迹，**单独报告这笔采样成本**，不能隐含在600步恢复预算里 |
| 找到了权重但四步视频异常 | 保存异常视频和记录，先诊断起点；不继续蒸馏 |

恢复方案和额外成本须先报告，确认后才安排恢复训练，这是用户本轮要求。恢复入口为 `experiments.iclr.cd_original`，详见 [9帧原配置任务](ICLR_CD_ORIGINAL_TASK.md)。只恢复两个训练 run；本轮不因此追加旧任务的四个完整 VBench。可先做只读/CPU 预检：

```bash
"$OF_PY" -m experiments.iclr.cd_original \
  --data "$CD_DATA" --generator-ckpt "$CD_INIT" \
  --output /新的独立四步恢复目录 --preflight
```

## 2. 本轮固定设置

| 项目 | Original / Rectified 均相同 |
|---|---|
| Profile | `original_rectified_progressive_v1` |
| 顺序 | 各自核验过的4步 teacher → 2步 student → 1步 student |
| 训练预算 | **每阶段300次 optimizer updates**；2组×2阶段=4个阶段 run，共1200次 |
| Seed / 优化器 | 2026；AdamW lr=2e-6，betas=(0.9,0.999)，weight decay=0.01，clip=10 |
| Batch / 训练窗口 | 8卡×每卡1；沿用历史训练的**前9 latent frames**，block=1 |
| 数据 | 相同 clean latent LMDB；排除前64行及这些 prompt 文本的重复行 |
| 固定留出样例 | 前64行中 row=0,4,...,60 共16例，生成 seed=2026+row；两组、各阶段固定 |
| 权重 / 目标 | 冻结上一阶段的 raw teacher；student 同源初始化；新 AdamW；无 EMA target / EMA 导出 / GAN / DMD 损失 |
| 训练条件 | clean 过去帧 teacher forcing；推理使用自身生成的过去帧，不输入真实视频 |
| 推理长度 | 21 latent frames → 81视频帧，832×480，16fps，block=1 |
| 最终评测 | **raw、纯 all1、无 FFE、original prompts 944×1**，全16维 VBench |

留出仅针对本轮渐进训练：历史 teacher 可能见过这些数据，不能宣称是历史 teacher 的独立测试集。训练窗口取9帧是为了沿用历史四步训练窗口，避免额外改成21帧；正式生成仍与历史评测相同为21 latent frames。

Rectified 的含义由**四步起点的训练来源**决定。渐进阶段两组使用相同算法，不再额外手工拉直、加端点监督或改变 loss 权重。结果无论方向如何均完整保留；不根据 VBench 选 seed、checkpoint 或修改预算。

## 3. 确定性采样和训练目标

参考 [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) 的确定性采样逐次减半思想，在当前 flow 参数化中显式实现。令 `x_t=(1-t)x0+t*noise`，teacher 一次确定性更新为：

```text
x_s = (s/t) x_t + (1-s/t) D_teacher(x_t,t)
```

连续两次 teacher 更新得到 `x_end`，student 的 clean 目标为：

```text
x0_target = [x_end - (end/t) x_t] / [1 - end/t]
loss = mean((D_student(x_t,t) - stopgrad(x0_target))²)
```

两次 teacher 之间不重加噪；固定输入与 teacher 应给出可重复的 target。每个阶段第一步实际检查重复 teacher forward 与 RNG 不变。该保证针对**采样路径确定性**，不宣称 GPU backward 跨设备逐位一致。

使用实际 shifted sigma，**不再次 warp**：

```text
4步: 1, 0.9375, 5/6, 0.625, 0
2步: 1, 5/6, 0
1步: 1, 0
```

4→2 的两个粗区间交替，各150次更新；2→1 为同一区间300次更新。每步全局样本行、prompt、噪声 hash 累积入配对摘要，两个 arm 不一致即停止。旧四步推理会在步间重新加噪，**不同于此次确定性 sampler**；因此起点同时输出旧 sampler 与确定性 sampler 的同一批视频，核实新 sampler 下的起点质量。

## 4. 训练机执行顺序

先保留机器本地改动，更新 `ICLR`，记录实际提交。优先独立 checkout/worktree，禁止 `reset --hard` 覆盖其它任务。沿用已跑通的 B200 环境和 `wan_models/` 资产链接，记录 torch/CUDA/flash-attn 版本。下面路径变量按实际环境填写，示例可参考：

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export PD_ROOT="$R/cd_progressive_s2026_v1"
export PD_SPEC="$R/cd_progressive_s2026_sources.json"
export PD_DATA="$CLEAN_LMDB"
export PD_VB_PY="$R/tools/vbpy"
```

`PD_ROOT` 必须尚不存在；`PD_SPEC` 在其外部。`OF_PY` 为训练环境 Python，`PD_DATA` 是 clean LMDB，不是 CD trajectory LMDB。配置和 manifest 生成后不可修改；改动应创建新目录并保留旧结果。

先 CPU 预检：

```bash
CUDA_VISIBLE_DEVICES='' "$OF_PY" -m pytest -q tests/test_iclr_progressive.py
"$OF_PY" -m experiments.iclr.progressive prepare \
  --source-pair "$PD_SPEC" --data "$PD_DATA" --output "$PD_ROOT"
```

准备只核查来源、数据、prompts、配置与代码 hash，**不构成视频质量通过**。GPU 操作默认 dry-run；以下 `--execute` 会检查8卡空闲并创建独立 tmux。已有进程占卡时等待或换时段，不停止其它会话。

### A. 四步起点视频

```bash
"$OF_PY" -m experiments.iclr.progressive preview --phase source \
  --output "$PD_ROOT" --python "$OF_PY" --gpus 0,1,2,3,4,5,6,7 \
  --session pd26_source --execute
```

共2组×2采样器×16例=64视频。在 `previews/source/` 中检查全部配对 MP4，关注首帧质量、后续结构、运动、闪烁、融化和重复静止；同时提供12帧 contact sheet 便于比较。检查完将 `review.template.json` 的副本写入 `reviews/source.json`：保留 `export_sha256`，逐个填写 `usable`、`method` 和有内容的 `notes`。`method` 只能是 `video_playback` 或 `temporal_frames`。确实只看逐时刻画面时必须保留 `dynamic_playback_pending=true`，明确说明未完成动态播放；不得声称“已播放”。机器可解码、分数有限或文件存在不能代替画面检查。

`usable` 指没有画面崩溃，不要求每例语义完美，也不要求 Rectified 的主观质量更高。若某个视频异常，不把 `usable` 改成 true 绕过检查。保留全部视频、说明失败样例和哪种 sampler 异常，先定位起点问题。两组都在确定性四步下可用后进入 B。

### B. 4→2，两组各300步

```bash
"$OF_PY" -m experiments.iclr.progressive train --stage 4to2 \
  --output "$PD_ROOT" --python "$OF_PY" --gpus 0,1,2,3,4,5,6,7 \
  --session pd26_4to2 --execute
```

两组依次占用8卡。训练机 AI 必须分别观察 Original 和 Rectified **至少到第10步**，查看 `logs/pd26_4to2.log`、`training/4to2/<arm>/metrics.jsonl` 和 `step10_health.json`，确认 loss/gradient 有限、teacher 重复检查通过、输入形状和实际更新数正常。不能启动后直接离开，也不能把自动 health 文件冒充人工观察记录。把观察到的 step、数值与时间写入 `operator_notes.md`。

用实际前10步训练耗时估算300步耗时，再加加载/保存时间；两组、下一阶段分别更新估计。不能直接给出“2–3天”。结束后：

```bash
"$OF_PY" -m experiments.iclr.progressive preview --phase 4to2 \
  --output "$PD_ROOT" --python "$OF_PY" --session pd26_preview2 --execute
```

输出32个两步视频。对照相同样例的四步 teacher，逐项记录 `reviews/4to2.json`。未检查或明显崩溃，不进入下一阶段。

### C. 2→1，两组各300步

```bash
"$OF_PY" -m experiments.iclr.progressive train --stage 2to1 \
  --output "$PD_ROOT" --python "$OF_PY" --session pd26_2to1 --execute
"$OF_PY" -m experiments.iclr.progressive preview --phase 2to1 \
  --output "$PD_ROOT" --python "$OF_PY" --session pd26_preview1 --execute
```

**第二条需等第一条训练完成后执行**；launcher 后台返回不等于训练完成。两组仍分别盯到第10步。student 从各自两步 raw 结果初始化，teacher 固定为该两步结果，优化器重置；不能交叉使用另一组权重。32个一步视频检查后填写 `reviews/2to1.json`，再进入 D。

### D. 终点正式 VBench

```bash
"$OF_PY" -m experiments.iclr.progressive evaluate \
  --output "$PD_ROOT" --python "$OF_PY" --vbench-python "$PD_VB_PY" \
  --session pd26_final --execute
```

导出 Original/Rectified 各944个视频，并依次计算全16维 VBench。最终仅这两个条件，无 FFE、无 Qwen、无 EMA；第一帧也只有一次 denoise，生成后的零时刻 cache 写入不属于额外去噪。生成 seed 按原始 prompt 全局序号0..943，两组一致；这是本轮固定协议，不声称逐视频复现旧 shard 的随机流。

最终视频在 `final/<arm>/videos`，rank 导出记录与 checkpoint hash 保存在 `final/`。评测完成可重验：

```bash
"$OF_PY" -m experiments.iclr.progressive report --output "$PD_ROOT"
```

留出预览共128视频，正式评测1888视频，总计2016视频。只对最终点跑 VBench；预览不是额外的 checkpoint 扫分。

## 5. 保留内容与执行边界

保留 source 清单、manifest、代码/配置/数据/权重 hash、所有4个阶段的 step300 raw、完整日志、10步观察笔记、每阶段所有预览 MP4/contact sheet/review、正式1888 MP4、16维原始 JSON 和 `report.json`。失败也保留，不覆盖目录、不自动 weights-only resume、不清理其它任务。自动 launcher 仅给自己子进程设置 `CAUSAL_DISABLE_FLEX_ATTENTION=1`，使用仓库已有 segmented attention fallback；不改 tmux 全局环境。两组同样设置，实际执行环境写入记录。

CPU 验证覆盖合并两步 teacher 的数学等价、teacher 无梯度/无采样 RNG、连续阶段微型优化、所有帧相同 NFE、真实生成帧 cache、来源拒绝规则、数据留出、raw lineage、视频检查门槛、MP4 编解码和原始944 prompts。CPU 合成视频只验证文件管线，**没有验证真实 Wan 输出质量或8卡 FSDP运行**；这些须在训练机执行上面的检查。

本轮当前代码仅新增独立入口和文档；旧 GAN、frame/chunk、1000步 CD 的训练与推理代码保持原样。若 GPU 冒烟发现问题，先记录错误与实际模块路径，修复新增入口后重做对应验证；不静默改预算、seed、loss 或起点。
