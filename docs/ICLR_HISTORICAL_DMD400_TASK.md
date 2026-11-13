# 历史 DMD-only 连续训练 400 步，对照已有 DMD+GAN 83.93

本轮唯一任务：**从历史 ODE 初始化重新训练一个 seed=0 的 DMD-only，连续 400 iterations，然后评 raw / Qwen / 944×5 / 历史 FFE。复用已有 DMD+GAN step400 的 83.93，不新增 GAN 训练。** 用户已经授权执行，不需要再次询问是否训练。不要启动之前的 CD、frame/chunk、seed42 或 GAN300 任务。

## 对照是谁

| 项目 | 历史 DMD+GAN | 本轮 DMD-only |
|---|---|---|
| training seed | 0（历史训练器的固定零，不随机化） | 0 |
| 初始化 | 历史 `checkpoints/framewise/causal_ode.pt` | 同一初始化文件 |
| 预算 | 同一次 600-step 训练中的 step400 | 从 step0 连续训练至 step400 |
| G / critic 更新 | 80 / 400 | 80 / 400 |
| GAN G/D 权重 | 0.03 / 0.03 | 0 / 0 |
| block / 梯度窗口 | framewise / 17 latent frames | 相同 |
| 其它训练设置 | 历史 rebuttal 配方 | 保持相同 |
| 评测 | raw、Qwen、944×5、旧版 FFE | 相同 |
| Total / Quality / Semantic | **83.93 / 85.41 / 78.04** | 待实测 |

历史 GAN checkpoint 后缀：
`train_1step_one_forcing_600/seed_0/checkpoint_model_000400/model.pt`；MD5：`3bb7732b43167b0ad5c2fd0ee4f9f957`。
来源：`54841ba66cda33b77ba018d78f47ee2d052cc224:experiments/rebuttal/results/ICLR2027_APPENDIX_GAN_ABLATION.md`。
旧 step200 的同协议结果是 DMD-only 77.21、DMD+GAN 80.58，保留该表，不用新 step400 替换它。

**不要从 DMD-only step200 权重续训。** 历史 checkpoint 未保存 AdamW 状态及完整随机/数据迭代状态，旧 `resume_ckpt` 只是模型和步数恢复。原 GAN 是连续训练。本轮必须从 ODE 开始，400 是总 iterations，不是 400 次 generator 更新。

**不要切换到官方 main 或本轮 ICLRTrainer。** 83.93 对应历史 rebuttal 训练路线；本轮目标是与该结果做预算匹配。训练快照固定在历史结果记录的 `d0abec6518e7d7dc567da9eb3abbefadab0f89b2`，评测快照固定在上述 `54841ba`。入口会创建两个独立 worktree，不修改正在被其它任务使用的 checkout。

**旧版 FFE 的确切含义**：首块 **4 个 latent frames 联合做四步**，之后每块一帧、一步；不是新版 `main_ffe` 的“仅首个 latent frame 做四步”。本轮保留此设置以对齐 83.93，不能用新版推理结果与它直接配对。

## 先核实来源并准备

在独立 checkout 获取 GitHub `Aurora-edu/One-Forcing` 的 `ICLR` 最新提交。不要在其它 session 的工作目录切分支、reset 或拉取覆盖文件。

1. 找到历史训练目录中的 `resolved_config.yaml`、`launch_command.txt`、`run_metadata.json`、初始化/训练数据/prompt-cache 路径，核对并记录到本轮 `operator_notes.md`。优先搜索 `/data/fengjiaqi/`、现机已有实验目录和历史挂载；旧记录曾使用 `/data/banyuanhao/iclr2027_ckpts/` 以及 `/mnt/lustre/lusasias2/yuanhao_arena_ai/One-Forcing/`。它们是检索线索，不是必须存在的现机路径。
2. 复用同一 `causal_ode.pt`、Wan1.3B、Wan14B teacher、clean-latent LMDB 和训练 prompt embedding cache。核对能恢复的 hash/配置/数据集身份。路径可以改映射，不能换成最近 CD 权重、公开 `one_forcing.pt` 或不同训练数据。若旧 resolved config/launch 记录找不到，可按固定快照的历史 README 恢复，并明确记录哪些来源未能独立核验；不要把“配置恢复”写成“原始运行来源全部已验证”。若找到实质冲突，先用历史日志/代码定位，不能静默换训练配方。
3. 找历史 `full400_ffe_paper_protocol_summary.json`、完整 16 维 JSON、视频和环境记录。GAN 原权重不是训练 DMD 的输入，缺它不阻塞这一个训练。已有 JSON 可直接复用；只有 Markdown 时保留报告精度，汇总工具会标记为两位小数参考。
4. 如果 prompt cache 缺失，使用固定训练快照的 `experiments/rebuttal/build_prompt_cache.py` 和同一 Wan T5、同一 LMDB 重建完整 cache。这是输入恢复，不是额外训练。保留 BF16 缓存方式，不切成新的在线编码路线。

CPU 准备示例（变量均替换为本机真实绝对路径）：

```bash
export TASK=/data/fengjiaqi/One-Forcing/historical_dmd400_s0
export OF_PY=/ABS/PATH/train-env/bin/python
export VB_PY=/ABS/PATH/vbench-env/bin/python
export ODE=/ABS/PATH/historical/causal_ode.pt
export TEACHER=/ABS/PATH/wan_models/Wan2.1-T2V-14B
export CLEAN_LMDB=/ABS/PATH/historical/clean_data
export PROMPT_CACHE=/ABS/PATH/historical/prompt_cache/clean_data_umt5_bf16
export WAN_ROOT=/ABS/PATH/wan_models

# 若浅克隆缺历史对象，先获取历史分支；这不会切换当前 checkout。
git fetch origin review iclr2027-rebuttal-gan-ablation-audit

"$OF_PY" -m experiments.iclr.historical_dmd400 prepare --output "$TASK"
"$OF_PY" -m experiments.iclr.historical_dmd400 check --output "$TASK"
"$OF_PY" -m pytest -q tests/test_iclr_historical_dmd400.py

"$OF_PY" -m experiments.iclr.historical_dmd400 train \
  --output "$TASK" --generator "$ODE" --teacher "$TEACHER" \
  --data "$CLEAN_LMDB" --prompt-cache "$PROMPT_CACHE" --wan-root "$WAN_ROOT" \
  --python "$OF_PY"
```

不带 `--execute` 时只打印命令，不查询 GPU、不训练。`prepare` 拒绝已有任务目录；已准备过则用 `check`，不要删除原目录重做。生成的配置与历史 DMD-only 配置只改变总预算 `max_steps=400`；launcher 保留旧训练的 fake-score CPU offload、manual generator backward、generator optimizer-state CPU offload、rank0 preload。不要临时关闭这些路径以“更贴近官方”。

代码/依赖/调度问题由训练机 AI 自行排查，不要停下来要求用户重新授权。B200 需要可用的 CUDA/PyTorch 内核；复用当时 B200 评测环境更合适。记录依赖差异，优先修环境，不修改模型数学、学习率、seed、更新比例、数据或评测设置。不能用更换实验条件冒充环境修复。

## 启动一个训练，观察至少第 10 步

```bash
"$OF_PY" -m experiments.iclr.historical_dmd400 train \
  --output "$TASK" --generator "$ODE" --teacher "$TEACHER" \
  --data "$CLEAN_LMDB" --prompt-cache "$PROMPT_CACHE" --wan-root "$WAN_ROOT" \
  --python "$OF_PY" --gpus 0,1,2,3,4,5,6,7 \
  --session of_hist_dmd400_train --execute

export DMD_RUN="$TASK/runs/historical_dmd_only_400/seed_0"
tail -n 40 "$DMD_RUN/train.log"

# 实际日志到第10步后执行；还需人工记录观察，不能把检查命令当成已经盯过。
"$OF_PY" -m experiments.iclr.historical_dmd400 health --output "$TASK"
```

必须使用 8 张空闲 GPU，global batch=8。独立 tmux 持续运行；Slurm 集群则申请独立 8 卡节点，并在分配节点内执行，遵循当地 tmux/调度方式。其它 session 的进程、GPU、文件全部保留，不能 kill、覆盖或抢占。没有空闲资源就等待/排队，不降到单卡改变 batch。

第10步确认：runtime seed=0；10 次 critic / 2 次 G 更新；loss/grad norm 有限；GAN、R1、R2 损失均为零；从 ODE 开始而非 resume。把观察时间、实际数值、source commit 和资产来源写入 `operator_notes.md`。训练过程中保留所有日志和 100/200/300/400 步 checkpoint，仅 400 步做新增完整 VBench。

若前置检查失败且训练尚未创建 run 目录，可修复环境后重试。若训练途中中断，由于历史代码没有完整状态恢复，保留失败输出，使用新任务目录从 ODE 连续重跑，不能拼接两个 200 步片段冒充连续 400 步。不要因预览分数或效果不符合预期而反复重训挑选结果。

训练结束：

```bash
"$OF_PY" -m experiments.iclr.historical_dmd400 health --output "$TASK" --final
```

## 先看视频，再做完整评测

```bash
"$OF_PY" -m experiments.iclr.historical_dmd400 preview \
  --output "$TASK" --python "$OF_PY" --session of_hist_dmd400_preview --execute
```

预览是固定 8 个 prompt、各 sample0，保留正式评测的 rewrite 和 noise seed，输出在 `$TASK/preview/videos/`。检查全部视频的开头、中段、结尾以及动态过程，至少包括人物/动物、运动、多物体场景；若固定八例未覆盖，则追加对应正式 manifest 样例做观察，不能修改正式 manifest。记录是否模糊、闪烁、身份丢失、冻结或画面崩溃。程序完成标记不等于视频正常。能找到历史 GAN 视频时，查看相同 prompt/sample 的对照。

确认导出路径正确后启动完整评测：

```bash
"$OF_PY" -m experiments.iclr.historical_dmd400 evaluate \
  --output "$TASK" --python "$OF_PY" --vbench-python "$VB_PY" \
  --session of_hist_dmd400_vbench --execute
```

固定 **4720 条视频，81 RGB frames、16fps、全部16维、raw generator、zero attention sink**；Qwen 仅用于生成 conditioning，评分用原始 VBench 文本。推理入口来自独立历史 evaluation worktree。查明软件错误可修复外围环境后重试；若确认权重本身产生差视频，保留视频并如实评测、报告，不改训练设置追求某个差值。

## 汇总与交付

```bash
# 找到历史原始 summary 时优先使用，工具重算16维并核对协议/来源。
"$OF_PY" -m experiments.iclr.historical_dmd400 report \
  --output "$TASK" \
  --reference-summary /ABS/PATH/full400_ffe_paper_protocol_summary.json

# 如果只找到了历史 Markdown 报告，省略 --reference-summary。
# 工具会明确标记参考值为四舍五入值，不会伪造原始16维或完整精度。
```

交付 `comparison.json`、`comparison.md`、新的完整16维结果及 protocol summary、`task.json`、`launch.json`、`training_audit.json`、`operator_notes.md`、resolved config、训练日志和代表视频。保留全部原始结果和 checkpoint。明确本轮训练/评测成本、200/400 步各自的来源、历史数据/初始化/环境对齐情况；不需要再申请是否汇总的确认。

本轮 CPU 测试验证的是来源/配置锁定、命令、连续更新检查、checkpoint 步数与结果汇总，不能写成已在本机验证真实 Wan 训练或视频。训练机需完成实际第10步观察和视频检查。报告正负差值都保留，最终比较固定为 **400 vs 400**。

主线程验证记录（2026-11-13）：`test_iclr_historical_dmd400.py`、`test_paper_vbench_protocol.py`、`test_iclr2027_priority.py` 合计 **18 passed**，另通过 Python 编译检查。没有启动 GPU 训练或生成视频，也没有停止其它 session。
