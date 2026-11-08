# Framewise / chunkwise：18 帧反传窗口、seed 42

这是本轮训练机任务。只新增下面 **2 个独立训练 run**，不重跑旧 seeds 0/1 或 CD。原计划第三组「公开 One-Forcing 对应配方独立训练 chunkwise、original prompts 评测」已由用户取消；不要自行恢复。

| Run | 每 block latent 帧数 | 实际反传窗口 | training seed | R1/R2 | 正式训练 |
|---|---:|---:|---:|---|---:|
| `frame_w18_s42` | 1 | 最后 18 帧 | 42 | 0 / 0 | 200 iterations |
| `chunk_w18_s42` | 3 | 最后 18 帧（6 个完整 block） | 42 | 0 / 0 | 200 iterations |

两组都设置 `gradient_num_blocks=0, gradient_num_frames=18`，rollout / score 仍为 21 latent 帧，前 3 帧无梯度。仅 `experiment_id` 和 `num_frame_per_block` 在两组之间不同。专用 profile 为 `matched_window18_s42_v1`；默认 CLI 仍指向旧四组，所以生成时必须显式传入新 profile。

共同保留 main 配方：同一 raw ODE 初始化、14B teacher、clean-data LMDB、判别器 layers [21,29] / 2 registers / FFN 2048、batch 1/GPU、G:D=1:5、EMA=0.99（第 50 步初始化）、训练 all1。200 iterations 对应 40 次 G update、200 次 critic update。训练 seed 固定为 42、不随机化；sampler 沿用 main 的 seed 0，评测噪声 seeds 沿用 pinned manifest，三者不能混为一谈。

**只评第 200 步，EMA + `main_ffe` + pinned Qwen prompts，944×5 samples，全 16 维 VBench。** 不加 raw、all1、all4、original prompts 或中间 checkpoint 的 VBench。保留 0/50/100/150/200 保存点供故障排查，它们不代表 5 次评测。`main_ffe` 在第一整个 native block 用四步，之后一步：frame 第一块是 1 latent 帧、chunk 是 3 帧；本轮保留这个既有推理定义。不能改用 review 的 first4then1 `ffe`。

本轮控制梯度帧数和更新次数，记录各自实际耗时；不能称为严格相同 wall-clock compute。一个 training seed 只报告一次配对差值，不估计 seed 方差，也不能仅凭最终 EMA 分数认定整个训练过程稳定。

## 1. 接手与生成配置

使用 GitHub `Aurora-edu/One-Forcing` 的最新 `ICLR` 分支。保留训练机已有修改、其他 session 和旧实验目录；禁止 kill/pkill 或抢占其他任务的 GPU。沿用已跑通的 B200 训练/VBench 环境及 `tools/vbpy`，不要为了本轮重装依赖或改评分实现。

下面路径来自上轮训练机，先确认存在；若机器路径已变，只映射路径。不要下载新的初始化来替换已审核资产。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export PAIR_ROOT="$R/iclr2027_window18_s42"
export VB_RUNNER="$R/tools/vbpy"

"$OF_PY" -m experiments.iclr.make_configs \
  --profile matched_window18_s42_v1 \
  --generator-ckpt "$ODE_INIT" --teacher "$TEACHER" --data "$CLEAN_LMDB" \
  --output "$PAIR_ROOT/configs"

"$OF_PY" -m experiments.iclr.launch \
  --matrix "$PAIR_ROOT/configs/matrix.json" --output "$PAIR_ROOT/runs" \
  --python "$OF_PY" --dry-run

"$OF_PY" -m experiments.iclr.train \
  --config "$PAIR_ROOT/configs/frame_w18_s42.yaml" \
  --output "$PAIR_ROOT/runs/frame_w18_s42" --dry-run
"$OF_PY" -m experiments.iclr.train \
  --config "$PAIR_ROOT/configs/chunk_w18_s42.yaml" \
  --output "$PAIR_ROOT/runs/chunk_w18_s42" --dry-run
```

计划必须恰好 2 个 run；两组 `effective_gradient_frames` 都为 18。配置目录已有内容时生成器会拒绝覆盖；不要复用旧目录或旧权重继续训练。各自从同一 raw ODE 独立开始。源码 CPU 回归包括真实 temporal rollout 的梯度检查，可运行 `python -m pytest -q tests/test_iclr_window_pair.py`；这里的 toy 验证不等同于 Wan GPU 验证。

## 2. 8 卡正式训练

仅在 8 张卡均空闲时启动；启动器会检查占用，并创建独立 tmux。逐组执行，第一组结束、GPU 释放后再启动第二组：

```bash
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$PAIR_ROOT/configs/matrix.json" --output "$PAIR_ROOT/runs" \
  --run frame_w18_s42 --python "$OF_PY" --session iclr_w18_frame42 --execute

# 第一组完成后执行：
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$PAIR_ROOT/configs/matrix.json" --output "$PAIR_ROOT/runs" \
  --run chunk_w18_s42 --python "$OF_PY" --session iclr_w18_chunk42 --execute
```

每组至少盯到第 10 步，检查 `metrics.jsonl` 的 loss / grad norm 有限、`runtime_seed.txt` 为 42、`resolved_config.yaml` 为新 profile/window，并核对实际消耗和输出磁盘空间。每组每 5 步含一次 G update，估时要覆盖完整周期和 checkpoint I/O，不能只用 critic 步估算。在正式训练头 10 步完成本轮 GPU 冒烟验证，不另外增加正式训练组。

不要在失稳、OOM 或非有限数后自行调正则、窗口、EMA 或换 seed；保留失败日志并报告。完整恢复只能使用该 run 的完整 checkpoint，启动器要求新的输出目录；smoke 或旧四组 checkpoint 不得充当正式初始化。任一组实际失败应保留为本轮结果，不能通过多跑挑选较高分。

两组完成后：

```bash
"$OF_PY" -m experiments.iclr.audit_initialization \
  --runs "$PAIR_ROOT/runs" --profile matched_window18_s42_v1
```

必须确认两组实际加载的 raw generator、初始 critic tensor hashes 一致。检查 `training.done` 均为 step=200、generator_updates=40、critic_updates=200；独立训练不是互相复用 checkpoint。

## 3. 终点评测

先准备与旧主评测完全相同的 pinned inputs：

```bash
"$OF_PY" -m experiments.rebuttal.paper_vbench_protocol prepare \
  --output_dir "$PAIR_ROOT/manifests" --samples_per_prompt 5
```

下面整段在独立 tmux 中顺序运行，两组训练都结束且 8 卡空闲后执行。`PAIR_ROOT`、`OF_PY`、`VB_RUNNER` 在该 tmux 内也需设置好。

```bash
for run in frame_w18_s42 chunk_w18_s42; do
  "$OF_PY" -m experiments.iclr.evaluate curve \
    --index "$PAIR_ROOT/runs/$run/checkpoints.jsonl" \
    --config "$PAIR_ROOT/runs/$run/resolved_config.yaml" \
    --prompts "$PAIR_ROOT/manifests/vbench_prompts.txt" \
    --extended-prompts "$PAIR_ROOT/manifests/vbench_qwen_rewrites.txt" \
    --manifest "$PAIR_ROOT/manifests/vbench_qwen_5sample_seed0.jsonl" \
    --full-info experiments/rebuttal/assets/qwen_vbench/VBench_full_info.json \
    --samples-per-prompt 5 --output "$PAIR_ROOT/evaluations/$run" \
    --python "$OF_PY" --vbench-python "$VB_RUNNER" \
    --gpus 0,1,2,3,4,5,6,7 --execute || break
done

"$OF_PY" -m experiments.iclr.summarize_window_pair \
  --evaluations "$PAIR_ROOT/evaluations" --output "$PAIR_ROOT/summary"
```

虽然子命令仍叫 `curve`，这里只选第 200 步，不评中间点。执行前会核对保存的配置、实际更新次数、checkpoint metadata 和 EMA 是否存在；不能静默回退 raw。汇总还会核验实际 `export.intent.json`、`export.done` 和完整 16 维评分记录。旧 `experiments.iclr.summarize` 专用于 seeds 0/1 四组，不能用于本轮。

## 4. 视频检查与交付

从正式导出的 MP4 中预先固定至少 16 个相同 prompt/sample pairs，覆盖人物、物体、镜头运动和大幅动作，实际观看两组完整视频；检查结构崩溃、白屏/花屏、闪烁、冻结、运动差异。保留具体文件名、观看方式和判断。若当前工具不能检查动态视频，明确标注未完成，不把 contact sheet 或正常指标写成「视频正常」。不要只挑高分或好看的样例。视频不正常也要按原设置如实交付结果。

交付两组配置、seed、commit/环境、初始化核验、训练日志与曲线、实际耗时、最终 raw/EMA checkpoint 路径、评分原始 JSON、导出 provenance、视频路径与观看记录，以及 `summary/points.csv`、`paired_comparison.csv`、`protocol.json`。结果表列出两组 Total/Quality/Semantic 和各维度原始数值、frame−chunk 配对差值；保留 score 是 0–1 还是百分制的单位。不要只交截图或均值，也不要把本轮和旧四组混算。

## 第三组取消的核查依据

2026-11-08 检查公开 `JiaqiFeng/OneForcing` revision `09c02046ca08c12e8a6a048276a939c749767540` 的 `checkpoints/one_forcing.pt`，LFS SHA256 为 `2214c9c37d0dbda936303f6c1672b016f18cb238917b9200559ef5d6001e93d5`。安全扫描其完整 `data.pkl`（483654 bytes）可见 step=200、generator、critic、generator_ema，没有 training seed/config/RNG；模型卡也未提供 seed。旧 `main@c9a2350` 在 `config.seed == 0` 时随机产生 seed 并广播，故配置的 0 不能证明真实 training seed。用户随后明确回复「那先不做第三组了」。
