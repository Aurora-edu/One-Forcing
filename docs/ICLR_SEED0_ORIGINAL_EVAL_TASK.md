# 旧 seed=0 frame/chunk：复用权重评测 original prompts

这是 [本轮合并任务](ICLR_CURRENT_EXPERIMENTS_TASK.md) 的子任务 C。**不训练、不续训**；复用上一轮 `frame_roff_s0` / `chunk_roff_s0` 已完成 Qwen VBench 的同一份第 200 步权重，只新增 original prompts 评测。不要用 seed=42 的新权重或公开 `one_forcing.pt` 替换。

| 设置 | 本次两条件 |
|---|---|
| 模型 | 旧 `frame_roff_s0` / `chunk_roff_s0`，training seed=0 |
| Checkpoint / 权重 | 原 Qwen 导出实际使用的 step-200 `model.pt`，`generator_ema` |
| 推理 | framewise / chunkwise 各按原 native block，`main_ffe`，21 latent / 81 RGB frames，16 fps |
| Conditioning | original VBench prompts，取消 Qwen rewrite |
| 样本和噪声 | **944×5**；prompt/sample/noise seeds 与旧 Qwen 评测逐条对应 |
| 评分 | 全 16 维，相同 B200 VBench 环境与 wrapper |

与旧 Qwen 评测比较时只改变 conditioning。这里继续用 5 samples/prompt，不套用此前公开权重 original-prompts 的 944×1 协议，也不套用 CD 的 raw/all1 协议。旧训练 `gradient_num_blocks=17` 对应 frame 17 / chunk 21 个梯度帧，保持历史事实，不把旧权重标成 18 帧匹配窗口实验。训练 seed=0 也不是评测所有视频都使用 noise seed=0：本协议仍按 `prompt_index*5+sample_index` 配对。

## 1. 找回并核对实际 Qwen 导出来源

沿用上一轮训练机路径；先保留修改并更新 `ICLR`，阅读 AGENTS.md。只读取已有配置和权重，不重跑旧训练。路径通常如下：

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export SEED0_ORIGINAL_ROOT="$ICLR_ROOT/seed0_original_5sample"
export SEED0_VB_PY="$R/tools/vbpy"
export SEED0_FRAME_CKPT="$ICLR_ROOT/runs/frame_roff_s0/checkpoint_model_000200/model.pt"
export SEED0_CHUNK_CKPT="$ICLR_ROOT/runs/chunk_roff_s0/checkpoint_model_000200/model.pt"
export SEED0_FRAME_CONFIG="$ICLR_ROOT/runs/frame_roff_s0/resolved_config.yaml"
export SEED0_CHUNK_CONFIG="$ICLR_ROOT/runs/chunk_roff_s0/resolved_config.yaml"
```

执行前从旧评测目录的 `iclr_cell.json`、`videos/export.intent.json` / `export.done`、实际 shard YAML 和原始 VBench JSON 核对：

- 旧 cell 为 `frame_roff_s0` / `chunk_roff_s0`、seed=0、step=200、EMA、main_ffe、Qwen 944×5，旧评分完成 16 维。
- **以旧 export.intent.json 中实际 checkpoint_path 和 config_path 为准**。若与上方惯用路径不同，找到该实际源文件并设置对应变量；恢复训练目录存在不代表当时评测使用了它。核对 checkpoint 的 step / EMA key、complete.json 中的 seed/block/profile 与原训练配置。不能改 cell 标签来让路径一致。
- 继续使用旧导出的原始训练/推理配置所确定的 attention、sink、context、block 和 FFE 设置。`main_ffe` 保留第一 native block 四步、后续一步；不能换成 first4then1 `ffe`。
- 旧 Qwen manifest 与本次生成的 original manifest 的 prompt_index、sample_index、prompt、seed、output_name 必须逐条相同，只去掉 extended_prompt 与 rewrite_file_sha256。保留来源文件路径、hash 和核对记录。若发现旧配置/权重与上述标签矛盾，先报告实际情况，不能换一个模型充数。

本机轻量结果包不含这些完整权重和导出记录；实际源路径核实由训练机完成。缺权重时记录缺项，不触发重训。

## 2. CPU 准备与 dry-run

两个模型分别使用独立目录；现有 `prompt_ablation` 会准备相同的 original manifest。`--conditioning original` 只生成/评分 original，不重跑 Qwen；目录中同时存在 qwen.jsonl / rewrites.txt 输入文件不表示会执行 Qwen 条件。

```bash
"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint "$SEED0_FRAME_CKPT" --config "$SEED0_FRAME_CONFIG" \
  --method framewise --schedule main_ffe --weight-source generator_ema \
  --conditioning original --samples-per-prompt 5 \
  --output "$SEED0_ORIGINAL_ROOT/frame_roff_s0" \
  --python "$OF_PY" --vbench-python "$SEED0_VB_PY" --prepare-only

"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint "$SEED0_CHUNK_CKPT" --config "$SEED0_CHUNK_CONFIG" \
  --method chunkwise --schedule main_ffe --weight-source generator_ema \
  --conditioning original --samples-per-prompt 5 \
  --output "$SEED0_ORIGINAL_ROOT/chunk_roff_s0" \
  --python "$OF_PY" --vbench-python "$SEED0_VB_PY" --prepare-only
```

先核对两组 original manifests 一致，并与旧 Qwen manifest 做上述配对检查。相同参数改为 `--dry-run` 只打印命令；每组必须只有一个 original 条件，EMA/main_ffe/5 samples，不出现新的训练命令。

## 3. 在空闲 8 卡的独立 tmux 中评测

在该 tmux 内也设置好上述变量。长评测与 A/B 训练顺序运行，不占用其它 session 的 GPU；入口会检查空闲，不停止进程。

```bash
"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint "$SEED0_FRAME_CKPT" --config "$SEED0_FRAME_CONFIG" \
  --method framewise --schedule main_ffe --weight-source generator_ema \
  --conditioning original --samples-per-prompt 5 \
  --output "$SEED0_ORIGINAL_ROOT/frame_roff_s0" \
  --python "$OF_PY" --vbench-python "$SEED0_VB_PY" --execute

# 上一条成功结束且 GPU 释放后运行：
"$OF_PY" -m experiments.iclr.prompt_ablation \
  --checkpoint "$SEED0_CHUNK_CKPT" --config "$SEED0_CHUNK_CONFIG" \
  --method chunkwise --schedule main_ffe --weight-source generator_ema \
  --conditioning original --samples-per-prompt 5 \
  --output "$SEED0_ORIGINAL_ROOT/chunk_roff_s0" \
  --python "$OF_PY" --vbench-python "$SEED0_VB_PY" --execute
```

共两个新增评测条件、9440 条视频；不新增训练、不重新生成 Qwen 视频、不重跑中间 checkpoint 或 seed=1。已完成并通过协议核验的 original 导出/评分可以复用；换目录前先查清来源，不覆盖旧结果。执行会自动核对实际 export intent、EMA、视频数量和完整评分协议，并生成每组 `prompt_comparison.json` / `prompt_scores.csv`。

## 4. 比较和交付

实际观看至少 16 个固定 prompt/sample pairs 的 frame/chunk 完整 MP4，记录模糊、运动、冻结、闪烁或崩溃及具体文件名。若工具不能检查动态画面，保留明确未完成状态并提供 MP4；不能只根据分数或静态拼图宣布正常。

从旧 Qwen 原始 VBench JSON 和本次 original 原始 JSON 用同一 `official_totals` 计算，输出对照 CSV：每行一个指标，包含 frame_Qwen、frame_original、chunk_Qwen、chunk_original、两个 original−Qwen 差值、Qwen 条件的 frame−chunk，以及 original 条件的 frame−chunk。至少报告 Total / Quality / Semantic 和完整 16 维单次原始值。不要只引用旧汇总百分数，也不要把 5 个生成样本当作 5 个训练 seeds。

交付源 checkpoint/config 的核验记录及路径/hash、两模型各自的原始 VBench JSON / scoring protocol / export intent / export.done、prompt/noise manifests、`prompt_comparison.json` / `prompt_scores.csv`、上述跨条件对照 CSV、同 prompts MP4 和观看记录。C 的结果与 A 的 seed=42、18 帧训练窗口对照分别呈现，不能混算训练 seed 方差。
