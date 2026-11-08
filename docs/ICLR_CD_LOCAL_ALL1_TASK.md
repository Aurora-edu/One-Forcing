# 训练机任务：高噪声状态局部渐进拉直，只评一步

本文件是本次新实验的唯一执行范围。只做 **Original 与 Local-progressive Rectified 的 all1 对照**。不生成四步样例，不运行四步 VBench，不沿用旧 `evaluate cd` 的默认四条件命令。本次是新干预方案，不称为原论文实验的精确复现，也不以“Rectified 必须涨分”作为验收条件。

## 固定设置与唯一干预

训练配置 `experiments/iclr/configs/cd_local_all1.yaml`，profile=`local_progressive_cd_all1_v1`。继承原 9 帧纯 CD：每组 300 updates、global batch=8、AdamW lr=2e-6、betas=(0.9,0.999)、weight decay=0.01、clip=10、target EMA=0.95。endpoint loss=0、holdout=0；四个相邻对按原规则循环，各更新 75 次。target EMA 仅用于训练，最终导出 raw generator。

每个训练 seed 内两组同初始化、同数据、同样本顺序和预算，区别如下：

| 条件 | 轨迹 |
|---|---|
| Original（输出目录 curved） | 原始轨迹，alpha 恒为 0 |
| Local Rectified（输出目录 rectified） | 只对训练窗口的第 1 个中间状态（t=937.5/1000）逐渐做局部线性插值 |

现有 ODE 时刻为 `[1000, 937.5, 833.3333, 625, 0]`，另有独立 clean-history 状态。对 state 1，先计算其两个邻居 state 0 与 state 2 之间在真实时间上的线性插值，再令：

`x_new = (1-alpha) * x_original + alpha * x_local_line`

`alpha=min(optimizer_step/100, 1)`；只修改 state 1 的前 9 latent frames。所有其它状态、全局噪声和 ODE 端点、clean history 以及 frame 9 之后均保持原值。插值在 float32 中计算，再转训练 dtype。不能从已经全局拉直的数据再次做局部插值；新 profile 的 loader 始终读取 curved 原始状态。

这应称为“高噪声**状态**局部渐进拉直”：邻居跨度到 t≈0.8333，不能声称只改变连续区间 t≥0.9。本轮保留 teacher forcing，不宣称消除了训推历史错配；共同生成历史的重训练不属于本次范围。

## 应用更新与环境

本更新已提交到 GitHub 的 `ICLR` 分支。训练机先检查工作区并保留已有改动，再拉取并合并该分支；不得执行破坏性 reset，不覆盖其它 session 的更改。若训练机有额外修复，与远端更新逐处合并。已有 B200 barrier 修复保留一次。若通过离线补丁交付，则先执行 `git apply --check`，通过后再应用；已经包含本次提交时不要重复应用补丁。

沿用已跑通的 B200 训练环境和 VBench wrapper。不要为“跑通”变更科学参数。先运行本次更新的 CPU 测试，再执行 GPU smoke。下面给出上一轮机器的路径；其它机器按真实位置映射。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export CDLP_BASE="$ICLR_ROOT/cd_local_progressive_all1_v1"
export CDLP_DATA="$CD_DATA"
export CDLP_INIT="$CD_INIT"
export CDLP_VB_PY="$R/tools/vbpy"
export CAUSAL_DISABLE_FLEX_ATTENTION=1
export CDLP_SEED=2026
export CDLP_OUT="$CDLP_BASE/seed$CDLP_SEED"
```

正式建议完成 seeds 2026、2027 各一对，共四个 run。先完整完成 seed 2026，再更换 `CDLP_SEED=2027` 和独立输出目录，按相同流程执行。不要同时占用同一批卡，不因某 seed 的分数不理想而跳过它；不自动追加其它 seeds。每个 seed 只有两个最终 all1 评测条件。

## 资产复用与预检

复用已有 21 帧配对轨迹 LMDB，实际训练仅取前 9 帧。新入口检查源 checkpoint SHA256 与数据 manifest 一致、raw 权重、CFG=6、时刻和局部处理位置；通过则不重建 LMDB，也不重新生成整套 6505 条轨迹。历史源模型身份尚未确认时如实记录，不能把 hash 匹配的可复用性当作精确复现证明。

```bash
"$OF_PY" -m experiments.iclr.cd_local \
  --data "$CDLP_DATA" --generator-ckpt "$CDLP_INIT" \
  --seed "$CDLP_SEED" --output "$CDLP_OUT" --preflight
```

## Smoke 与正式训练

空闲 8 卡上先跑两个 10-update smoke，独立 tmux，每组观察到第 10 步。检查 loss/梯度有限、endpoint loss/updates=0、Original alpha=0、Rectified alpha=step/100；第 10 步 alpha 应是 0.1，不是 1。

```bash
"$OF_PY" -m experiments.iclr.cd_local \
  --data "$CDLP_DATA" --generator-ckpt "$CDLP_INIT" --seed "$CDLP_SEED" \
  --output "$CDLP_OUT/smoke" --session "cdlp_smoke_$CDLP_SEED" --smoke --execute
```

smoke 完成后，只用 `--schedule all1` 为两个 step-10 raw 模型导出相同的 4 个 demo prompts 视频。首帧也一步，不使用 FFE。实际检查视频；不能把“成功解码”或静态拼图当作动态验收通过。若代理不能播放，提供实际 MP4 并明确标为待作者动态检查，不伪称已播放。

随后正式训练两组，各观察前 10 步，入口自动创建独立 tmux 顺序执行：

```bash
"$OF_PY" -m experiments.iclr.cd_local \
  --data "$CDLP_DATA" --generator-ckpt "$CDLP_INIT" --seed "$CDLP_SEED" \
  --output "$CDLP_OUT" --session "cdlp_formal_$CDLP_SEED" --execute
```

默认不带 `--execute` 为 CPU dry-run。正式只保存第 300 步；不做中间 VBench。日志逐步记录 alpha 和全 rank 平均局部干预 MSE。完成后 `paired_training_audit.json` 核对全部训练控制项、样本顺序、75/75/75/75 对计数、纯 CD 和实际 alpha 日程。中断不能假装恢复 optimizer/target EMA，须使用新输出根目录重新运行。

## 只测一步的诊断

恢复 holdout=0 后，前 64 行不再保证未参与训练。先由 CPU 工具从**本次实际未使用**、且 prompt 与所有已训练行不重复的样本中，固定选 64 行，不读取模型分数来选样本：

```bash
"$OF_PY" -m experiments.iclr.cd_local_validation \
  --training-root "$CDLP_OUT" --data "$CDLP_DATA" \
  --output "$CDLP_OUT/validation_rows.json" --count 64

"$OF_PY" -m experiments.iclr.cd_diagnostics \
  --training-profile local_progressive_cd_all1_v1 \
  --training-root "$CDLP_OUT" --data "$CDLP_DATA" \
  --validation-rows "$CDLP_OUT/validation_rows.json" \
  --output "$CDLP_OUT/diagnostics" --session "cdlp_diag_$CDLP_SEED" --execute
```

该选择只保证这些固定 300 updates 下未见过，不是永久数据集留出划分；若样本不足就报告，不回退到训练样本。诊断只加载 step-300，所有帧均一步，比较真实历史与自由历史，输出逐帧误差和 `one_step_endpoint_errors.csv`，分别报告首帧、前 9 帧、完整 21 帧。超出 9 帧训练窗口的部分要注明外推。free-history 对固定 teacher-forced 端点的 MSE 是敏感性诊断，不是严格因果分解。

## 只跑两个 all1 VBench 条件

使用相同原始 944 prompts×1、相同 noise manifest、全 16 维、raw generator，无 FFE，首帧也只一步。在独立 tmux 内运行：

```bash
"$OF_PY" -m experiments.iclr.evaluate cd \
  --training-profile local_progressive_cd_all1_v1 \
  --original "$CDLP_OUT/curved/checkpoint_model_000300/model.pt" \
  --rectified "$CDLP_OUT/rectified/checkpoint_model_000300/model.pt" \
  --config experiments/rebuttal/configs/eval_all1.yaml \
  --prompts "$ICLR_ROOT/manifests/original.txt" \
  --manifest "$ICLR_ROOT/manifests/original_seed0.jsonl" --full-info "$INFO" \
  --output "$CDLP_OUT/eval" --vbench-python "$CDLP_VB_PY" --execute

"$OF_PY" -m experiments.iclr.summarize_cd \
  --evaluations "$CDLP_OUT/eval" --output "$CDLP_OUT/summary"
```

执行前 dry-run 应恰好打印两个导出命令，均为 `--schedule all1`。新 profile 的评测入口不会生成四步条件，汇总只要求 `cd_original_all1` / `cd_rectified_all1`，发现混入其它 CD 条件就拒绝。每个 seed 共 1888 条评测视频；两个 seed 共 3776 条。

## 交付与解释

报告每个 seed 的 Original / Rectified 单次分数及配对差值，两 seeds 的均值、样本 SD；n=2 不声称显著性。保留负结果，不按 VBench 修改 ramp、state index、seed 或 checkpoint。若固定历史误差改善而自由生成没有改善，下一步才考虑共同生成历史下的新配对训练；不能直接替换历史却沿用旧条件下的 teacher 目标。

打包 preflight、配置、两组 run.intent/training.done/metrics、paired audit、验证行选择、诊断 CSV/JSON、两条件原始 VBench JSON/protocol/环境偏差、导出校验和 seed manifests、固定同 prompts 的实际 MP4、视频检查记录及 summary。保留已有 B200 wrapper 和修复，记录实际运行的 Git commit；若使用离线补丁，同时记录补丁 hash。不要把大权重或 LMDB 放进轻量结果包。
