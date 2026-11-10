# 本轮训练机任务：CD 1000步权重补测 FFE + 官方 frame/chunk seed42

本文件是本轮唯一总入口，代码分支为 `Aurora-edu/One-Forcing:ICLR`。用户已确认：CD 优先复用已有1000步权重，**找不到就按原配方重训，再评 FFE**；frame/chunk 使用完全未修改的官方代码，两组 training seed=42、评 raw。不要按旧总任务文件追加实验。

## 本轮清单

| 条件 | 训练 | 最终权重 | 评测 |
|---|---|---|---|
| CD Original（目录名 curved） | 优先复用；缺失时按1000步配方重训 | step1000 raw | original prompts，944×1，main_ffe |
| CD Global Rectified | 优先复用；缺失时按1000步配方重训 | step1000 raw | original prompts，944×1，main_ffe |
| 官方 frame | 独立训练，seed42，200 iterations | step200 raw | Qwen prompts，944×5，官方首块增强 |
| 官方 chunk | 独立训练，seed42，200 iterations；只改 block=3 | step200 raw | Qwen prompts，944×5，官方首块增强 |

共 **2–4个新训练 run、4个评测条件、11,328条评测视频**：两份CD权重都可复用时只新增2个训练，CD两组均需重训时为4个。均跑全16维 VBench，只测所列最终点。CD 原有或重训的 training seed 均为2026；frame/chunk 为42。CD 此处指历史 `curvature_controlled_causal_intervention` 的端点回归实验，报告中写明训练目标和 FFE，不能标成纯 all1 相邻 CD。

本轮不跑 DMD-only/GAN 新消融、300步 CD、四步 CD 重评或其它 seeds。已交付的 seed0 日志不用再次收集。

## 接入与准备

在独立 ICLR checkout/worktree 接入本次提交，先读适用的 AGENTS.md。保留其它 session 的进程、工作区修改和训练输出；不要对它们 reset/pull 或清理。数据优先查 `/data/fengjiaqi/`，训练机旧资产位置为 `/data/data/cuijiaxing/oneforcing_iclr/`，按实际路径映射。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export VB_RUNNER="$R/tools/vbpy"
export OFFICIAL_ROOT="$R/official_main_c9a2350_frame_chunk"
export OFFICIAL_PAIR="$R/official_frame_chunk_raw_s42"
export WAN13="$R/assets/wan_models/Wan2.1-T2V-1.3B"

CUDA_VISIBLE_DEVICES='' "$OF_PY" -m pytest -q \
  tests/test_curvature_1000_ffe.py tests/test_iclr_official_frame_chunk.py
```

沿用已验证的 B200 推理/训练环境及 VBench wrapper，归档实际版本、wrapper 源码和 hash。本机上述17项 CPU 检查已通过，包括真实 shell 的配置生成检查；尚未进行真实 Wan/B200 训练或视频生成验证。

## 任务一：CD 1000步权重的 FFE 评测

先阅读 [CD 1000步 FFE 详细说明](ICLR_CURVATURE_1000_FFE_TASK.md)。找回合格的1000步配对权重就直接评测；缺失时执行 [1000步 CD 重训说明](ICLR_CURVATURE_1000_RETRAIN_TASK.md)，重训已获用户授权，不需再次确认。只缺一组且能证明初始化、数据、配置和运行条件一致时补训缺失组；否则两组用同一份已核实资产重新配对训练。不能因旧权重缺失而跳过 CD，也不能拿300步权重替代。

设置 `CURVATURE_ROOT` 为真实实验根目录，必须包含：

```text
training/curved/checkpoint_model_001000/model.pt
training/rectified/checkpoint_model_001000/model.pt
manifests/vbench_prompts.txt
manifests/vbench_single_sample_seed0.jsonl
```

历史根路径是 `/mnt/lustre/lusasias2/yuanhao_arena_ai/One-Forcing/eval/followup/curvature_control`；迁移后路径可以不同。核对两份 checkpoint 内 `step=1000`、`generator` 和 `experiment_metadata`（experiment、intervention、seed、初始化和数据来源），以及原训练 run.intent/training.done；记录 checkpoint SHA256。缺少来源证据时明确标注，不只凭文件名认定是旧权重。

两组保持原始同一套 prompt/noise manifest，只把原 all1 改为 **main_ffe：第一 latent frame 4步、后续20帧各1步**。不改成首4帧联合生成的旧 `ffe`。

```bash
"$OF_PY" -m experiments.rebuttal.evaluate_curvature_1000_ffe \
  --experiment-root "$CURVATURE_ROOT" --full-info-path "$INFO" \
  --python "$OF_PY" --vbench-python "$VB_RUNNER" \
  --gpus 0,1,2,3,4,5,6,7 --dry-run
```

dry-run 应只有两组 raw/step1000 审计、两组 main_ffe 评测和一次汇总，没有训练命令。8卡空闲后，在独立 tmux 内用同一命令将 `--dry-run` 改为 `--execute`，将 stdout/stderr 保存为日志。默认输出 `$CURVATURE_ROOT/vbench_main_ffe/`，已有输出保留；中断后使用 `--output` 指定新空目录。不要一边评 CD 一边用同一批卡训练。

完成标记为 `curvature_main_ffe_summary.json` 加两组完整的原始16维结果及 export/protocol 记录。固定检查 prompt indices `0,59,118,…,885`（16项）、sample=0 的两组实际视频，记录运动、结构、闪烁、冻结和首帧衔接。保留对应 MP4。

## 任务二：官方 frame/chunk seed42

完整执行 [官方 frame/chunk 详细说明](ICLR_OFFICIAL_FRAME_CHUNK_TASK.md)：prepare/check → frame train → chunk train → frame evaluate → chunk evaluate → report。每个阶段完成并释放GPU后再启动下一个阶段，不能把 tmux 创建成功当成训练结束。

关键设置：官方 `main@c9a2350e8740531562011fc9618e1a928d911ae0` 的训练和生成代码逐文件零修改；两组共同覆盖 training seed=42，唯一组间配置差异为 block=1/3；200 iterations，即40次G更新和200次critic更新。评测仅用 raw，Qwen944×5。沿用官方反传窗口17/21帧以及首块1/3帧的联动，不改回之前的18帧方案。

独立 tmux 启动每组训练，**每组至少盯到第10步**，检查实际日志中的 seed、loss/梯度和 GPU 状态。训练完成须有各自的 `training.done`，之后只评 step200。实际检查入口固定的16对视频；异常结果保留，不自行换seed、切EMA或延长训练。详细文件中的命令负责启动、监控、评测和汇总，本轮不调用旧的 `experiments.iclr.train`。

## 交付

逐项报告四个条件的完成状态，不省略 CD。汇总分别提供 CD 的 Original/Rectified 及差值、frame/chunk 及差值；列明 Total/Quality/Semantic 和全部16维，百分制与0–1分数明确区分。

轻量结果包包含本轮实际 Git commit、命令、配置、环境/资产来源和 hash、所有新训练的完整日志与step10记录、CD权重复用或重训的来源记录、四条件原始VBench JSON与export/protocol、汇总，以及固定视频样例和检查记录。必须实际查看生成视频；若工具只能抽帧，写清已检查的范围并将完整动态检查标为待完成，不能用解码成功代替观看。模型权重、LMDB和全量视频留在训练机，列出路径。

使用训练机已有方式将轻量结果包上传到之前的 Google Drive 结果位置，保持原共享范围，返回可访问链接及压缩包 SHA256。上传不可用时给出本地包路径、hash和具体原因，不声称已上传。任何缺失项或失败阶段单列，保留日志；其余独立任务继续完成。
