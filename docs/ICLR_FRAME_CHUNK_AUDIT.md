# 现有 frame/chunk 实验核查（2026-11-08）

本任务核查上一轮四个第 200 步 run。使用现有 checkpoint、日志和导出产物，不新增训练，不改变科学配置，也不补跑中间 VBench。原配置 CD 任务按其独立说明继续；GPU 被该任务或其它 session 占用时等待，不能中断它们。

## 先核对旧实验的实际设置

作者此前观察到 chunkwise 在 200 步、无 R1/R2 时失稳。确认那次试验是从 `chunkwise_config.yaml` 关闭正则，还是从 `config.yaml` 只改 block=3。仓库旧 chunk 配置的初始化路径、判别器 layers=[29] / 1 register / FFN=512，与本次 main 对齐的共享 framewise 初始化、layers=[21,29] / 2 registers / FFN=2048 不同；文件路径不同并不能证明权重内容不同，需要实际 hash。

本次还有两个已知联动：gradient_num_blocks=17 对应 frame 17 帧、chunk 21 帧；main_ffe 增强 frame 的第一帧、chunk 的前三帧。它们应被披露，不能在没有直接对照时认定为某次分差的原因。

另需区分 raw 训练状态与 EMA 评测：当前代码第 50 步创建 EMA，之后只在 G 更新时维护。第 200 步前通常只有 30 次 EMA 更新；decay=0.99 时，参数中的初始 shadow 系数为 0.99^30≈0.74。这是现有 main 设置的名义加权，不是运行错误，也不能据此认定视觉表现来自某个 checkpoint；需用下述 raw/EMA 直接对照确认。

## 只读核查命令

更新 `ICLR` 并保留训练机已有修改，沿用训练环境。以下命令在 CPU 上读取现有权重的 metadata 和少量确定性 tensor samples，不初始化 Wan、不使用 GPU、不修改源产物。用新输出目录，避免覆盖。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
"$OF_PY" -m experiments.iclr.audit_frame_chunk \
  --root "$ICLR_ROOT" --output "$ICLR_ROOT/frame_chunk_audit_v1" \
  --check-weights --prepare-visuals
```

按 `audit.json` 核对：

- 实际 resolved_config、checkpoint complete.json 的 seed/block/profile，以及 step=200、40 G / 200 critic updates；四组配置差异及初始化 hash。
- 实际 model.pt 的步数、raw/EMA keys 和 shapes，少量权重 samples 的 fingerprint。样本 hash 相同只说明需要检查完整权重，不能当作完整内容相同的证明。
- export.intent.json、实际 shard 推理 YAML、checkpoint 路径、EMA 标记和真正使用的 prompt/rewrite/noise manifests。
- 原始 VBench JSON 的 16 维分数与汇总 CSV 是否一致，并输出 `score_contributions.csv`，分解每个维度对官方 Total 差值的贡献。该分解不改变官方指标。

`issues` 表示发现矛盾；`incomplete` 表示缺原始产物；`metadata_checked` 只说明核对的 metadata 一致，不能证明视频动态正常、训练全程稳定或不存在实现问题。若目录因恢复训练发生变化，先找到真实运行目录和 checkpoint，再按其记录核对，不通过改标签来让审计通过。

## 64 个终点视觉诊断样例

`--prepare-visuals` 只生成脚本与诊断推理配置，不启动 GPU。固定 `prompts/demos.txt` 前 4 条、相同噪声 seed 流，比较四个模型的 raw/EMA × all1/main_ffe，共 64 条视频，只用现有第 200 步权重。与现有 shell 的 all1 设置不同，此诊断直接调用 exporter，保留相同 attention/sink 配置，只改变是否启用首块四步，以便解释 FFE 的影响。

先检查四组实际配置和权重标签，没有错误后再导出；在空闲 GPU 上于独立 tmux 中运行：

```bash
FC_AUDIT_GPU=0 bash "$ICLR_ROOT/frame_chunk_audit_v1/export_visuals.sh"
```

每个导出前会检查指定 GPU 是否被占用，存在其他进程时退出等待，不停止它们。脚本顺序导出，不做 VBench。实际播放同 prompt 的四种模式，分别记录 raw 是否崩溃、EMA 是否掩盖不稳定、FFE 是否改变结论，以及闪烁、冲白、主体/背景结构和运动。保留 MP4；不能只依据帧差或静态拼图宣布视频稳定。

## 交付

打包 audit.json、score_contributions.csv、实际四组 resolved_config/initialization/complete/checkpoints.jsonl/metrics.jsonl、export intent 与 shard 推理配置、四组原始 VBench JSON、同 prompts 的实际诊断 MP4 和检查记录。报告错误发生在哪个文件或哪个设置，保留所有 seed；没有证据前不把分数反向视为实现错误，也不以提高某一方法得分为目标调整训练。
