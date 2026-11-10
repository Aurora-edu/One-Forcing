# 1000步端点回归实验：将 all1 评测替换为 FFE

本任务对应7月旧实验 `curvature_causal_intervention`，使用
`experiments/rebuttal/train_curvature_ode.py` 和
`experiments/rebuttal/configs/train_curvature_ode.yaml`。它不是300步相邻 CD，
不是300步端点锚定 CD，也不是局部渐进拉直实验 B。

用户只要求改变原 all1 的推理 schedule。训练沿用原配方：Original / Global
Rectified 各1000 updates、seed=2026、每卡 batch=1、8卡、21 latent frames、
lr=2e-6、四个噪声输入时刻随机采样、直接回归保存的 teacher ODE 端点、真实历史
teacher forcing。不得增加相邻 CD/EMA 目标、额外损失或最高噪声采样权重。

新评测使用最终 step-1000 **raw generator**；同一份944个 original prompts、
每个prompt一个样本、原始 seed manifest、21 latent frames、16 fps、全部16维。
模型是 framewise 自回归：每次生成1个 latent frame，历史由此前自身生成的帧组成。

唯一推理变化是 `--schedule main_ffe`：首个 latent frame 做4次去噪，后续20帧各1次。
首帧伪时间为 `[1000,750,500,250]`，shift=5 后实际时间为
`[1000,937.5,833.3333,625]`；后续帧从1000一步预测。`local_attn_size=21`、
`sink_size=0` 保持原设置。**不要使用 `--schedule ffe`**，它会把首个 block 改成
4个 latent frames，不符合这次“其它不变”的要求。结果标为 FFE，不能标为 all1。

## 执行

优先直接复用原来两组1000步权重和 manifests，不重训。旧机器根路径记录为：

```text
/mnt/lustre/lusasias2/yuanhao_arena_ai/One-Forcing/eval/followup/curvature_control
```

迁移后目录应包含 `training/{curved,rectified}/checkpoint_model_001000/model.pt`、
`manifests/vbench_prompts.txt`、`manifests/vbench_single_sample_seed0.jsonl`。
本地仓库只归档了历史分数，没有这两份权重。若训练机也没有，先核实原初始化和
轨迹数据来源，再用上述原训练入口重建；不得把11月的300步权重当作替代品，或
擅自用另一份初始化声称只改了FFE。重建训练须独立tmux，每组观察至少前10步。

沿用训练机已验证的推理环境和 VBench wrapper。在仓库根目录设置真实路径后，
先用默认 dry-run 核对命令：

```bash
"$OF_PY" -m experiments.rebuttal.evaluate_curvature_1000_ffe \
  --experiment-root "$CURVATURE_ROOT" \
  --full-info-path "$INFO" \
  --python "$OF_PY" --vbench-python "$VB_RUNNER" \
  --gpus 0,1,2,3,4,5,6,7 --dry-run
```

确认两个权重的来源和完整性后，在空闲GPU上、独立tmux内，将 `--dry-run` 换成
`--execute`。入口复用原始prompt/noise manifest，检查step=1000和raw权重，并拒绝
占用已有计算进程的GPU。只新增两组FFE评测，不启动训练，不重新运行原四步对照。

输出独立保存到 `$CURVATURE_ROOT/vbench_main_ffe/`，已有all1/all4结果保留。
入口拒绝覆盖已有输出；中断后的再次执行可用 `--output` 指定一个新的空目录，保留旧记录。
其中 `curvature_main_ffe_summary.json` 汇总两组全16维及 Rectified−Original。
原来的all4对照继续使用原结果；若换过权重或数据，必须另记为新训练结果，不能
与旧四步分数混成同权重对照。

必须检查导出的实际视频，包含首帧与后续帧衔接、运动和画面是否崩溃。只有静态
抽帧时标明检查范围，不声称已播放。交付原始VBench结果、export/protocol记录、
汇总及同prompt视频。此次仅改变评测，不增加新的训练超参数搜索。

本机已做CPU验证：dry-run不创建输出，两组复用同一manifest且均为raw/step1000，
改动noise seed会被拒绝；运行真实shell的配置生成逻辑（GPU导出替换为占位程序），
确认相对all1只增加首帧schedule，pipeline的21个block仍均为1帧、NFE为`[4,1,…,1]`。
未在本机加载实际模型或执行GPU评测，视频检查留给训练机。

## 追加上传：旧 seed=0 frame/chunk 的完整训练日志

本轮上传 Google Drive 的结果压缩包还须包含以下两个旧 run 的原始文件，用于重绘仅含 seed=0 的 framewise/chunkwise 学习曲线：

- `runs/frame_roff_s0/metrics.jsonl`
- `runs/chunk_roff_s0/metrics.jsonl`

从旧实验根目录 `$ICLR_ROOT`（此前为 `/data/data/cuijiaxing/oneforcing_iclr/iclr2027_main_200`）读取，按上述相对路径放入结果包。两份日志均应覆盖完整的第 1–200 步，保留所有原始字段和数值；不要用 seed=42、smoke 日志或曲线 PNG 替代。这里只补交已有日志，无需重新训练或评测，也不改变本轮 CD 设置。

打包后检查压缩包确实含有这两个文件，并在交付说明中列出各自的记录数和起止 step。如果原日志缺失，明确记录缺失项和已查找的路径，不从图片反推数值。若现成的 `train_curves/formal_A.csv` 仍在，也一并附上，但不替代上述原始日志。
