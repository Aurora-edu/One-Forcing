# 实验 A 与标准 One-Forcing 的对齐记录

用户确认：GitHub `main` 是主表所用的标准实现，83.76 使用 **EMA 权重**评测。本轮以 `main@c9a2350e8740531562011fc9618e1a928d911ae0` 为基准。

`experiments/iclr/configs/main_reference.yaml` 是该提交 `config.yaml` 的逐字副本，Git blob 为 `1caabde9e340552c4c9e9d39f8f5f90c03321b46`。`base.yaml` 继承它，增加路径、计时、恢复和来源记录；配置校验禁止修改参考配置中的训练超参数。无需切换或覆盖当前 ICLR 分支。

## 对照设置

| 设置 | framewise | chunkwise |
|---|---|---|
| `num_frame_per_block` | 1 | 3（唯一主动改变的模型设置） |
| `gradient_num_blocks` | 17 | 17 |
| 实际参与梯度的 latent 帧数 | 17 | 21，按原实现裁到视频长度 |
| 训练视频 / score 窗口 | 21 / 21 | 21 / 21 |
| EMA | 0.99，第 50 步开始维护 | 相同 |
| G / critic 学习率 | 1e-5 / 1e-5 | 相同 |
| G : critic 更新次数 | 1 : 5 | 相同 |
| GAN G / D 权重 | 0.03 / 0.03 | 相同 |
| R1 / R2 | 0 / 0 | 相同 |
| 判别器 | layers [21,29]，2 registers，FFN 2048 | 相同 |
| training seeds | 0、1 | 0、1，分别配对 |

不再使用之前的“全部 21 梯度帧、EMA 关闭、固定判别器 seed 12345”方案。四组共享同一份 raw generator 初始化；同一 seed 内的 frame/chunk 判别器初始 tensor hash 必须相同，跨 seed 按 main 的模型初始化规则变化。梯度窗口随 block 大小的变化是原实现的联动，必须披露；本实验不单独隔离该联动机制。

## 评测和预算

用户已确认使用 EMA。主评测用 `main_ffe`：保持固定 block 大小，**第一个 block 四步，其余 block 一步**。因此 framewise 的第一帧四步，chunkwise 的前三个 latent frames 所在首块四步。这与 main 的 `scripts/infer.sh` 一致。

review 分支的 `ffe` 是先联合生成 4 个 latent frames 再逐帧生成，并额外覆盖 attention 参数，不能代替 main。新增 `main_ffe` 路径保留 main 的 `model_kwargs` 默认值；旧 `ffe` 路径继续用于已有历史实验。

正式 VBench 仅评测第 200 步，采用固定 Qwen rewrites、944 prompts × 5 samples，同一份噪声 manifest、原始 VBench scoring prompts 和全部 16 维。四组共 18,880 条视频。

**用户最新要求：frame/chunk 各两个 seed 全部固定训练 200 iterations，到第 200 步结束。** 不再按时间预算 T 延长，也不运行 750 步。每组对应 40 次 generator 更新和 200 次 critic 更新。保存第 **0、50、100、150、200** 步，但仅第 200 步 EMA 进入 VBench；0/50/100/150 不做全量或子集 VBench，不生成初始化质量参考。中间 checkpoint 用于恢复，训练日志保留。

训练耗时、初始化、checkpoint I/O 和其他开销分别记录，报告每组实际训练成本。相同 iterations 不等于相同 wall-clock compute，本轮不能声称完成了严格的等时长训练对照。终点汇总输出第 200 步 EMA 的两 seed 结果，无需重复评测；本轮不提供 VBench 学习曲线。总计仍为 4 个 A + 2 个 CD = **6 个正式 run**。

## 实现与复现边界

- ICLR 入口复用现有 One-Forcing 损失与优化步骤，关闭 review 的可选手动 backward、optimizer CPU offload 和额外判别器优化器。保留完整状态保存、计时和非有限值检测。
- main 的配置 seed=0 会触发随机取 seed；本轮显式固定 seeds 0/1（rank 加偏移），保证配对可审计。这是新训练重复，不是复播历史随机流。
- sampler 按 main 的默认 seed 0；main 的 `cycle()` 不调用 `set_epoch`，本轮也重复相同排列。可恢复 loader 使用独立 RNG，保存数据位置，不声称与旧 DataLoader 的全局随机数消耗逐位一致。
- 保留 review 中 EMA 导出修复：转换为可加载的完整参数名，不将 FSDP flat/shard 参数名当完整权重。ICLR 同时保存 raw、EMA 全权重及每 rank 的 EMA/optimizer/RNG/data 状态；使用 checkpoint schema 3，拒绝旧 schema 1/2 无缝恢复。
- 第 200 步训练超参数和评测权重选择与 main 对齐，但原主表完整 rewrite 文件/随机流未归档。固定仓库 Qwen assets 的新结果不能预先保证数值恰好为 83.76；初始化和训练数据来源仍需训练机核对。
- CPU 检查覆盖配置、真实训练循环的 toy EMA/恢复及推理 block 调度；Wan 多卡数值行为、内存和视频动态画面仍需训练机验证。

执行命令见[运行手册](../experiments/iclr/README.md)，交接入口见[训练机任务](ICLR_TRAINING_MACHINE_TASK.md)。
