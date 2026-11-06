# 训练机 Codex 执行任务：ICLR 2027 补充实验

请接手并执行本文件中的实验。这是执行任务，不要只返回计划。先检查环境和资产，再完成 GPU 验证、正式训练及评测。某项缺少资产时，继续推进其他独立工作。

## 仓库与已有进度

- 仓库：<https://github.com/Aurora-edu/One-Forcing>
- 工作分支：`ICLR`，使用包含本文件的最新版本。
- 实验代码已验证基线：`f4bf6ddb101584ee40e2437829403a4a79312b2d`。这是基线记录，不要求退回该提交而丢失后续文档。
- 上一台机器完成了全套 90 项 CPU 测试，以及 11 项配置、dry-run 和合成分析检查。
- 真实 Wan GPU 训练、FSDP 恢复、视频生成和视觉检查尚未完成，必须在本训练机验证。

先阅读适用的 `AGENTS.md`、[实验运行手册](../experiments/iclr/README.md)和[本机验证记录](ICLR_LOCAL_VALIDATION.md)。本文件说明要完成什么；运行手册提供命令、参数及实现限制。

## 执行约束

1. 不要停止、重启或干扰其他 session 的进程；GPU 被占用时不能直接抢占。
2. 数据优先在 `/data/fengjiaqi/` 查找；训练机路径不同时，记录真实路径映射。
3. 长时间训练使用独立 tmux，保存完整日志。每个新训练运行至少观察到第 10 步，确认正常后才能转入后台监控。
4. 必须检查生成视频的实际动态画面，不能只看指标或静态拼图。固定 prompts，对各组、各训练 seed 和关键 checkpoint 抽样检查；疑似崩溃视频逐一检查。
5. 不覆盖已有实验输出，不删除其他任务的数据，不把 synthetic fixtures 当成论文结果。
6. 已授权完成以下工作，无须对每个正常步骤重复请求确认。改变科学对照、无法确认 checkpoint 来源或缺少关键资产时，提出具体问题；同时继续独立工作。
7. 发现代码问题可以修复并补必要回归测试，记录修改和 Git commit。不要用悄悄改变实验设置来掩盖实现问题。

## 一、检查环境、GPU 与资产

检查工作区后获取 `ICLR` 分支，保留现有未提交修改，不使用破坏性 reset。确认包含上述代码基线和本文件。

识别 8 张 GPU 的型号、显存、占用、CPU 内存和磁盘空间。完整 checkpoint 预计需要约 2 TB 量级空间，视频另计。使用独立训练环境和独立 VBench 环境，遵守 `requirements.txt` / `requirements-vbench.txt`，记录软件版本、GPU 信息和实际 Git commit。所有实验组使用相同环境及同一组卡。

找到并核实以下资产：

- Wan2.1-T2V-1.3B 基础模型、T5、tokenizer 和 VAE。
- Wan2.1-T2V-14B teacher。
- 真实视频 latent LMDB。
- 有训练来源记录的 raw causal ODE 初始化。
- 论文 original/rectified CD checkpoints、原始配置和轨迹数据。

所有 frame/chunk 对照必须使用同一份已确认的 raw 初始化。不能仅凭文件名认定身份，也不能默认将 `/data/fengjiaqi/model.pt` 当作 ODE 初始化。

检查仓库相对路径 `wan_models` 是否正确；不要依赖上一台机器失效的 `Causal_forcing` 软链接。执行 CPU 测试、资产 preflight 和各入口 dry-run。记录缺失资产，避免启动到一半才发现路径错误。

## 二、先完成真实 GPU smoke 与恢复验证

分别执行四种设置，各至少 10 iterations：

- `frame_roff_s0`
- `frame_ron_s0`
- `chunk_roff_s0`
- `chunk_ron_s0`

使用 `experiments.iclr.make_configs` 与 `experiments.iclr.launch`，按运行手册生成 smoke 配置，在独立 tmux 中顺序运行。

核对 losses、gradient norms、GAN logits、G/critic 实际更新次数、21 latent frames 的实际梯度窗口、各 rank 状态和 checkpoint 完整性。记录训练耗时、显存、CPU 内存及 checkpoint 大小。

完成一次真实 FSDP 中途保存/恢复验证：

- 使用非最终的完整 checkpoint；必要时专门安排能产生中间 checkpoint 的 smoke。
- 保持相同配置、world size、模型和优化器设置。
- 恢复后真正继续执行更新，不能只加载最终 checkpoint 后立即退出。
- 核对 step、optimizer、RNG、数据位置和累计训练时间。

导出固定 prompts 的 smoke 视频，实际检查动态画面。10 步模型质量差不等于实现错误，但需排除数值发散、解码错误和画面崩溃。

如果 OOM 或出现实现问题，先诊断并修复。允许统一调整所有组的系统设置并记录；不允许只对某一组降低分辨率、缩短梯度窗口、换 teacher 或改变学习率来“跑通”对照。

## 三、正式实验 A：framewise/chunkwise 的受控稳定性

完成 `2 × 2 × 3 = 12` 个训练运行：framewise/chunkwise，R1/R2 off/on，training seeds 0、1、2。

使用 ICLR 统一配置：相同 raw 初始化、判别器结构、训练数据、batch、优化器、G:D 更新比例、正则定义、精度、offload 和分片设置。统一 21 latent frames、21 梯度帧和 score 帧。开启正则的定义保留代码中与旧 chunkwise 对齐的设置，不能只复制权重而改变归一化方式。

根据 smoke/pilot 测得的完整 G/D 更新周期耗时，预先确定统一训练预算 T，使最慢组能覆盖约 750 iterations 的量级。运行手册中的 7200 秒是示例，不能未经测量就认定为合适预算。

正式实验按相同累计训练时间停止，不再把 200 与 750 步直接作为效率比较。保留 0、0.25T、0.5T、0.75T、T 的 checkpoint，报告实际保存时间及 overshoot。初始化、checkpoint I/O、训练与其他开销分别记录。

每个 seed 都保留，包括失败或发散的运行。不能更换不利 seed、挑最佳 checkpoint 或只报告成功组。执行初始化审计，确认实际加载的 generator/critic 权重一致。

评测统一使用：

- 同一套 944 个原始 VBench prompts。
- 同一份生成 seed manifest，每 prompt 1 个样本。
- Raw generator，不用 EMA。
- 所有组统一 all1，不开 FFE。
- 全部 16 个 VBench 维度。

使用 `evaluate curve` 和 `summarize`，输出真实训练时间轴上的各 seed 曲线、G/critic updates、三训练 seeds 的均值和样本 SD、固定预算终点表，以及 dynamic degree、imaging quality、subject consistency 等分项。保留固定 prompts 的视频对照和视觉检查记录。

这一协议与论文 83.76 的设置不同，不以“必须达到 83.76”作为验收标准。未完成或发散的 seed 必须如实报告，不能为凑齐表格删除失败结果。

## 四、正式实验 B：original/rectified CD 的一步与四步

优先找到论文原始两个 300-update checkpoint，核对来源和配置，评测四个条件：Original × all1、Original × all4、Rectified × all1、Rectified × all4。

统一原始 prompts、生成 seeds、raw 权重和全 16 维评测。all1 必须所有 block 都一步，禁止开启 FFE。

使用 `experiments.iclr.evaluate cd` 和 `summarize_cd` 生成四条件表及配对增益，检查对应视频的运动冻结、结构崩溃和视觉质量。单一训练 seed 的配对聚合增益不能解释为训练 seed 的置信区间。

需要训练预算曲线时，使用同一配置从头训练到 900，保存 300/600/900。旧 checkpoint 缺少 optimizer/target 状态时，不得声称无缝续训。

特别注意：CD 工具默认继承 review 分支的 **9 latent frames 小实验配置**。先核对论文原始设置，不能把默认 9 帧实验直接称为论文复现。两个 arm 必须使用同一份已确认配置。训练中的 EMA target 与评测使用的 raw generator 要区分。

原论文 checkpoints 或轨迹缺失时，列出缺失内容，继续 A/C，不用未知 checkpoint 替代。若有合法且已核实的资产可以构建新受控实验，明确标为新实验，保留来源和配置。

## 五、正式实验 C：时间参数化与路径几何

使用 `experiments.iclr.collect_teacher`：

- Bidirectional Wan2.1-T2V-1.3B，CFG=6，shift=8。
- 预先固定 100 个含多种运动与场景的 prompts 及其顺序。
- 保存完整 50-step 轨迹与真实 timesteps。
- 同一 prompt 文件的前 20 个，再采集 200-step 轨迹做分辨率复核。

不能将旧 causal 脚本保存的几个稀疏状态冒充完整 bidirectional teacher 轨迹。记录权重 hash、solver、negative prompt、CFG、shift 和 noise seeds。当前 collector 使用 Euler；原论文 solver 未核实前，不声称精确复现原轨迹。

用 `experiments.iclr.trajectory` 分析 flow time、非线性时间参数化和弧长坐标，固定物理高噪声区 t≥0.9，采用非均匀区间加权及跨边界裁切。同时分析 latent 与 temporal-difference latent，报告路径/弦长、正交偏离和转角等指标。

50/200-step 对比只使用共同的 20 个 prompts。区分速度偏差、路径几何和因果解释，不预设结果一定支持论文的原强结论。高噪声局部干预属于后续可选实验，不以它阻塞本次三个必做实验。

## 六、持续沟通和最终交付

定期报告当前运行、step、耗时、ETA、已完成事项、缺失资产和失败原因。任何设置调整都说明是否影响受控比较。

最终交付：

1. 运行目录、tmux/log/checkpoint 路径、配置、环境与初始化来源记录。
2. GPU smoke、FSDP 恢复、评测和视频检查的真实完成状态。
3. 四组 matched-time 曲线和三 seed 终点表。
4. CD all1/all4 结果表。
5. 时间参数化与几何复核结果。
6. 固定 prompt 视频对照及视觉检查记录。
7. 可用于 rebuttal 的中文结果说明，包含负结果、局限及应收敛的论文表述。
8. 若修改代码，给出 Git commit 和复现命令。

**不要将 dry-run、CPU toy 测试或成功加载 checkpoint 当成真实 GPU 实验完成。**
