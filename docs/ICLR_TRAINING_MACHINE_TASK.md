# 训练机 Codex 执行任务：ICLR 2027 补充实验

请接手并执行本文件中的实验。这是执行任务，不要只返回计划。先检查环境和资产，再完成 GPU 验证、正式训练及评测。某项缺少资产时，继续推进其他独立工作。

**最新范围：实验 A 跑 4 个正式训练 run，依次为 `frame_roff_s0`、`chunk_roff_s0`、`frame_roff_s1`、`chunk_roff_s1`，R1/R2 全部关闭，四组都固定 200 iterations 后结束，保存 0/50/100/150/200 步，不按时间预算延长。用户已新增 chunk seed 1，用于两组 seed 的 frame/chunk 配对比较。旧 3-run / 12-run 配置已不适用，必须在新目录重新生成；不增加正则开启或 seed 2 的训练。**

**CD 权重已确认找不到，实验 B 必须重训 original/rectified，各 300 updates、training seed 0，共 2 个 run。A+B 合计 6 个正式训练 run；CD 的 all1/all4 是 4 个评测条件，不是 4 次训练。用户已指定 AR diffusion 初始化，并确认“严格轨迹对照 + 历史诊断”，详见 [CD 一步实验设计](ICLR_CD_ONE_STEP_DESIGN.md)。**

**CD 核查更新：论文 Table 1 对应的历史脚本、配置和汇总结果已经找到，历史记录还包含同一对 step-300 checkpoint 的一步评测。详见[CD 来源与重建说明](ICLR_CD_PROVENANCE.md)。旧配置 seed=2026，当前 ICLR 封装默认 seed=0，必须将后者标为新受控重跑，不能称为原设置的精确复现。旧成品权重丢失不代表实验流程丢失；共享初始化和轨迹数据可以重新准备。**

**最新对齐要求：实验 A 的 framewise 使用标准 main 的训练设置；chunkwise 仅把每 block 帧数从 1 改为 3，其余模型设置不变。用户确认主表使用 EMA。以 [main 对齐记录](ICLR_MAIN_ALIGNMENT.md) 为准，旧的“21 梯度帧、EMA 关闭、raw/all1 主评测”方案已取消，不能复用其训练产物。**

**三位审稿人合并范围：nPEg 的 A/B/C 为最高优先级；其后追加 PkyQ/o6Jx 的主表协议核对与原始 prompts 评测。正式训练仍为 6 个 run；A 的两个 frame seeds 同时提供标准配置第 200 步 EMA 的重复性证据。完整覆盖关系见[三位审稿人合并任务](ICLR_REVIEWER_COVERAGE.md)。**

## 仓库与已有进度

- 仓库：<https://github.com/Aurora-edu/One-Forcing>
- 工作分支：`ICLR`，使用包含本文件的最新版本。
- 实验代码已验证基线：`f4bf6ddb101584ee40e2437829403a4a79312b2d`。这是基线记录，不要求退回该提交而丢失后续文档。
- 原五-run 方案通过本机全套 100 项 CPU 测试；六-run 修订的验证结果见[本机验证记录](ICLR_LOCAL_VALIDATION.md)。检查覆盖配置、CD 默认重训预算、launcher dry-run、配对汇总、图表导出及 original/Qwen 配对评测入口。
- 真实 Wan GPU 训练、FSDP 恢复、视频生成和视觉检查尚未完成，必须在本训练机验证。

先阅读适用的 `AGENTS.md`、[实验运行手册](../experiments/iclr/README.md)、[三位审稿人合并任务](ICLR_REVIEWER_COVERAGE.md)和[本机验证记录](ICLR_LOCAL_VALIDATION.md)。本文件说明要完成什么；运行手册提供命令、参数及实现限制。本轮优先级以这两份 ICLR 任务为准，不沿用旧 `experiments/rebuttal` 文档中的其它执行顺序或排除 CD 的旧计划。

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

识别 8 张 GPU 的型号、显存、占用、CPU 内存和磁盘空间。实验 A 共 4 runs，每组第 0、50、100、150、200 步共 5 个保存点，包含 raw、EMA 和恢复状态；按实测大小估算存储，另计 smoke、CD、轨迹和视频。使用独立训练环境和独立 VBench 环境，遵守 `requirements.txt` / `requirements-vbench.txt`，记录软件版本、GPU 信息和实际 Git commit。所有实验组使用相同环境及同一组卡。

找到并核实以下资产：

- Wan2.1-T2V-1.3B 基础模型、T5、tokenizer 和 VAE。
- Wan2.1-T2V-14B teacher。
- 真实视频 latent LMDB。
- 有训练来源记录的 raw causal ODE 初始化。
- CD 使用新的 `experiments/iclr/configs/cd_one_step.yaml`，初始化及轨迹源均为指定的 Causal-Forcing `checkpoints/framewise/ar_diffusion.pt`，核对固定 SHA256；LMDB manifest 必须含源模型 hash。旧 CD 成品权重不再是前置条件。

所有 frame/chunk 对照必须使用同一份已确认的 raw 初始化。不能仅凭文件名认定身份，也不能默认将 `/data/fengjiaqi/model.pt` 当作 ODE 初始化。

检查仓库相对路径 `wan_models` 是否正确；不要依赖上一台机器失效的 `Causal_forcing` 软链接。执行 CPU 测试、资产 preflight 和各入口 dry-run。记录缺失资产，避免启动到一半才发现路径错误。

## 二、先完成真实 GPU smoke 与恢复验证

先验证两种粒度，正则均关闭，各至少 10 iterations：

- `frame_roff_s0`
- `chunk_roff_s0`

使用 `experiments.iclr.make_configs` 与 `experiments.iclr.launch`，按运行手册生成 smoke 配置，分别用 `--run` 指定上述两项，在独立 tmux 中顺序运行。smoke 不计入 4 个正式 frame/chunk run；正式 `frame_roff_s1` 和 `chunk_roff_s1` 也必须各自观察前 10 步。

核对 losses、gradient norms、GAN logits、G/critic 实际更新次数、21 latent frames 的实际梯度窗口、各 rank 状态和 checkpoint 完整性。记录训练耗时、显存、CPU 内存及 checkpoint 大小。

完成一次真实 FSDP 中途保存/恢复验证：

- 使用非最终的完整 checkpoint；必要时专门安排能产生中间 checkpoint 的 smoke。
- 保持相同配置、world size、模型和优化器设置。
- 恢复后真正继续执行更新，不能只加载最终 checkpoint 后立即退出。
- 核对 step、optimizer、EMA、RNG、数据位置和累计训练时间。10 步 smoke 尚未启用 EMA；至少一次恢复检查需覆盖第 50 步后的 EMA checkpoint，可在正式 run 上按计划执行。

导出固定 prompts 的 smoke 视频，实际检查动态画面。10 步模型质量差不等于实现错误，但需排除数值发散、解码错误和画面崩溃。

如果 OOM 或出现实现问题，先诊断并修复。保留 main 的计算设置；若必须调整 offload、精度或训练超参数，先说明与 main 的差异，再确定方案。不能只对某一组降低分辨率、缩短梯度窗口、换 teacher 或改变学习率来“跑通”对照。

## 三、正式实验 A：framewise/chunkwise 的受控稳定性

完成以下 4 个正式训练运行，按表中顺序执行：

| run | 粒度 | training seed | R1/R2 |
|---|---|---:|---|
| `frame_roff_s0` | framewise | 0 | 0 / 0 |
| `chunk_roff_s0` | chunkwise | 0 | 0 / 0 |
| `frame_roff_s1` | framewise | 1 | 0 / 0 |
| `chunk_roff_s1` | chunkwise | 1 | 0 / 0 |

使用 pinned main 配置：相同 raw 初始化、判别器结构、训练数据、batch、优化器、G:D 更新比例、精度、offload 和分片设置。统一 21 latent frames 和 score 帧、`gradient_num_blocks=17`、EMA=0.99（从第 50 步开始），R1/R2=0。按 main 的规则，实际梯度窗口为 frame 17 帧 / chunk 21 帧，不额外覆盖；报告这一联动。同一 seed 内审计判别器初始化一致，跨 seed 按 main 的模型初始化规则变化。`--run all` 只运行上述 4 组，CD 用单独入口。

**四个 run 都固定训练 200 iterations，随后立即结束。** 对应每组 40 次 generator 更新、200 次 critic 更新。保存 0、50、100、150、200 步，不再设置时间预算 T，也不延长到 750 步。smoke/pilot 只估算 ETA 和资源需求，耗时估算包含完整 G/D 更新周期。

质量曲线按相同步数比较；初始化、checkpoint I/O、训练与其他开销分别记录，用实际时间报告成本。相同 iterations 不等于相同 wall-clock compute，不能将本实验标成严格等时长对照。

每个 seed 都保留，包括失败或发散的运行。不能更换不利 seed、挑最佳 checkpoint 或只报告成功组。执行初始化审计，确认四组 generator 一致、同 seed 的 frame/chunk critic 一致。

评测统一使用：

- 同一套 944 个 VBench prompts 的 pinned Qwen rewrites，评分仍使用 original prompts。
- 同一份生成 seed manifest，每 prompt 5 个样本。
- EMA generator；第 0 步无 EMA，单独标为 raw 初始化参考。
- main 的 `main_ffe`：固定 block，第一个 block 四步，后续一步。不要用 review 的 `ffe`。
- 全部 16 个 VBench 维度。

使用 `evaluate curve` 和 `summarize`，输出训练步数轴上的 4 条 run 曲线、G/critic updates、200 步终点表和实际训练耗时，以及 dynamic degree、imaging quality、subject consistency 等分项。`same_seed_comparison.csv` 分别报告 seeds 0、1 的 frame−chunk 差值；`paired_step_summary.csv` 报告这两个配对差值的均值和样本 SD。frame 和 chunk 均报告 n=2 的得分均值、样本 SD 和每次运行结果。保留固定 prompts 的视频对照和视觉检查记录。

报告中明确：该方案检验了关闭正则、相同 200 iterations 下的粒度差异，并对两种粒度各做两个 seeds 的重复性检查。n=2 仍是小样本，不据此声称统计显著或普遍稳定；本轮未做正则开/关消融。

完整曲线汇总自动输出 `step200_seed_summary.csv`，含两个 frame seeds 的 EMA 结果，无需重复评测。只需终点时可选择 `evaluate curve --reference-step 200` 和 `summarize --reference-step 200`。历史 rewrite/随机流未归档，因此不以“必须恰好达到 83.76”作为验收标准。未完成或发散的 seed 必须如实报告。

## 四、正式实验 B：端点监督 CD 与严格配对诊断

执行 [CD 一步实验详细设计](ICLR_CD_ONE_STEP_DESIGN.md) 和[运行手册第 5 节](../experiments/iclr/README.md#5-重训-originalrectified-cd再评测一步与四步)，其中给出数据准备、smoke、训练、评测和诊断命令。

| 组别 | intervention | training seed | updates | latent frames |
|---|---|---:|---:|---:|
| Original | `curved` | 0 | 300 | 21 |
| Rectified | `rectified` | 0 | 300 | 21 |

用户已指定从 Causal-Forcing 的 `checkpoints/framewise/ar_diffusion.pt` raw 初始化，并用同一权重重新生成配对轨迹。固定源模型 SHA256，不把 A 的 `causal_ode.pt` 或丢失的 CD 成品权重混用。旧数据缺少源 hash 时重新采集，不猜测或补写来源。

两个 arm 每次更新都用 `(L_adjacent_CD + L_high_noise_endpoint)/2`。仅三个中间轨迹状态不同；噪声、teacher 端点、clean conditioning、prompts、初始化、holdout、训练配置和数据顺序一致。保持 teacher forcing，以保留跨组固定端点；不能把本次称为完全消除了训推历史差异。21 帧、端点监督和留出集是新设置，不能把它包装为旧 9 帧纯 CD 的精确复现。

先各跑 10 步 smoke 并检查视频，再各跑 300 步正式训练，每次均在独立 tmux 观察到至少第 10 步。逐步保存 loss、梯度和耗时；检查点为 0、50、100、150、200、250、300，主结果固定 step-300 raw generator。训练 target EMA 不用于评测。保存完整配置、数据 manifest、源 hash、sample-order digest 和配对审计文件；checkpoint 不含 optimizer/target EMA 恢复状态，中断需从共同初始化重跑。

优先评测 original/rectified 的 all1，再评 all4；统一原始 944 prompts × 1、生成 seeds、raw 权重、全 16 维、无 FFE。输出四条件表和配对增益，实际播放视频检查运动冻结、结构和画面崩溃。若一步仍负，照实报告，不能以四步改善推定一步必须改善。

固定第 0–63 个轨迹行为留出诊断，两组训练均排除。用 `cd_diagnostics` 在七个保存点比较相同 KV sampler 下的真实历史与自由生成历史；输出逐帧 MSE、history gap、预测变化、训练与诊断 learning curves 及固定样本配对 MP4。它只诊断固定 teacher-forced target 下的历史敏感性，不替代 VBench，不是严格因果分解，不用于挑超参数或最佳 checkpoint。

本轮不增加 600/900 步或其他 CD seeds。共享数据缺失时按指定 AR 初始化重建，同时继续 A/C；旧成品权重丢失本身不阻塞 B。

## 五、正式实验 C：时间参数化与路径几何

使用 `experiments.iclr.collect_teacher`：

- Bidirectional Wan2.1-T2V-1.3B，CFG=6，shift=8。
- 预先固定 100 个含多种运动与场景的 prompts 及其顺序。
- 保存完整 50-step 轨迹与真实 timesteps。
- 同一 prompt 文件的前 20 个，再采集 200-step 轨迹做分辨率复核。

不能将旧 causal 脚本保存的几个稀疏状态冒充完整 bidirectional teacher 轨迹。记录权重 hash、solver、negative prompt、CFG、shift 和 noise seeds。当前 collector 使用 Euler；原论文 solver 未核实前，不声称精确复现原轨迹。

用 `experiments.iclr.trajectory` 分析 flow time、非线性时间参数化和弧长坐标，固定物理高噪声区 t≥0.9，采用非均匀区间加权及跨边界裁切。同时分析 latent 与 temporal-difference latent，报告路径/弦长、正交偏离和转角等指标。

50/200-step 对比只使用共同的 20 个 prompts。区分速度偏差、路径几何和因果解释，不预设结果一定支持论文的原强结论。高噪声局部干预属于后续可选实验，不以它阻塞本次三个必做实验。

## 六、较低优先级 D/E：主表协议与 prompts 补充评测

完成可执行的 A/B/C 后追加评测；CPU 协议核对可穿插进行，不抢占 P0 的训练或评测 GPU。执行[合并任务](ICLR_REVIEWER_COVERAGE.md)中的 P1-D/P1-E：

1. 填写 [Table 2 协议核对表](ICLR_BASELINE_PROTOCOL_AUDIT.csv)，区分统一重评、作者报告和未知来源，逐行说明 prompt rewriting、raw/EMA、FFE/NFE、样本数和评测版本。
2. 找到来源已核实的主模型 checkpoint，使用 `experiments.iclr.prompt_ablation` 比较 original 与 pinned-Qwen，保持 checkpoint、权重来源、采样方案及噪声 seeds 不变。默认 944×1，标为补充评测，不能与历史 944×5 混报。
3. 对有真实权重和可验证推理路径的一步基线补 original prompts，形成可用模型的共同协议表；四步 Self Forcing 仅为单独的四步参考。缺失基线如实标注，不用历史分数或非原生 checkpoint 填表。
4. 复用已有 GAN 消融和本轮 CD 结果回应相似问题，不新增重复训练、正则开启组或第三个 training seed。

P0-A 现已恢复 main 的训练设置；两个 frame seeds 的第 200 步 EMA 表可回答标准配置的初步训练重复性。明确 n=2、固定的新 Qwen 评测资产和历史随机流限制；step-400 或多个生成 seeds 不能代替这张第 200 步表，0.30 分优势仍不应声称统计显著。

## 七、持续沟通和最终交付

定期报告当前运行、step、耗时、ETA、已完成事项、缺失资产和失败原因。任何设置调整都说明是否影响受控比较。

最终交付：

1. 运行目录、tmux/log/checkpoint 路径、配置、环境与初始化来源记录。
2. GPU smoke、FSDP 恢复、评测和视频检查的真实完成状态。
3. 四个 run 在 0/50/100/150/200 步的质量曲线、各自实际耗时、seeds 0/1 的 frame/chunk 配对差值及其均值/样本 SD，以及第 200 步 EMA 表。
4. 两个 CD step-300 raw checkpoints、全部七个保存点、配对训练审计、all1/all4 结果表、历史诊断 CSV/learning curves 与实际视频检查记录。
5. 时间参数化与几何复核结果。
6. 固定 prompt 视频对照及视觉检查记录。
7. 可用于 rebuttal 的中文结果说明，包含负结果、局限及应收敛的论文表述。
8. 若修改代码，给出 Git commit 和复现命令。
9. Table 2 的逐行协议核对表，以及可用主模型/基线的 original prompts 补充表和主模型 original/Qwen 配对结果。
10. 分别给 nPEg、PkyQ、o6Jx 的回应要点，明确共享证据、n=2 方差的局限与历史协议缺项；所有评测与训练条件分清。

**不要将 dry-run、CPU toy 测试或成功加载 checkpoint 当成真实 GPU 实验完成。**
