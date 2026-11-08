# ICLR 本机验证记录

最近验证日期：2026-11-08（较早检查日期见各节）。工作目录：`/home/jiaqi/One-Forcing`。

**当前追加范围：** [18 帧 frame/chunk、一步 CD、旧 seed=0 original 评测](ICLR_CURRENT_EXPERIMENTS_TASK.md)，仍为 4 个正式 run、6 个新增终点评测条件。frame/chunk 为 seed=42 各 200 步；CD 为 seed=2026 的 Original/Rectified 各 300 步、只评 all1；另复用旧 seed=0 frame/chunk 权重补 original/EMA/main_ffe/944×5 评测，不重新训练。取消的公开配方独立 chunk 第三组继续取消。CPU 检查见末节；下文旧四组/旧 CD 范围均是此前任务记录。

最新范围：实验 A 为 `frame_roff_s0`、`chunk_roff_s0`、`frame_roff_s1`、`chunk_roff_s1`，R1/R2 全关；加上 CD 两组，共 6 个正式训练 run。A 以 `main@c9a2350` 为标准，恢复梯度 17-block 策略、EMA 和 main 首块增强；四组均固定 200 iterations，在 0/50/100/150/200 步保存。下文旧 12-run、三-run、五-run 以及 EMA-off 六-run 检查均为历史记录。最新决定只在第 200 步做 VBench，中间 checkpoint 继续保存但不评测；下文完整曲线检查属于历史记录。当前设置见 [main 对齐记录](ICLR_MAIN_ALIGNMENT.md)及本文件末尾。

## 分支与模型

`ICLR` 从 `origin/review` 的 `bfd5e2d` 创建。用户提供的论文 PDF 保留在工作区，不提交。
实验主体代码由 `gpt-6.1-sol`、`xhigh` 编写。该模型因额度限制中断后，用户要求继续，主会话接手补齐、集成和验证；未将后续工作声称为仍由该模型执行。

## 本机环境

使用现有环境，不修改其他会话的依赖：

```text
/home/jiaqi/miniconda3/envs/causal_forcing/bin/python
Python       3.10.20
torch        2.5.1+cu118
torchvision  0.20.1+cu118
numpy        1.26.4
transformers 4.51.3
diffusers    0.31.0
accelerate   1.13.0
flash-attn   2.7.3
```

本机 numpy / transformers 与仓库锁定版本（1.24.4 / 4.49.0）不同。
CPU 检查通过不能视为两个版本组合的 GPU 数值等价验证。8 卡正式训练应使用独立环境，遵守仓库依赖版本，所有实验使用完全相同的软件环境。

## 已核对的数据资产

| 资产 | 路径与检查结果 |
|---|---|
| 真实视频 latent LMDB | `/data/fengjiaqi/One-Forcing/mixkit_latents_lmdb`；shape=`[5763,21,16,60,104]`；第一个 latent 全部有限，prompt 存在 |
| Wan 1.3B 基础模型 | `/home/jiaqi/causal_forcing/wan_models/Wan2.1-T2V-1.3B`；模型、VAE、T5 和 tokenizer 文件存在 |
| Wan 14B teacher 目录 | `/data/fengjiaqi/wan_models/Wan2.1-T2V-14B` |
| 未确认来源的 checkpoint | `/data/fengjiaqi/model.pt`，含 `generator/critic/generator_ema`，没有 step；不能据此认定为 ODE 初始化或论文 CD checkpoint |
| 失效路径 | `/data/fengjiaqi/Causal_forcing` 指向不存在的 `/data/fengjiaqi/causal_forcing`，新实验不要依赖它 |

原始 ODE 初始化、original/rectified CD checkpoint 以及用于轨迹分析的真实 teacher 轨迹，尚未在上述目录核实。迁移时必须显式提供有来源记录的文件，不能用未确认身份的 checkpoint 替代。

## 基线检查

以下是在新增代码前对继承的 `review` 分支进行的检查：

```bash
python -m pytest -q
# 53 passed
python train.py --help
python scripts/export_videos.py --help
python experiments/rebuttal/train_curvature_cd.py --help
python experiments/rebuttal/validate_configs.py
# Rebuttal configs are paired and valid.
```

## GPU 验证状态

本机有 8 张 RTX A6000（48 GiB）。检查时全部被其他会话的 vLLM worker 使用，每卡约占用 39–47 GiB，利用率约 100%。空闲检查正确拒绝启动训练；没有停止、重启或修改这些进程。

用户在本次会话中明确选择“先完成 CPU 验证，GPU 验证留到训练机”。因此本次交付范围为代码、CPU 验证和迁移命令。真实模型训练至少 10 步、分布式 FSDP 恢复、视频生成和实际视频检查留待训练机执行。CPU 测试、配置生成、解析器检查与 dry-run 都不等于完成这些 GPU 检查。

可用 GPU 到位后，按新实验运行手册在独立 tmux 中执行 smoke，至少观察到第 10 步，再导出视频并实际检查动态画面，最后启动完整多种子实验。主实验结果不能使用 smoke 的缩小规模设置。

机器可读环境记录：`/data/fengjiaqi/One-Forcing/iclr_local_validation/assets.json` 与 `gpu_inventory.json`。这些是本机诊断产物，不是论文实验结果，也不随 Git 提交。

## 原方案的 CPU 验证（历史记录）

- 全套测试：`python -m pytest -q`，90 passed（包含原有 53 项）。
- Python 编译和 `git diff --check` 通过。
- 11 个配置/命令/分析检查通过：生成 12 个配置、8 卡 launcher dry-run、chunkwise 入口、CD 900-step 命令、四条件 CD 评测命令、50/200-step teacher 采集计划、合成轨迹分析、两种结果汇总计划和初始化审计计划。
- 使用真实 ICLR 训练循环、tiny CPU 模型运行 10 步，并验证 step-5 保存后恢复到 step-10 与连续运行的模型、AdamW 状态、RNG、数据流位置完全一致。CUDA/FSDP collectives 在该测试中替换为 CPU 等价物，不是 8 卡验证。
- 验证 NaN 失败、不同 topology/config/缺失 rank 状态拒绝恢复、恢复后不被初始化权重覆盖。
- 几何分析使用显式标记为 `synthetic_test` 的匀速直线、变速直线、弯曲路径；CLI 必须加 `--synthetic-check`，输出标记为 `synthetic_validation`。默认 teacher 分析拒绝这些 fixtures。它们不是实际 teacher 轨迹。
- 用 synthetic VBench fixtures 验证 12 组、多时间点的 CSV/PDF/PNG 导出，并查看 PNG 排版。没有将 synthetic 指标用作论文结果。

最终机器可读命令检查记录：`/data/fengjiaqi/One-Forcing/iclr_local_validation/final_cpu_v2/report.json`。前面的临时目录是开发过程中间产物，不作为最终验收记录。

## 三-run 方案修订后的 CPU 验证（历史记录）

2026-11-06，在相同环境执行：

```bash
/home/jiaqi/miniconda3/envs/causal_forcing/bin/python -m pytest -q
# 95 passed in 23.91s
git diff --check
```

- 配置和 launcher dry-run 仅生成并依次列出 `frame_roff_s0`、`chunk_roff_s0`、`frame_roff_s1`；全部关闭 R1/R2。
- 拒绝旧 12-run 矩阵、额外 seed、正则开启、重复/错序 run、名称与配置不一致及其他配置混杂项。
- Synthetic VBench fixtures 验证三条曲线与 CSV/PDF/PNG 导出；frame n=2 使用样本 SD，chunk n=1 的 SD 为未定义值，CSV 留空。
- `same_seed_comparison.csv` 使用 frame seed 0 减 chunk seed 0；测试明确区分该差值与 frame 两 seed 均值减 chunk 单次结果。
- 初始化审计默认检查 3 个 run。恢复、CD 与轨迹等既有 CPU 回归测试继续通过。

本次修订未启动 GPU 训练或生成真实视频；GPU 验证与动态视频检查仍须在训练机完成。

## CD 权重丢失后的重训方案（五-run 阶段历史记录）

用户确认旧 CD 权重找不到，实验 B 改为必须从共享 raw 初始化重训 `curved` / `rectified`，各 300 updates、seed 0，再评 all1/all4。A+B 共 5 个正式训练 run，smoke 和评测条件单独计数。旧 CD 权重不再是前置资产；配对轨迹 LMDB 及其来源初始化仍需在训练机核实。

`experiments.iclr.cd` 默认预算由 900 改为 300。执行 `python -m pytest -q tests/test_iclr_protocol.py`，25 passed（7.62s），包括默认 CD dry-run 恰好输出两组、相同 seed、300 updates 和不支持续训的检查；`git diff --check` 通过。本次仅修改入口默认值及执行文档，未启动真实 CD 训练。

## 三位审稿人合并后的 CPU 验证（五-run 阶段历史记录）

用户选择 nPEg 优先、正式训练仍为 5 个 run；PkyQ/o6Jx 的新增工作限于评测与协议核对，主配置训练 seed 方差暂不完整补齐。新增任务说明为 `docs/ICLR_REVIEWER_COVERAGE.md`，主任务已链接该文件。

- 全套 `python -m pytest -q`：100 passed（25.46s）。`git diff --check` 通过。
- 新增 `experiments.iclr.prompt_ablation`，默认 dry-run；可在 CPU 准备 944×1 或 944×5 的原始/Qwen 配对 manifests，执行评测要求 tmux 和空闲 GPU。
- 5 项新增 CPU 测试验证 prompt/sample/noise seed 配对、配置与权重选择、单样本/五样本混用拒绝、不同 checkpoint 拒绝，以及无权重/GPU 的 dry-run。
- Synthetic VBench fixtures 仅验证汇总逻辑，未用于填写任何真实结果。
- 已从本地 PDF Table 2 整理 13 行协议核对 CSV，验证字段数量和 CSV 格式。纸面分数标为历史引用，无法从当前证据确定的协议字段保留 `unknown`，不表示已重评。

尚未运行新增 GPU 评测或检查新增 MP4；没有新增主配置训练，也没有改变原五个训练 run 的安排。所有修改仍需随代码迁移到训练机后执行。

## 六-run 方案：新增 chunk seed 1 后的验证（历史记录）

最新用户决定增加 `chunk_roff_s1`。实验 A 变为 frame/chunk 各 seeds 0、1，共 4 个 run；实验 B 仍为两个 seed-0、300-update CD run，正式训练总计 6 个。nPEg 优先级不变，不新增主配置训练。

```bash
/home/jiaqi/miniconda3/envs/causal_forcing/bin/python -m pytest -q tests/test_iclr_protocol.py tests/test_iclr_reporting.py
# 37 passed in 16.25s
git diff --check
```

- 配置生成和 launcher dry-run 按顺序输出 frame s0、chunk s0、frame s1、chunk s1；拒绝旧三-run/十二-run 矩阵、重复和错序运行。
- 初始化审计默认检查 4 个 frame/chunk run；CD 两组仍由独立入口执行。
- Synthetic VBench fixtures 验证两种粒度各 n=2 的均值/样本 SD、两个 seed 的配对差值，以及 `paired_time_summary.csv` 中两组差值的均值/样本 SD；缺失 chunk seed 1 或配对 seed 会报错。
- 真实时间曲线、CSV/PDF/PNG 导出，以及 CD 默认 300 步与四条件评测的相关 CPU 回归检查通过。所有合成分数仅用于测试。

本次没有启动 GPU 训练或视频生成，也没有修改其他 session 的进程。真实六-run 实验及视频检查仍在训练机完成。

## 按 main 对齐后的六-run 验证（历史时间预算方案）

用户确认标准实现为 GitHub main，主表使用 EMA。核对提交 `c9a2350e8740531562011fc9618e1a928d911ae0`；保存的 `main_reference.yaml` 与该提交 `config.yaml` 的 Git blob 均为 `1caabde9e340552c4c9e9d39f8f5f90c03321b46`。

全套 CPU 回归：**109 passed in 30.25s**。本次显式隐藏 GPU；Wan 的 T5 模块在导入时会求值 `torch.cuda.current_device()`，因此只在测试进程临时替换该查询，不改生产代码或共享环境：

```bash
CUDA_VISIBLE_DEVICES='' /home/jiaqi/miniconda3/envs/causal_forcing/bin/python - <<'PY'
import torch
import pytest
torch.cuda.current_device = lambda: 0
raise SystemExit(pytest.main(['-q', '--tb=short', '--show-capture=no']))
PY
```

验证内容：

- 四个配置继承 main；梯度仍按 17 blocks 计算，frame/chunk 实际为 17/21 帧；EMA=0.99，从第 50 步开始。修改 EMA、梯度策略等参考超参数会被拒绝。
- 数据 sampler 保持 main 的 seed 0 和每轮重复排列，恢复后数据顺序一致；模型初始化按同 seed 配对审计。
- 真实 ICLR 循环在 tiny CPU 模型上验证 10 步 EMA/AdamW/RNG/数据恢复等价；该 toy 单独将 EMA 起点提前至 5，正式配置仍为 50。
- 额外用 CPU toy 验证预算在第 200 步之前、同时、之后达到的三种情况：第 200 步 checkpoint 和匹配时间终点分别正确保存，重合时不重复保存。
- 上述检查发现并修复 checkpoint index 中 clock 的完整 fractions 列表覆盖本次保存 fractions 的问题，防止所有 checkpoint 被错误标成所有时间点。
- 实际执行 `scripts/infer.sh` 的配置生成部分，替换 GPU exporter 为测试 stub；验证 frame/chunk 的固定 block 数、首块四步/后续一步及 main attention 默认值。没有生成真实视频。
- 合成分数仅验证 matched-time 配对均值/SD、step-200 独立表和旧 raw/all1 产物拒绝；没有作为论文结果。
- 配置生成、四条 8 卡 launcher 命令、train dry-run、固定 944×5 Qwen manifests、chunk 第 200 步 EMA/main_ffe 评测计划全部通过。CLI 检查在临时目录执行，没有加载模型或启动 GPU。
- `git diff --check` 和所改 shell 脚本语法检查通过。

本轮改动保留在本机 ICLR 工作区，未提交或推送。Wan 多卡训练、实际 FSDP EMA 导出/恢复和 MP4 动态画面检查仍由训练机完成；未干扰其他 session 的进程。


## 固定 frame/chunk 各 200 步后的 CPU 验证

用户最新要求：四个 frame/chunk run 都在第 200 步结束，不再由时间预算延长。正式保存点改为 0、50、100、150、200；smoke 仍为 10 步，保存 0、5、10。每个正式 run 为 40 次 generator 更新和 200 次 critic 更新。新 profile 为 `main_block_ablation_200_v2`，checkpoint schema 为 3；旧时间预算配置需重新生成，旧 schema 不支持本轮完整恢复。

```bash
CUDA_VISIBLE_DEVICES='' /home/jiaqi/miniconda3/envs/causal_forcing/bin/python - <<'PYTEST'
import torch
import pytest
torch.cuda.current_device = lambda: 0
raise SystemExit(pytest.main(['-q', '--tb=short', '--show-capture=no',
    'tests/test_iclr_protocol.py', 'tests/test_iclr_restart.py', 'tests/test_iclr_reporting.py']))
PYTEST
# 55 passed in 26.87s
```

- 真实 ICLR 循环的 CPU toy 验证 frame/chunk 在每步耗时极短或极长时，都只完成 200 步，保存五个指定 checkpoint，实际累计耗时单独记录。
- 完整恢复测试核对 optimizer、EMA、RNG、数据位置与不中断运行一致；这些测试替换了 CUDA/FSDP collectives，不代表真实多卡验证。
- 四配置生成、launcher/train dry-run、旧时间预算和非 200 步正式配置拒绝，以及已有推理调度检查通过。
- 合成 VBench 数据验证相同步数下的两 seed 配对汇总、各自实际耗时、CSV/PDF/PNG 导出；完整曲线同时导出 `step200_seed_summary.csv`，无需重复评测终点。合成数据不作为论文结果。
- 训练机任务、运行手册、main 对齐记录与审稿人覆盖说明已同步；固定步数结果不标为严格匹配 wall-clock compute。
- `git diff --check` 通过。

本次仅做本地代码、文档修改和 CPU 验证；未启动 GPU 训练、生成视频或修改 CD 设置，也未提交或推送 Git。

## 发布到 ICLR 分支前的完整验证

用户要求将当前实验更新到 GitHub。固定 200 步代码、main 对齐评测、original/Qwen 补充评测、三位审稿人任务和 CD 来源说明一并纳入本次更新，论文 PDF 不纳入提交。

- 使用上文 CPU bootstrap，去掉测试文件限制运行全套 pytest：**113 passed in 30.34s**。
- `git diff --check` 及 `bash -n scripts/infer.sh experiments/rebuttal/run_vbench_condition.sh` 通过。
- `main_reference.yaml` 与 `main@c9a2350` 的 `config.yaml` 逐字相同；新增说明的本地链接有效。
- CD 历史一步结果、数据重建流程和 seed 2026/当前默认 seed 0 的差异已写入 `ICLR_CD_PROVENANCE.md`，本次未改变 CD 默认训练设置。

这些是 CPU 和静态检查，GPU 训练、真实 FSDP 恢复及视频检查仍需训练机执行。

## 端点监督 CD 与严格历史诊断（当前设置）

用户指定 Causal-Forcing `checkpoints/framewise/ar_diffusion.pt`，并选择“严格配对轨迹 + 历史误差诊断”。当前 CD 配置为 `endpoint_anchored_cd_v1`：两组各 300 updates、seed=0、21 latent frames，每步等权 CD/最高噪声端点损失；保留 teacher forcing，固定前 64 个轨迹行留出。frame/chunk 四个 200 步 run 不变，共 6 个正式 run。

使用上文 CPU bootstrap，显式隐藏 GPU：**125 passed in 35.64s**。相关定向回归在新增评测/数据测试前为 60 passed in 24.30s；最终以全套 125 项为准。`git diff --check`、所改 Python 的语法解析、相关 shell 语法与新增文档本地链接检查通过。

- 检查四种 CD 相邻 pair 的分次 backward 与完整组合目标梯度一致，EMA target 不累积梯度；固定参数下 original/rectified 的端点项相同、CD 项和梯度保留轨迹差异；非有限损失会停止。
- 检查固定配置、源模型 hash、21 帧数据形状、训练/留出行隔离、两组样本顺序审计；合成源轨迹经真实 LMDB builder 后保留源 hash。
- 使用真实 `CausalInferencePipeline.inference` 控制流和 tiny CPU generator，验证首帧相同、真实历史仅在预测后缓存、当前/未来 clean latent 不泄漏，以及 forced/free 多次调用之间 cache 重置。
- 使用合成 checkpoints 检查正式评测前置审计：两组来源和配置一致、固定 raw step-300；错误步数或旧 9 帧元数据被拒绝。
- 检查诊断逐帧配对覆盖、负 history gap 的保留、CSV/PDF/PNG 导出及无权重/无 GPU 的 CLI dry-run。所有合成数据仅用于测试，不是实验结果。

新增 [CD 详细设计](ICLR_CD_ONE_STEP_DESIGN.md)，同步训练机任务、运行手册、审稿人覆盖和历史来源记录。历史 all1 −1.44 / all4 +9.95 保留，新设置不宣称保证正收益。没有启动 GPU 训练或生成真实视频；8 卡 FSDP、显存、各组前 10 步、真实视频目视检查及 VBench 仍须在训练机执行。没有修改其它 session 的进程。


## 仅评测第 200 步 VBench（2026-11-07）

用户明确选择 frame/chunk 只测第 200 步 VBench。四个 run 的训练预算和保存点不变；中间 0/50/100/150 不做全量或子集 VBench。主评测保留 EMA/main_ffe/pinned-Qwen/944×5，共 18,880 条视频。CD 最终第 300 步 all1/all4 评测和历史诊断保持原范围。

在相同 Python 环境、显式隐藏 GPU 后执行：

```bash
CUDA_VISIBLE_DEVICES='' /home/jiaqi/miniconda3/envs/causal_forcing/bin/python - <<'PYTEST'
import torch
import pytest
torch.cuda.current_device = lambda: 0
raise SystemExit(pytest.main(['-q', '--tb=short', '--show-capture=no',
    'tests/test_iclr_protocol.py', 'tests/test_iclr_reporting.py',
    'tests/test_iclr_cd_one_step.py', 'tests/test_iclr_prompt_ablation.py']))
PYTEST
# 63 passed in 25.05s
```

- 索引包含五个保存点时，默认 API 和 CLI 只选择 step 200、真实 block 粒度和 EMA；没有终点时失败，不回退到中间点。
- Synthetic fixtures 验证默认终点汇总、两 seed 单次值/均值/样本 SD、两组 frame−chunk 配对差值、实际训练耗时；拒绝混入中间点、遗漏 seed 或不一致的协议。汇总不再输出 VBench 曲线。
- CD 配对训练协议、all1/all4 表、历史诊断和 original/Qwen 配对评测的相关 CPU 检查通过。
- `git diff --check` 通过。以上均为 CPU 检查与合成数据，不是实际 VBench 分数。未启动 GPU 训练或生成真实视频，未干扰其他 session 的进程。


## 原配置 9 帧纯 CD 对照（2026-11-08）

用户要求恢复可追溯的原 CD 训练配方。本次新增独立 `experiments.iclr.cd_original` / `configs/cd_original.yaml`，固定 seed=2026、9 latent frames、300 updates、纯相邻状态 CD、holdout=0、原优化器及 target EMA=0.95；原参考 YAML 的内容 hash 被固定，未修改参考文件。21 帧端点监督变体保持独立历史产物。本轮训练机只新增两组原配方训练及最终 all1/all4 评测，详细交接见 `ICLR_CD_ORIGINAL_TASK.md`。

在相同本机环境、显式隐藏 GPU 后运行全套 CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' /home/jiaqi/miniconda3/envs/causal_forcing/bin/python - <<'PYTEST'
import torch
import pytest
torch.cuda.current_device = lambda: 0
raise SystemExit(pytest.main(['-q', '--tb=short', '--show-capture=no']))
PYTEST
# 136 passed in 36.74s
```

- 10 步真实 `backward_cd_objective` / AdamW CPU toy 与纯 CD 参考优化一致；target 无梯度、没有额外 endpoint forward，frame 9 之后注入 NaN 不进入训练窗口。
- 构造明确标为 synthetic 的 21 帧 LMDB，用真实数据读取器和 CLI 验证 hash 匹配时可复用、只截取 9 帧且不修改 manifest；源权重变化时拒绝复用。不运行轨迹生成。
- 验证两个独立 300-update 日志的 75/75/75/75 相邻对计数、全零端点更新、实际 sample-order digest、raw checkpoint 及 profile 隔离；原配置与 21 帧变体不能交叉用于评测。
- 四条件合成评测验证负增益保留、historical_comparison.csv 与历史原始汇总一致，并明确历史初始化匹配未确认；合成分数不用于论文。
- 默认与 smoke CLI dry-run 均在不存在的资产路径上通过，不查询 GPU、不启动训练；执行文档中的 bash 片段通过 `bash -n`，`git diff --check` 通过。
- 纳入训练机的同一处 B200 多 rank intent 写入竞态修复（255c65e 的 barrier），不更改训练损失或优化器。

本次未启动真实 Wan/GPU 训练或生成新视频；两个真实 10 步 smoke、正式训练、四条件评测和实际视频动态检查由训练机按新任务执行。已完成实验保留，不因原配方追加任务重跑；源初始化/轨迹的历史身份尚未确认时，不能保证恢复论文精确分数。


## 现有 frame/chunk 结果核查工具（2026-11-08）

新增 `experiments.iclr.audit_frame_chunk`，读取训练机现有的四个 step-200 run：核对配置、初始化、实际 checkpoint metadata 与少量 raw/EMA tensor samples、实际导出 shard 记录、推理设置、manifest 和原始评分 JSON，并核对汇总标签。它不启动训练或评分，可准备 64 个终点 raw/EMA × all1/main_ffe 视觉诊断样例的导出计划；各模式 attention/sink 设置保持相同。脚本不会自动执行 GPU 导出。

```bash
CUDA_VISIBLE_DEVICES='' /home/jiaqi/miniconda3/envs/causal_forcing/bin/python -m pytest -q tests/test_iclr_frame_chunk_audit.py --tb=short --show-capture=no
```

4 项 CPU 检查通过：官方总分贡献分解精确相加、视觉计划的配置/权重组合和 attention 不变、实际 checkpoint 步数与导出 method 错位检测、缺原始产物时返回 incomplete。原轻量结果包缺训练机权重/配置/原始评分 JSON；在该包上执行新命令正确报告 incomplete，实际读取的权重文件数为零，不将缺少证据当作发现运行错误。本次没有改动训练或正式评测配置，没有启动 GPU 作业或干扰其他 session。只读 metadata 与 sampled tensor 检查不能证明完整权重一致或动态视频正常，后续真实检查按 `ICLR_FRAME_CHUNK_AUDIT.md` 在训练机完成。

## 18 帧窗口、seed 42 的两组追加对照（2026-11-08）

用户确认 frame/chunk 两边统一 18 latent 帧完整反传窗口、seed=42、R1/R2 关闭，仅第 200 步 EMA/main_ffe/pinned-Qwen 944×5 评测。第三组独立 chunk/original-prompts 因旧公开权重 training seed 无法核实，已由用户取消。新增 profile `matched_window18_s42_v1` 与独立汇总入口，旧四组配置及 CD 设置保留。

```bash
CUDA_VISIBLE_DEVICES='' /home/jiaqi/miniconda3/envs/causal_forcing/bin/python - <<'PYTEST'
import torch, pytest
torch.cuda.current_device = lambda: 0
raise SystemExit(pytest.main(['-q', '--tb=short', '--show-capture=no']))
PYTEST
# 171 passed in 42.21s
```

- 新增 17 项测试覆盖配置隔离、真实 temporal rollout / loss mask、checkpoint 配置核验、初始化 tensor hashes、最终 EMA 选择和单 seed 配对汇总。真实训练 pipeline 以 tiny generator 替换 Wan、空 cache 替换大 cache；frame/chunk 每帧独立参数的实际梯度都精确为前三帧 0、后十八帧 1，chunk 对应 6 个完整 block。
- CPU CLI 配置生成和 launcher/train dry-run 均通过，只生成 2 个正式 run、各 200 iterations、actual gradient frames 18；没有 GPU 查询、权重加载或训练进程。
- pinned Qwen 输入核验通过：944 prompts、每个终点 4720 条 manifest；不会额外跑 original prompts。合成评分只用于验证配对计算、拒绝 raw 冒充 EMA / 不同 manifest / 缺失方法 / 重复终点；不输出 seed SD。
- CPU 命令产物：`/data/fengjiaqi/One-Forcing/iclr_window18_s42_cpu_18vtjasa/report.json`，包含配置计划和输入 hashes。这是验证产物，不是新实验分数。

没有运行真实 Wan/GPU 训练或生成新视频，也未检查本轮动态视频；这些由训练机按 `ICLR_WINDOW18_SEED42_TASK.md` 执行，每组盯到第 10 步。未停止或修改其他 session 的进程。

## 合并追加的一步局部拉直 CD（2026-11-08）

用户要求将此前新增的一步 CD 并入训练机任务，并明确只做 seed=2026 的 Original/Local Rectified 一对。新增总入口 `ICLR_CURRENT_EXPERIMENTS_TASK.md`，整轮为两个 frame/chunk run + 两个 CD run；取消的独立 chunk/original-prompts 组保持取消。同步更新 CD 子任务和旧入口，去掉本轮运行 seed=2027、报告两 seed SD 的要求。本次只改交接文档，复用既有 `cd_local` / `local_progressive_cd_all1_v1` 代码，不修改训练参数或损失。

- 隐藏 GPU 后运行 `tests/test_iclr_cd_local.py` 和 `tests/test_iclr_cd_original.py`：**25 passed in 12.63s**，覆盖局部干预、纯 CD 配方、同源预检、只评一步、checkpoint/样本配对与汇总。
- 正式 CLI dry-run：seed=2026、恰好两个训练条件，各 300 updates。Smoke dry-run：恰好两个条件，各 10 updates。评测 dry-run：恰好两个 all1 条件、raw、无 FFE、944×1；不包含 all4。
- 机器可读计划：`/data/fengjiaqi/One-Forcing/iclr_current_handoff_cpu_e6827s0n/report.json`。以上无真实资产读取或 GPU 访问，没有新增训练结果。

真实 preflight、10 步 GPU 验证、正式训练、评分和动态视频检查仍由训练机按合并任务执行；不因文档更新停止其它 session 的进程。

## 旧 seed=0 权重复用：original prompts 追加评测（2026-11-08）

用户要求复用上一轮 `frame_roff_s0` / `chunk_roff_s0`，将 Qwen prompts 改为 original 再评测，不重新训练。新增 `ICLR_SEED0_ORIGINAL_EVAL_TASK.md` 并并入总入口；沿用实际旧 step-200 EMA、main_ffe、21 帧、16 fps、944×5 samples 和相同噪声 seeds。旧反传窗口 frame 17 / chunk 21 保持历史记录，不冒充新 18 帧对照。整轮仍为 4 个新正式训练 run，增加两个 original 评测后为 6 个新终点条件。

- 隐藏 GPU 后运行 `tests/test_iclr_prompt_ablation.py` 和 main_ffe 的真实 shell/pipeline 配置检查：**7 passed in 4.49s**。本次仅改交接文档，复用已有 `prompt_ablation` CLI。
- CPU CLI dry-run 对两方法各恰好一个 original 条件，显式 EMA/main_ffe/5 samples；没有 Qwen 重评、训练或 GPU 查询。
- `--prepare-only` 为两模型分别生成相同的 4720 条 original manifest，并验证与 pinned Qwen 的 prompt_index / sample_index / prompt / seed / output_name 逐条对应，只删除 rewrite 字段。没有加载真实 checkpoint。
- CPU 验证产物：`/data/fengjiaqi/One-Forcing/iclr_seed0_original_cpu_jz6usjlp/report.json`。本机没有旧正式权重和完整导出记录；训练机须按旧 export.intent 的实际路径核实源 checkpoint/config 后执行，不用公开权重或新 seed=42 模型替换。

没有新增真实 GPU 训练、视频或 VBench 分数；本轮 C 的动态视频检查与原始评分仍待训练机完成。
