# ICLR 本机验证记录

验证日期：2026-11-06。工作目录：`/home/jiaqi/One-Forcing`。

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
CPU 检查通过不能视为两个版本组合的 GPU 数值等价验证。8 卡正式训练应使用独立环境，遵守仓库依赖版本，四组实验使用完全相同的软件环境。

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

## 新增实现的最终 CPU 验证

- 全套测试：`python -m pytest -q`，90 passed（包含原有 53 项）。
- Python 编译和 `git diff --check` 通过。
- 11 个配置/命令/分析检查通过：生成 12 个配置、8 卡 launcher dry-run、chunkwise 入口、CD 900-step 命令、四条件 CD 评测命令、50/200-step teacher 采集计划、合成轨迹分析、两种结果汇总计划和初始化审计计划。
- 使用真实 ICLR 训练循环、tiny CPU 模型运行 10 步，并验证 step-5 保存后恢复到 step-10 与连续运行的模型、AdamW 状态、RNG、数据流位置完全一致。CUDA/FSDP collectives 在该测试中替换为 CPU 等价物，不是 8 卡验证。
- 验证 NaN 失败、不同 topology/config/缺失 rank 状态拒绝恢复、恢复后不被初始化权重覆盖。
- 几何分析使用显式标记为 `synthetic_test` 的匀速直线、变速直线、弯曲路径；CLI 必须加 `--synthetic-check`，输出标记为 `synthetic_validation`。默认 teacher 分析拒绝这些 fixtures。它们不是实际 teacher 轨迹。
- 用 synthetic VBench fixtures 验证 12 组、多时间点的 CSV/PDF/PNG 导出，并查看 PNG 排版。没有将 synthetic 指标用作论文结果。

最终机器可读命令检查记录：`/data/fengjiaqi/One-Forcing/iclr_local_validation/final_cpu_v2/report.json`。前面的临时目录是开发过程中间产物，不作为最终验收记录。
