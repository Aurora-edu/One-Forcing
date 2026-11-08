# CD 实验来源与重建说明

**2026-11-08：** 用户要求补原配置 CD。新入口固定 9 帧、纯 CD、seed=2026，执行与复用条件见 [ICLR_CD_ORIGINAL_TASK.md](ICLR_CD_ORIGINAL_TASK.md)。下文“当前新实验选择”指上一轮 21 帧端点监督变体；历史初始化身份尚未闭环。

本文件记录 2026-11-06 对用户 PDF、仓库历史脚本和公开模型资产的核查。保持当前训练范围：frame/chunk 各两个 seed、每组 200 iterations；CD 两组各 300 updates。历史结果不是本轮新测量，旧 CD 成品权重尚未找到。

## 当前新实验选择

用户已指定新实验的初始化和轨迹源为 Causal-Forcing `checkpoints/framewise/ar_diffusion.pt`（下表固定 SHA256），并选择严格轨迹配对与留出历史诊断。当前配置使用 21 帧、每步端点监督、seed=0、各 300 updates，详见 [CD 一步实验设计](ICLR_CD_ONE_STEP_DESIGN.md)。这是明确的新实验选择，不能反推历史 Table 1 的输入文件。下面的 9 帧/纯 CD/seed=2026 只描述历史设置。

## 原实验已经定位

PDF 第 5 页 Table 1 与第 18 页 Appendix B 描述 original/rectified 两个 300-step checkpoint 的四步评测。对应实现为：

- [实验说明](../experiments/rebuttal/CURVATURE_CD_SMALL.md)：相邻状态一致性损失，四对相邻状态轮流训练，每对 75 次；仅用前 9 个 latent frames，framewise block=1。
- [训练配置](../experiments/rebuttal/configs/train_curvature_cd_small.yaml)：seed=2026、AdamW lr=2e-6、目标网络 EMA=0.95；目标 EMA 仅用于训练，评测使用 raw generator。
- [原运行入口](../experiments/rebuttal/run_curvature_cd_small.sh)：未覆盖训练 seed；同一对 checkpoint 分别评 all1/all4，944 个原始 prompts、每 prompt 一个样本、全部 16 维，不开 FFE。
- [训练代码](../experiments/rebuttal/train_curvature_cd.py)：online 与 target 从同一 raw 因果模型初始化，且该模型必须与生成轨迹的模型一致。

上述历史配置与入口可追溯到实验起始提交 `37317379e8009e4f61f2e671ee156d47d98d1157`；当前 trainer 增加了端点监督可选分支，旧配置默认 weight=0，保留旧目标。当前 ICLR 封装显式使用 seed=0，属于新受控重跑；不要把生成 seed=0 与历史配置中的 training seed=2026 混淆，也不能仅凭配置推断未归档的运行时覆盖。

## 已归档的历史结果

来源：[final_metrics.json](../experiments/rebuttal/results/final_metrics.json) 的 `reviewer_runbook_experiments.curvature_cd_small`，以及 [RUN_REPORT.md](../experiments/rebuttal/results/RUN_REPORT.md) 的 cd_small 完成记录。以下为百分制 Total：

| 采样 | Original | Rectified | Rectified − Original |
|---|---:|---:|---:|
| all1 | 56.79 | 55.35 | −1.44 |
| all4 | 61.16 | 71.11 | +9.95 |

四步的 Total/Quality/Semantic 均与 PDF Table 1 对上。历史记录表明一步对照已经执行，但 PDF 未展示。一步负结果应如实保留；四步改善不能单独证明高噪声曲率导致一步失败。仓库汇总并未归档原始视频、完整运行日志和模型权重，本次未重新验证这些 GPU 产物。

## 初始化与数据怎样重新准备

历史报告注明 `causal_ode.pt` 与 6,505 条 `clean_data` LMDB 来自 `JiaqiFeng/OneForcing`。这确认了该轮资产来源，但没有独立锁定 CD 命令实际传入的初始化文件；不要仅凭主模型的初始化推断 CD 使用同一份文件。

公开资产元数据已核实存在：

| 用途 | 仓库 / 文件 | LFS SHA256 |
|---|---|---|
| Framewise AR diffusion，可用于重建 CD 初始化与轨迹源 | `zhuhz22/Causal-Forcing` / `framewise/ar_diffusion.pt` | `721ea4d8f6d23099ccbefc6f5741b170402f509738f5a7b2db10e91b76857825` |
| One-Forcing 的 ODE 初始化 | `JiaqiFeng/OneForcing` / `checkpoints/framewise/causal_ode.pt` | `58bd2f6c3b757d6582541c4354eff2626ea0ba4cd1fcf48c729a3d24518379bf` |
| 原始 clean latent 数据 | `JiaqiFeng/OneForcing` / `clean_data/data.mdb` | `c9f171697a08ba30b9c8db182c7e87f4d53a16cdcc541248917c5eb902e8630c` |

这两种模型权重不同。公开 AR 权重可以用于新的配对实验，但未证明它就是历史 Table 1 的初始化。下载后核对文件 hash、raw 权重加载和数据条数，并记录所选模型；不复用来源不明的本地 `model.pt`。元数据核查未下载这些大文件。

配对数据流程完整保留：

1. [get_causal_ode_data_framewise.py](../get_causal_ode_data_framewise.py) 读取选定 raw 因果模型及 clean LMDB；使用 21 latent frames、48 步、CFG=6、shift=5、生成 seed=0+样本序号，clean latent 作为因果条件。
2. 保存索引 0/12/24/36 的状态、生成终点和独立 clean conditioning，连同真实 timesteps 与模型来源。
3. [build_curvature_dataset.py](../experiments/rebuttal/build_curvature_dataset.py) 将这些轨迹打包到一份共享 LMDB，写入 `curvature_dataset_manifest.json`。
4. [curvature_intervention.py](../experiments/rebuttal/curvature_intervention.py) 的 loader 为 rectified 组按时间线性化三个中间状态，保持噪声、生成终点、clean conditioning 和 prompt 一致；不独立采样第二套轨迹。
5. 两个 CD arm 都从生成这份轨迹的同一 raw 模型开始训练，并核对数据顺序和配对信息。

现成数据准备入口是 [run_curvature_causal_experiment.sh](../experiments/rebuttal/run_curvature_causal_experiment.sh) 的 **`prepare` 阶段**。该脚本的 `all` 还会执行另一套 1,000-step endpoint-regression 实验，不属于本轮 300-step CD 计划。本轮数据准备完成后切换到新的 ICLR CD 入口；`run_curvature_cd_small.sh` 留作历史流程参考，不用于当前计划。长 GPU 工作须在训练机独立 tmux 中执行并检查实际视频。

重建流程不依赖找回旧成品权重；精确复现历史数值仍需要核实原初始化和运行记录。当前主任务优先完成 frame/chunk 四组，CD 仍保留两个正式 run 的范围，不因这些历史记录自动增加或取消训练。
