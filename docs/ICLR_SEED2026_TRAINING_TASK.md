# 本轮训练机入口：seed2026 三组训练、四组评测及 seed0 日志上传

> 历史交接。新一轮 frame/chunk 已改为[官方原代码、step200 raw 对照](ICLR_OFFICIAL_FRAME_CHUNK_TASK.md)，不要用下文启动新的 frame/chunk 实验；其它独立任务按各自已确认的范围执行。

本文件对应最新确认的追加任务。代码位于 `Aurora-edu/One-Forcing` 的 `ICLR` 分支；使用交接prompt指定的提交或包含该提交的后续版本。保留训练机已有修改及其它session进程，不执行破坏性reset。先读取本文件，再按下面两份详细说明执行。旧总任务文件中的seed42/window18等计划不是本轮执行清单；独立CD任务仍由其原会话管理。

## 先补交旧 seed0 的训练曲线数据

用户需要原始metrics自己画train_curve。**优先打包上传现有日志，不必等新训练完成，不重训seed0。** 从旧seeds0/1实验根目录读取，历史路径为：

```text
/data/data/cuijiaxing/oneforcing_iclr/iclr2027_main_200
```

确认是旧训练seed=0的 `frame_roff_s0` / `chunk_roff_s0`，不是评测noise seed=0，也不是seed42或新seed2026。日志包必须保留以下相对路径：

```text
runs/frame_roff_s0/metrics.jsonl
runs/chunk_roff_s0/metrics.jsonl
```

两份文件按原始字节复制，保留全部字段和记录，预计各覆盖step 1–200。不要只交平滑后的CSV、截图或末尾10行；不要删失稳点、插值或补造缺失记录。每组另附现有的：

- `training_clock.jsonl`：累计训练耗时、G/critic更新次数，用于按耗时对齐。
- `resolved_config.yaml`、`runtime_seed.txt`、`initialization.json`、`source.json`：确认配置、实际seed和来源。
- `training.done`、`checkpoints.jsonl`：确认终点及训练预算。

若已有 `train_curves/formal_A.csv`，一并附上，不能替代原始JSONL。若原运行恢复过，保留各段原始日志及顺序说明，派生合并表单独存放。附 `seed0_metrics_inventory.json`，列出每份文件的来源路径、包内路径、SHA256、记录数、首末step、缺失/重复step；缺文件明确记录已查找的路径，不拿其他seed替代。记录数异常也保留原文件并说明。

建议日志包命名 `seed0_frame_chunk_train_metrics.tar.gz`。打包后检查压缩包确实含两份metrics及上述已有附件，核对包内文件hash。**上传到此前交付实验结果的Google Drive位置，并返回可访问的下载链接及压缩包SHA256。** 用训练机已有上传方式，不更改共享范围。若上传凭据/目标不可用，报告阻塞原因、已打包的绝对路径和hash；只打包在本地不算上传完成。最终新实验结果包也带上这份日志包或对应下载链接。

## 新实验：仅三组训练

执行 [frame/chunk seed2026 详细说明](ICLR_SEED2026_PAIR_TASK.md) 和 [DMD-only / GAN 详细说明](ICLR_GAN_QWEN_EVAL_TASK.md)。三组均从共享初始化独立训练，固定seed=2026，200 iterations（40次G、200次critic更新）；只评最终step200。

| 训练run | 设置 | 最终评测 |
|---|---|---|
| `frame_roff_s2026` | 旧framewise配方，仅换training seed | EMA、main_ffe、Qwen944×5 |
| `chunk_roff_s2026` | 旧chunkwise配方，仅换training seed | EMA、main_ffe、Qwen944×5 |
| `dmd_only_s2026` | 与本轮framewise相同，仅关闭GAN G/D权重 | raw、旧版ffe、Qwen944×1 |
| 复用framewise权重，无新增训练 | DMD+GAN对照 | 同一step200的raw、旧版ffe、Qwen944×1 |

合计三组训练、四组终点评测、11,328条新视频。frame/chunk保留旧 `gradient_num_blocks=17`（实际17/21帧），不改成window18。GAN消融两边均为framewise、17帧反传、raw/旧版FFE；不要把frame/chunk的EMA结果拿来充当GAN对照，也不额外训练第二个framewise GAN。固定Qwen rewrites和noise manifests，不重新改写prompts。

两份详细说明包含配置生成、dry-run、tmux启动、初始化/终点审计、评测和汇总命令。先核对共享raw初始化、teacher、LMDB与旧记录，再启动；不自行调整学习率、预算、正则、评测样本数或seed。用相同代码快照完成三组训练并保留来源。

## 执行与交付

- 沿用训练机已验证的8×B200环境；路径以实际资产位置为准。本机数据惯例为 `/data/fengjiaqi/`，B200机器旧记录为 `/data/data/cuijiaxing/oneforcing_iclr/`。
- 长任务使用独立tmux，按空闲GPU顺序执行，不终止其它session进程。每个训练run至少观察到第10步，检查实际seed及loss/梯度。
- 训练后运行配对初始化/终点审计。评测实际生成的视频必须检查：固定至少16个相同prompt/noise pairs观看完整MP4，确认结构、运动、闪烁和冻结情况。只有抽帧时如实记录检查范围。
- 上传配置、来源、metrics/计时、audit、实际export/protocol、全部原始VBench JSON、汇总、视频样例及检查记录，返回下载链接。模型权重和数据集不必放进轻量结果包，列明其保留路径。
- 本机已有97项CPU检查通过，尚未做GPU训练验证。正式训练的前10步和实际视频检查由训练机完成。
