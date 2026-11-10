# 官方原代码 frame/chunk 对照：seed 42、step200、raw

用户已确认：**官方代码完全零修改，两组改评 raw 权重。** 本文件替代旧的 `ICLR_SEED2026_PAIR_TASK.md`，需要独立重训 frame 和 chunk；不能复用此前 ICLRTrainer 的权重。

训练和视频生成均在独立的官方 `main@c9a2350e8740531562011fc9618e1a928d911ae0` checkout 中执行原始 `train.py`、`scripts/export_videos.py`。外部入口只负责配置、tmux、日志、原始权重检查、分片和 VBench，不导入或替换官方 Trainer，不修改官方 EMA 保存函数。不要把旧分支的 trainer/model/pipeline/utils 文件复制过去，也不要 monkeypatch 官方运行代码。

## 固定设置

| 项目 | frame | chunk |
|---|---|---|
| 原始训练 YAML | 官方 config.yaml | 相同 |
| 唯一组间配置差异 | num_frame_per_block=1 | num_frame_per_block=3 |
| training seed | 42，非零固定值 | 相同 |
| 预算 | 200 iterations，40 G / 200 critic | 相同 |
| gradient_num_blocks | 17 | 17 |
| 实际反传 latent 帧数 | 17 | 21，官方代码裁剪 |
| GAN G/D、R1/R2 | 0.03/0.03、0/0 | 相同 |
| 原始初始化、teacher、数据 | 同一 causal_ode.pt / Wan14B / clean_data | 相同 |
| 正式评测 | step200 raw、Qwen 944×5、16维 VBench | 相同 |
| 官方首块增强 | 第一帧4步，后续每帧1步 | 第一块3帧4步，后续每块1步 |

官方 YAML 的 seed=0 表示随机抽 seed；按用户最新要求，两组 training seed 均固定为42。这是两边相同的显式配置覆盖，不修改随机数实现。不改成18帧窗口、不增加正则、不调整EMA、不加训练prompt cache或手动backward/offload开关。训练内部照官方维护EMA，但评测明确只选 `generator` raw。评测噪声 seeds 保持原协议。

这个实验比较官方 frame 配方与“仅改 block”的 chunk 配方，包含官方反传窗口和首块长度的联动；不能称为完全隔离有效梯度帧数的比较。只测最终200步，不运行学习曲线。

## 接入与 CPU 核验

在新的或安全的 ICLR checkout 接入包含本文件的提交。不要对其它 session 的工作目录做 reset/pull。长任务用独立 tmux，不终止其它 session 进程。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export OFFICIAL_ROOT="$R/official_main_c9a2350_frame_chunk"
export OFFICIAL_PAIR="$R/official_frame_chunk_raw_s42"
export WAN13="$R/assets/wan_models/Wan2.1-T2V-1.3B"
export VB_RUNNER="$R/tools/vbpy"

"$OF_PY" -m experiments.iclr.official_frame_chunk prepare \
  --official-root "$OFFICIAL_ROOT" --pair "$OFFICIAL_PAIR"
"$OF_PY" -m experiments.iclr.official_frame_chunk check --pair "$OFFICIAL_PAIR"
```

路径可映射到真实训练机位置；ODE_INIT、TEACHER、CLEAN_LMDB、WAN13必须指原资产。prepare 创建隔离 worktree 和两份配置；不会加载模型、查询GPU或启动训练。每次训练和生成前后均按固定 Git 对象逐文件校验官方源码；即使标记 assume-unchanged 也不能绕过校验。两个训练配置除 block 外必须逐项相同。已有目录拒绝覆盖。若此前已生成 seed2026 配置，在新的 `official_frame_chunk_raw_s42` 目录重新执行 prepare。

本机仅做 CPU 验证。B200 沿用已验证的 torch2.7.1/cu128 环境及 VBench wrapper，记录环境版本和已有 wrapper 的源码/hash；这是硬件环境差异，不通过修改官方训练代码解决。若官方入口在该环境下失败，保留报错，不自动退回 ICLRTrainer。

## 两组独立训练

先 dry-run，确认 cwd 是 OFFICIAL_ROOT、命令为官方 train.py，且没有 experiments.iclr.train：

```bash
"$OF_PY" -m experiments.iclr.official_frame_chunk train \
  --pair "$OFFICIAL_PAIR" --arm frame --python "$OF_PY" \
  --generator "$ODE_INIT" --teacher "$TEACHER" --data "$CLEAN_LMDB" --wan "$WAN13" \
  --dry-run
```

8卡空闲时把 `--dry-run` 换成 `--execute`。入口创建独立 tmux；至少盯到第10步，核查损失、梯度、GPU占用，不能只看 tmux 是否存在。外部监控也会写 step10_health.json。完成后必须有 training.done，里面记录200步、40/200更新、完整raw权重检查与hash。原官方 checkpoint 不提供完整 optimizer/RNG 恢复，因此本任务不拼接断点续训。

```bash
"$OF_PY" -m experiments.iclr.official_frame_chunk monitor \
  --pair "$OFFICIAL_PAIR" --arm frame --min-step 10
```

frame 完成且GPU释放后，将训练命令的 `--arm frame` 改成 `--arm chunk`，同样先核验命令、再启动并盯到第10步。两组资产路径一致；入口锁定初始化完整hash及各资产文件大小/mtime，后者不冒充数据内容hash。训练机另保存环境、GPU型号和资产来源。

## 终点评测与视频检查

两组均用原官方生成器入口，**不传 --use_ema**。推理配置只增加官方 scripts/infer.sh 所用的首块四步 schedule。Qwen文本固定为仓库归档资产，不重新改写。

原官方 exporter 没有分片参数，外部把944条prompts连续分为8份，每份118条，使用8卡独立进程。每份process seed=首个全局prompt index×5，故所有初始噪声 seeds 仍为 `global_prompt_index*5+sample_index`。FFE中间重新加噪沿用官方每进程全局RNG，不再采用旧封装逐视频重置RNG；这点必须随报告披露，不能把新旧分数差直接归因于单个训练因素。frame/chunk使用相同分片和seeds。

```bash
"$OF_PY" -m experiments.iclr.official_frame_chunk evaluate \
  --pair "$OFFICIAL_PAIR" --arm frame --python "$OF_PY" \
  --vbench-python "$VB_RUNNER" --dry-run
```

核验后将 `--dry-run` 换成 `--execute`，独立tmux顺序完成frame，再运行chunk。每组生成4720条，共9440条，只有step200进入VBench。生成过程使用官方工作目录；VBench评分使用已有审计过的评分封装和原始scoring prompts。

入口检查raw权重完整性、所有视频数量/81帧/16fps和官方完成标记。`eval/{frame,chunk}/visual_samples.json` 固定16个相同prompt/sample pairs。**训练机AI必须实际查看这16对视频的完整动态过程**，记录清晰度、运动、冻结、闪烁和结构崩坏；仅抽帧时注明不能完整播放，不能用成功解码代替视觉检查。异常结果保留并上报，不自行换seed、延长训练或切EMA。

原官方 exporter 遇到已有文件会跳过，却不能恢复全局RNG。因此本入口拒绝覆盖或接着跑已有生成目录。导出失败时保留目录和日志，定位原因后另设新输出；若仅VBench评分失败，使用已记录 scoring_command.json 重跑评分，不重复生成视频。

```bash
"$OF_PY" -m experiments.iclr.official_frame_chunk report --pair "$OFFICIAL_PAIR"
```

交付 pair.json、configs/、launch.json、step10_health.json、train.log、training.done、环境和资产记录、raw权重路径/hash、export计划与完成记录、原始16维VBench JSON、summary.json、16对视频及真实检查记录。不要将新raw结果合并进旧EMA的跨seed统计。权重与完整视频留在训练机，不能未经要求上传大文件。
