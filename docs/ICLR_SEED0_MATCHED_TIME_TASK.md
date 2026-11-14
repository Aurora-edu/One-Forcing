# Seed 0：复用历史 Framewise，仅补跑等时间 Chunkwise

这是历史 ICLR 实验（Framewise 83.4519、Chunkwise 81.6281）的**等训练时间补充**。不是官方 seed42/raw 或 seed2027 那批实验。执行入口：`python -m experiments.iclr.matched_time`。

## 实验定义

用户已明确：Framewise 的分数和训练时间已经测过，直接复用。这次只新增 **1个chunkwise训练run + 1次最终VBench评测**。

1. 复用历史 `frame_roff_s0` 的200步结果：Total **83.45194158441419**、Quality **84.32098442924669**、Semantic **79.97577020508425**。
2. 直接使用其原始 `training_seconds = 4261.134528471157`，即 **71.0189分钟**，作为固定预算 T。
3. Chunkwise 从历史相同raw ODE初始化、同一 seed 0 独立训练，在累计训练时间第一次达到 T 的完整 iteration 后停止。不要把估算的261步写死。
4. 仅评chunk最终checkpoint：**EMA、main_ffe、固定 Qwen rewrites、944×5、全16维 VBench**。最终表将它与复用的历史frame结果比较，并标明各自来源。

历史时钟、初始化、完整精度分数及来源文件SHA256已经保存到 `experiments/iclr/assets/historical_frame200_seed0.json`；prepare时自动复制到任务目录。**不需要历史frame权重，不重训、不重新计时、不重新评测frame。** 旧版本的frame→chunk顺序训练方案已被本说明替代，CLI仅接受 `--arm chunk`，默认也是chunk。

T 使用历史日志的 `training_seconds`：每个 iteration 开始前同步，完成 forward/backward、optimizer/EMA 更新后同步 CUDA，取8个rank的最大耗时再累计。包括训练数据读取；不包括初始化、checkpoint I/O、日志/GC和评测。保存权重、日志/GC、整个训练进程的实际耗时另行报告。Chunk 最终超出 T 的量至多为最后一个 iteration，报告精确秒数。论文可称 **matched active training wall-clock time**，不要称整个作业端到端耗时严格相等。

历史chunk200为3268.6721秒（54.48分钟），本轮给chunk完整的71.0189分钟训练预算。沿用历史8卡训练条件并记录本轮GPU型号、精度、分片及软件环境；报告明确注明时间参照来自历史frame日志。

| 项目 | 固定值 |
|---|---|
| 历史源码 | `1ee818ee8535052f43868c9eecc8e4d6c4732406`；自动比对 model/pipeline/trainer/utils/wan 目录 |
| Trainer | 历史 ICLRTrainer；本次只增加时间停止、终点保存和审计，不换官方零修改训练入口 |
| seed | 字面值0，`randomize_seed=false` |
| 模型、损失 | DMD+GAN，G/D GAN系数各0.03，R1/R2均0 |
| 初始化 | 历史 framewise `causal_ode.pt` raw，SHA256 `58bd2f6c3b757d6582541c4354eff2626ea0ba4cd1fcf48c729a3d24518379bf` |
| Teacher / 数据 | 历史 Wan2.1-T2V-14B / clean-latent LMDB；历史未启用 prompt embedding cache |
| 优化 | G/critic lr=1e-5；batch=1/rank×8；每5个iteration更新1次G，每个iteration更新critic |
| 窗口 | 原样保留 `gradient_num_blocks=17`，实际 frame17帧/chunk21帧；不混入此前18帧消融 |
| EMA | decay=.99，start_step=50；chunk最终使用 `generator_ema`，与历史frame一致 |
| main_ffe | 首个原生block四步 `[1000,750,500,250]`，其余block一步；frame首1 latent、chunk首3 latent，与旧实验一致 |
| 生成 | 固定共同 prompt/noise manifest、21 latent→81 RGB帧、832×480、16FPS |

## 训练机执行

在独立 checkout/worktree 更新 ICLR，避免改动其它 session 正在执行的源码。若 shallow clone 缺少历史提交，先 fetch ICLR 的完整历史。不要启动、重跑或停止其它实验。工程路径、依赖、tmux等问题自行排查解决并继续，无需每一步询问用户。

先找到历史 `iclr2027_main_200/runs/frame_roff_s0/resolved_config.yaml` 和同目录 initialization/source 记录，核对资产。历史机器曾使用 `/data/data/cuijiaxing/oneforcing_iclr/assets/`；本机数据通常在 `/data/fengjiaqi/`，按实际位置映射。不要将公开 `one_forcing.pt` 当作训练初始化。

```bash
MT_ROOT=/data/fengjiaqi/One-Forcing/seed0_chunk_historical_time
MT_PY=/实际路径/训练环境/bin/python
MT_VBPY=/实际路径/已验证B200兼容的VBench包装器
MT_GPUS=0,1,2,3,4,5,6,7
MT_INIT=/实际路径/oneforcing_hf/checkpoints/framewise/causal_ode.pt
MT_TEACHER=/实际路径/wan_models/Wan2.1-T2V-14B
MT_DATA=/实际路径/oneforcing_hf/clean_data

"$MT_PY" -m experiments.iclr.matched_time prepare --output "$MT_ROOT" \
  --generator "$MT_INIT" --teacher "$MT_TEACHER" --data "$MT_DATA"
"$MT_PY" -m experiments.iclr.matched_time check --output "$MT_ROOT"

CUDA_VISIBLE_DEVICES='' "$MT_PY" -m pytest -q \
  tests/test_iclr_matched_time.py tests/test_iclr_restart.py \
  tests/test_iclr_protocol.py tests/test_iclr_prompt_ablation.py

"$MT_PY" -m experiments.iclr.matched_time train --output "$MT_ROOT" \
  --python "$MT_PY" --gpus "$MT_GPUS"
"$MT_PY" -m experiments.iclr.matched_time train --output "$MT_ROOT" \
  --python "$MT_PY" --gpus "$MT_GPUS" --session iclr_seed0_matched_time --execute
```

`prepare` 直接写好 `budget.json` 和 `chunk.yaml`，时间预算已来自历史日志，不依赖新的frame训练。此前按旧版两组方案准备过的目录保留原状，为本次新建任务目录。仓库相对路径 `wan_models/Wan2.1-T2V-1.3B` 必须指向同一历史资产（fake-score、text encoder、VAE）；在独立checkout中建立正确链接，不能改动其它任务共享的链接。

最后一条自动启动独立 tmux，**只跑chunk**，返回不代表任务完成。GPU占用时排队等待，不终止其它session。至少观察到第10步，检查 loss、grad 和GAN信号：

```bash
"$MT_PY" -m experiments.iclr.matched_time monitor --output "$MT_ROOT"
```

日志为 `logs/iclr_seed0_matched_time.log`；训练记录在 `runs/chunk/attempt_000/`。不根据视频或分数调整4261.1345秒的预算。

训练中断：保留失败attempt；修复工程问题后用新的tmux名字重复train命令。入口自动选择本任务最高步数的完整 `complete.json` checkpoint，在新attempt恢复 optimizer、EMA、RNG、数据位置和累计时间；不把只有 model.pt 的权重当成完整续训。不要导入历史200步权重拼接时间。Chunk 的1000步只是异常保护上限，达到上限却没达到T不算完成，需查明计时/运行异常。禁止靠更改seed、lr、数据或评测权重来改变结果。

续训会保留已保存状态，但丢弃更新的重复计算不在checkpoint时钟内。因此最终报告标出 `uninterrupted_chunk`；如需修复后重跑，仍然只重跑chunk，保留已有结果和重试成本，frame始终复用历史记录。

## 预览与最终评测

Chunk训练完成后，进入独立评测 tmux（环境变量在该shell内设置），先生成固定8个留出样例：

```bash
"$MT_PY" -m experiments.iclr.matched_time preview --output "$MT_ROOT" \
  --python "$MT_PY" --gpus "$MT_GPUS" --execute
```

实际查看 `preview/chunk/` 的MP4及首中尾帧，检查画面结构、运动和后段有无崩坏。不要只看指标或文件数。保留全部样例；工程错误定位修复后继续，若确认是真实模型退化也如实记录并完成既定评测，不挑选好视频替换。

然后在同一评测 tmux 运行：

```bash
"$MT_PY" -m experiments.iclr.matched_time evaluate --output "$MT_ROOT" \
  --python "$MT_PY" --vbench-python "$MT_VBPY" --gpus "$MT_GPUS" --execute
"$MT_PY" -m experiments.iclr.matched_time report --output "$MT_ROOT"
```

只对chunk真实终点生成4720个视频，不追加frame或中间checkpoint VBench。原始文本用于评分和文件名，Qwen文本只用于conditioning。复用历史已验证的VBench环境/包装器，不能通过升级评分库改变协议。

交付 `matched_time_results.csv/json`（历史frame与新chunk的Total/Quality/Semantic及frame−chunk）、实际步数/G与critic更新数、训练分钟数、超出预算秒数、checkpoint/日志/进程总耗时、`budget.json`、`hardware.json`、`asset_identity.json`、配置、source、初始化哈希、完整loss/grad/clock日志、step10观察记录、预览MP4与真实视频检查记录、VBench16维原始结果。表中frame标记为 `historical_reused`，chunk标记为 `new_training`。权重和全量视频留训练机，不提交GitHub。

本机在 `CUDA_VISIBLE_DEVICES=''` 下通过 **68项CPU测试**，覆盖：复用历史时间/分数、默认只跑chunk、拒绝frame/both启动、最终表来源与差值、优化器/EMA/RNG/数据恢复、时间终点保存、历史固定200步回归。GPU训练、FSDP恢复及实际视频检查在训练机完成。
