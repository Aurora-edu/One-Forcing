# 新增 One-Forcing：官方训练200步，seed2027，raw，Qwen 944×5

与 [Self Forcing / Causal-Forcing 评测任务](ICLR_OFFICIAL_BASELINES_FFE_QWEN_TASK.md)合在同一批执行。用户明确指定 training seed 为 **2027**；生成 seeds 保持公共 manifest 的 `5*prompt_index+sample_index`。只新增一个独立 framewise DMD+GAN run，不复用已发布 One-Forcing 权重、不续训历史权重，不训练 chunkwise/DMD-only。

## 固定设置

| 项目 | 设置 |
|---|---|
| 训练源码 | 原官方 `main@c9a2350e8740531562011fc9618e1a928d911ae0`，逐文件校验、零修改 |
| 配置 | 官方 `config.yaml`，科学参数仅覆盖 `seed: 2027`；资产/logdir 路径按机器映射 |
| 训练量 | 200 outer iterations：40次 G 更新、200次 critic 更新 |
| 模型 | framewise，1 latent frame/block；DMD+GAN，GAN G/D=0.03/0.03，R1/R2=0 |
| 学习率、反传窗口等 | 保留官方：G/critic lr=1e-5、gradient_num_blocks=17、21 latent rollout；不沿用旧 ICLRTrainer 配置 |
| 初始化 / teacher / 数据 | 已核实的 framewise `causal_ode.pt` raw / Wan2.1-T2V-14B / 官方 clean-latent LMDB |
| 终点权重 | step200 raw `generator`，不使用EMA，不修改官方EMA导出 |
| FFE | 原官方采样器，首1个 latent frame四步 `[1000,750,500,250]`，后20帧各一步 `[1000]` |
| 评测 | 公共固定 Qwen prompts，944×5、81帧、832×480、16FPS、全16维 VBench |

本次训练和模型/采样器保持官方源码。外部评测入口按同一 manifest 对每个视频重置 RNG，与这批 Self Forcing / Causal-Forcing 一致；因此本次不使用历史 `official_frame_chunk evaluate` 的“每个 shard 维持全局中间噪声 RNG”导出方式。Qwen 文本只用于 conditioning，原始文本用于文件名和评分。这是新的统一评测，不承诺精确复现历史 seed42 的分数。

## 1. CPU 准备与检查

在独立 checkout/worktree 更新 `ICLR`，不要覆盖其它 session 已在运行的代码。复用上一任务中的 `BASE_EVAL_ROOT`、`BASE_EVAL_PY`、`BASE_EVAL_VBPY`、`BASE_EVAL_WAN` 和空闲8卡 `BASE_EVAL_GPUS`。训练所需新增变量按真实资产位置填写：

```bash
OF2027_OFFICIAL="$BASE_EVAL_ROOT/source/one_forcing"
OF2027_RUN="$BASE_EVAL_ROOT/training/one_forcing_s2027_200"
OF2027_INIT=/实际路径/framewise/causal_ode.pt
OF2027_TEACHER=/实际路径/wan_models/Wan2.1-T2V-14B
OF2027_DATA=/实际路径/clean_latent_lmdb

"$BASE_EVAL_PY" -m experiments.iclr.official_one_forcing200 prepare \
  --official-root "$OF2027_OFFICIAL" --pair "$OF2027_RUN"
"$BASE_EVAL_PY" -m experiments.iclr.official_one_forcing200 check \
  --pair "$OF2027_RUN"

"$BASE_EVAL_PY" -m experiments.iclr.official_one_forcing200 train \
  --pair "$OF2027_RUN" --arm one_forcing --python "$BASE_EVAL_PY" \
  --gpus "$BASE_EVAL_GPUS" \
  --generator "$OF2027_INIT" --teacher "$OF2027_TEACHER" \
  --data "$OF2027_DATA" --wan "$BASE_EVAL_WAN" --dry-run
```

核对实际命令是官方目录的 `train.py`，没有 `experiments.iclr.train`。`configs/one_forcing.yaml` 与官方 YAML 应仅 seed 不同；不改 lr、GAN 权重、窗口、更新比例、batch、数据或 teacher 来迎合分数。

`prepare` 拒绝覆盖已有训练目录；已准备过就执行 `check` 并继续，不反复执行 prepare。CLI 名称沿用 `--pair` 表示运行目录，本 profile 只有 `one_forcing` 一组。

本机已完成训练命令与8卡评测 dry-run；训练入口、训练记录接入、旧 frame/chunk 和 GAN 入口共25项 CPU 检查通过。另6项公共协议/小张量采样检查覆盖三种官方采样器，确认 One-Forcing 首帧4次、其余20帧各1次去噪。训练机可复核：

```bash
CUDA_VISIBLE_DEVICES='' "$BASE_EVAL_PY" -m pytest -q \
  tests/test_iclr_official_one_forcing200.py \
  tests/test_iclr_official_frame_chunk.py tests/test_iclr_official_gan_ablation.py
OFFICIAL_BASELINE_SOURCE_ROOT="$BASE_EVAL_ROOT/source" CUDA_VISIBLE_DEVICES='' \
  "$BASE_EVAL_PY" -m unittest discover -s tests -p test_official_baselines.py -v
```

这些是 CPU 验证，尚不包含真实 Wan 训练或 GPU 视频生成。

## 2. 独立 tmux 训练，并持续观察

上面训练命令将 `--dry-run` 换成 `--execute`。入口自动创建独立 tmux `official_train_one_forcing_s2027`。`--execute` 返回仅表示已启动，**继续监控，不在这一步结束任务**：

```bash
"$BASE_EVAL_PY" -m experiments.iclr.official_one_forcing200 monitor \
  --pair "$OF2027_RUN" --arm one_forcing --min-step 10
```

至少观察到实际第10步，检查 `runs/one_forcing/train.log`、`step10_health.json`、`step10_objectives.json`。GAN G/D 信号必须实际存在，loss/grad 有限。随后继续等待200步完成，核对 `training.done` 记录的 seed2027、200/40/200 更新计数和 raw checkpoint SHA256。

GPU忙时排队，不终止其它会话。依赖/路径/硬件问题自行诊断，保持官方源码与实验设置；不能悄悄换 Trainer、降低预算或改 seed。中断后保留全部日志：官方 checkpoint 没有完整 optimizer/RNG 状态，不把不完整恢复当连续200步。修复后在新的训练目录从相同初始化重跑，并记录原因。

## 3. 接入同一套评测

训练结束后注册本次运行。入口核验官方源码、训练配置、完整日志、GAN信号、step200/raw参数和权重hash，防止把别的 seed/预算的模型接进来：

```bash
"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py prepare \
  --work-root "$BASE_EVAL_ROOT" --model one_forcing \
  --one-forcing-run "$OF2027_RUN" --wan-path "$BASE_EVAL_WAN"

"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py infer \
  --work-root "$BASE_EVAL_ROOT" --model one_forcing \
  --phase full --gpus "$BASE_EVAL_GPUS" --dry-run
```

此处不加 `--download-weights`，不下载/替换为 `one_forcing.pt`。CPU-only 时可在训练前注册计划并 dry-run；真正 `infer` 必须有通过核验的 training.done 和匹配权重。

在独立评测 tmux 里先生成10个不同 prompts，实际查看 MP4 和 `previews/` 的首中尾帧：

```bash
"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py infer \
  --work-root "$BASE_EVAL_ROOT" --model one_forcing --phase smoke --gpus 0
```

这里的 GPU 0 必须空闲，否则使用其它空闲卡。不得只检查数值或文件存在。发现结构崩坏、黑屏、棋盘格或后段异常时，保留样例并检查实际权重、资产、dtype、采样器导入和训练日志；不要换 EMA、挑样本或延长训练来覆盖结果。工程问题解决后继续既定评测，如确认是该运行的真实退化也保留并报告。

然后正式生成和评分：

```bash
"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py infer \
  --work-root "$BASE_EVAL_ROOT" --model one_forcing \
  --phase full --gpus "$BASE_EVAL_GPUS"

"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py score \
  --work-root "$BASE_EVAL_ROOT" --model one_forcing --phase full \
  --gpus "$BASE_EVAL_GPUS" --vbench-python "$BASE_EVAL_VBPY"

"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py summarize \
  --work-root "$BASE_EVAL_ROOT" --model all
```

最后一条只在三组评分都完成后执行。保留原来两个基线的结果，不重新训练或重复生成。交付三组 Total/Quality/Semantic、16维原始 JSON、完整视频及固定样例；新增组同时交付训练配置、源码commit、资产记录、step10记录、200步完整日志、training.done、raw checkpoint路径/hash。模型大文件留在训练机，不提交 GitHub。
