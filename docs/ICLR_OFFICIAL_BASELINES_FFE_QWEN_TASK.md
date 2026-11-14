# 官方 Self Forcing / Causal-Forcing 重评 + One-Forcing seed2027 训练200步

本批任务有三组：Self Forcing / Causal-Forcing 使用官方公开权重直接评测；**另新增一次 One-Forcing 官方代码训练，seed=2027、200 iterations、framewise、raw，再按相同 Qwen 944×5 协议评测**。新增组按 [One-Forcing 执行说明](ICLR_ONE_FORCING200_S2027_TASK.md)完成，不额外训练 chunkwise 或 DMD-only。

在 `Aurora-edu/One-Forcing` 的 `ICLR` 分支执行，公共评测入口为 `experiments/rebuttal/run_official_baselines.py`。路径、依赖、下载和资源调度问题自行解决，保留诊断日志。不要停止其它 session 的进程，也不要改动已有实验结果。如果两个基线已在运行，让其继续在原 checkout 中完成；用新的 checkout 准备新增组，复用已完成的视频和评分，不为新增组重跑基线。

## 1. 要评什么

使用原版官方公开权重，不用我们自己训练的 DMD-only，不替换为 Causal-Forcing++。

| 方法 | 官方文件 | 加载的参数 |
|---|---|---|
| Self Forcing | `gdhe17/Self-Forcing/checkpoints/self_forcing_dmd.pt` | `generator_ema`（官方推理示例使用 EMA） |
| 原版 Causal-Forcing chunkwise | `zhuhz22/Causal-Forcing/chunkwise/causal_forcing.pt` | `generator`（官方 chunkwise 示例使用 raw） |

这两个文件是**原生四步公开 checkpoint**。本任务测它们在**一步采样 + FFE** 下的表现，不宣称它们是重新训练的一步模型，也不直接宣称复现了旧 Table 2 的 77.18 / 78.39。结果应标为“official released checkpoint, reevaluated with one-step sampling + FFE”。两组采用同一生成/评分协议，参数分支遵循各自官方发布建议并如实注明。

下面先说明两个公开基线。新增 One-Forcing 采用相同 prompts、生成 seeds、视频规格和评分协议，但保留其原生 framewise / 首1帧 FFE（而两个基线是 chunkwise / 首3帧 FFE）。共同的 FFE 策略是首 native block 四步、后续一步，不把两种首块长度写成完全相同。

两个公开基线的协议：

- 保留各自官方模型实现和原生 **3 latent frames/block**。首个完整 block 用 `[1000,750,500,250]`，后续 block 用 `[1000]`；由官方 scheduler 做 timestep warp。这里 FFE 加强的是首 **3** 个 latent frame，不能写成只加强第 1 个 latent frame。
- 21 latent frames，解码为 81 帧，832×480，16 FPS，BF16；保持官方 KV cache、重加噪、context_noise=0、VAE、attention 行为。
- Qwen 改写只用于 text conditioning。文件名和 VBench 的标签/评分 metadata 始终用原始英文 prompts。
- 使用仓库固定的 **同一份 944 条 Qwen 改写**，不重新调用 Qwen。每 prompt 5 个样本，每组 **4720 个视频**，两组共 9440 个正式视频。
- `seed = prompt_index * 5 + sample_index`，每个视频单独设种子。换 GPU 数量、断点续跑不改变种子。
- 完整 16 维 VBench；按官方 normalized aggregation 输出 Total / Quality / Semantic，同时保留原始 JSON。

来源已经固定在 `experiments/rebuttal/official_baselines.lock.json`，包括 Git commit、Hugging Face revision、文件 SHA256 和 weight key：

- [Self Forcing 官方说明](https://github.com/guandeh17/Self-Forcing/tree/33593df3e81fa3ec10239271dd2c100facac6de1)
- [Causal-Forcing 官方说明](https://github.com/thu-ml/Causal-Forcing/tree/eb8705b2f00b7c361193b25aa3aae9e7edc94667)

代码从两个隔离的官方 checkout 导入模型，不使用本仓库的 One-Forcing 模型类。Causal-Forcing 使用官方 `denoising_step_list_first_chunk` 接口，模型源码不改。Self Forcing 只在内存中对其官方采样循环增加 per-block 步数选择；实际执行源码保存到 `audit/pipeline.shard*.py`。所有参数 `strict=True` 加载；不允许缺 EMA 时回退 raw 或用 `strict=False` 忽略模型参数。

## 2. 准备

使用独立 checkout/worktree，保留训练机已有修改。沿用已有 B200 推理环境及 VBench wrapper（例如已有的 `tools/vbpy`）；不要为了版本号强装不支持 B200 的旧 PyTorch。记录 torch/CUDA/torchvision/flash-attn/VBench 版本及 wrapper 源码。入口直接兼容可执行的 Python wrapper。

下面路径按实际机器映射；`BASE_EVAL_WAN` 指 Wan1.3B 目录本身，不是其父目录。此评测不需要 14B teacher 或训练数据。

```bash
BASE_EVAL_ROOT=/data/fengjiaqi/One-Forcing/official_baselines_ffe_qwen
BASE_EVAL_PY=/实际路径/推理环境/bin/python
BASE_EVAL_VBPY=/实际路径/tools/vbpy
BASE_EVAL_WAN=/实际路径/wan_models/Wan2.1-T2V-1.3B
BASE_EVAL_GPUS=0,1,2,3,4,5,6,7

"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py prepare \
  --work-root "$BASE_EVAL_ROOT" \
  --wan-path "$BASE_EVAL_WAN" --download-weights
```

`prepare` 下载两个官方权重（各约 5.68 GB）、固定官方源码、准备 Qwen manifest，并在独立 runtime 目录链接现有 Wan1.3B 资产。网络中断可用相同命令续传。如果已有权重，可提前软链接到下面位置，再省略 `--download-weights`；文件内容 SHA256 必须相同：

```text
<ROOT>/weights/self_forcing/checkpoints/self_forcing_dmd.pt
<ROOT>/weights/causal_forcing/chunkwise/causal_forcing.pt
```

CPU-only 准备可以省略 `--wan-path` 和 `--download-weights`。它不会生成任何视频。然后运行：

```bash
OFFICIAL_BASELINE_SOURCE_ROOT="$BASE_EVAL_ROOT/source" \
  "$BASE_EVAL_PY" -m unittest discover -s tests -p test_official_baselines.py -v

"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py infer \
  --work-root "$BASE_EVAL_ROOT" --phase full \
  --gpus "$BASE_EVAL_GPUS" --dry-run
```

## 3. GPU smoke：必须检查实际视频

长评测放到新建的独立 tmux，如 `official_baselines_ffe_qwen`。只使用空闲 GPU；没有空闲卡时安排稍后继续，不抢占其它任务。入口只管理自己创建的子进程。

先在一张空闲 GPU 上生成每个模型的 10 个不同 prompt：

```bash
"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py infer \
  --work-root "$BASE_EVAL_ROOT" --phase smoke --gpus 0
```

这里无训练；至少跟到每个模型的第 **10 个视频**。查看 `results/<model>/ffe/smoke/logs/shard0.log` 和 `audit/environment.shard0.json`，核对正确模型代码、strict load、参数分支、实际 warped timesteps。随后播放 MP4，并查看 `previews/` 下覆盖首、中、尾帧的 contact sheets，检查是否有黑屏、棋盘格、结构崩溃、后段突然变坏等问题。数值有限和可解码不能代替肉眼检查。

如果 FFE 视频异常，先用同一权重、同一 prompts/seeds 的原生四步做诊断，不通过训练或换模型来掩盖问题：

```bash
"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py infer \
  --work-root "$BASE_EVAL_ROOT" --model self_forcing \
  --phase smoke --schedule all4 --gpus 0
# 需要时对 causal_forcing 执行相同命令。
```

`all4` 是独立诊断目录，不进入主表汇总。若四步也坏，排查模型/资产、导入路径、flash-attn、dtype、权重加载。若四步正常而一步退化，则保留视频和结论，这可能是公开四步 checkpoint 压到一步的真实结果；继续完成既定一步评测，不按预期分数筛选视频或替换参数。工程问题自行修复并记录 diff；问题尚未解决时保持诊断任务，不把失败结果标成通过。

## 4. 正式生成与评分

在 smoke 视频检查后执行；两个模型依次使用同一组空闲卡，每张卡一个独立生成进程，按 manifest 分片：

```bash
"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py infer \
  --work-root "$BASE_EVAL_ROOT" --phase full --gpus "$BASE_EVAL_GPUS"

"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py score \
  --work-root "$BASE_EVAL_ROOT" --phase full --gpus "$BASE_EVAL_GPUS" \
  --vbench-python "$BASE_EVAL_VBPY"
```

可用 `--model self_forcing` / `--model causal_forcing` 单独执行。重复 `infer` 会检查已完成视频的种子、prompt、运行协议和 SHA256，再跳过；不会因为一个 MP4 存在就无条件复用。两个模型不会混用输出目录。若写 MP4 时中断导致单个视频缺 JSON，将该不完整文件移入诊断目录后重跑；不要删除整个实验，也不要变更 manifest。

`score` 复用 `scripts/run_vbench.py` 的标准覆盖检查和全 16 维评分。沿用之前已跑通的 B200 wrapper 处理 PyTorch 兼容性，保持 VBench 计算与归一化不变。新增代码没有放宽评分依赖检查；如 wrapper 失效，自行修复环境兼容并归档实际版本，不能用缺维度均值顶替 Total。

## 5. 交付

新增 One-Forcing 完成后，执行以下命令将三组放入同一份汇总（不会重新生成或评分）：

```bash
"$BASE_EVAL_PY" experiments/rebuttal/run_official_baselines.py summarize \
  --work-root "$BASE_EVAL_ROOT" --model all
```

`--model both` 仍只选择 Self Forcing / Causal-Forcing，`--model one_forcing` 只选择新增组，`--model all` 选择三组。三组各4720个正式视频，总计14160个。完成后交付 `<ROOT>/summary.csv`、`summary.json` 及以下原始文件：

- `official_sources.json`、固定 prompts/rewrites/manifest、运行命令、仓库 commit。
- 每组 `intent.json`、`audit/`、`logs/`、`metadata/`、`export.done.json`、`score_provenance.json`。
- 每组 `vbench/*_eval_results.json`、16 维结果和评测环境记录。
- smoke MP4 与 contact sheets，以及正式视频目录。保留全部视频；分享包太大时将原始视频单独归档，并给可访问链接。
- 简短说明实际视频是否正常，公开四步权重在一步下是否退化，工程修复有哪些。不要把 smoke 质量检查冒充正式 VBench。

这批结果补充的是“官方公开权重在统一 Qwen + FFE 协议下重新评测”的证据。它不追溯改变旧 Table 2 分数的来源，也不新增 ASD 结果。

## 本机验证范围

已通过 6 项 CPU 测试，包括固定 944×5 的 prompt/seed/分片覆盖、strict 参数加载、断点结果校验，以及执行两份锁定的官方采样循环（用小张量/假模型替代大模型）。验证了首 block 4 次去噪、后续 6 个 block 各 1 次、7 次 KV context refresh；再次生成时重置正确。四步控制模式与各自官方原始循环的 CPU 输出和调用序列完全一致。

本机尚未加载真实官方权重或在 GPU 上生成视频；真实 Wan/B200 运行和视频检查由训练机完成。
