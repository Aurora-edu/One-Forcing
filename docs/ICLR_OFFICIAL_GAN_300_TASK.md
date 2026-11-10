# 官方代码 GAN 消融：两组 seed42、300 iterations、raw、官方首帧增强

用户要求训练和推理过程对齐官方代码，再将GAN权重设为0作为消融，两组均训练300步。本任务新增两个从同一ODE初始化独立训练的run；不复用或续训旧seed2026/step200权重。先前的CD与200步frame/chunk按[总任务](ICLR_CD_FFE_AND_OFFICIAL_S42_TASK.md)继续执行，不中断已有进程。

## 固定设置

| 项目 | `dmd_gan` | `dmd_only` |
|---|---|---|
| 官方代码 | `main@c9a2350e8740531562011fc9618e1a928d911ae0` | 相同 |
| GAN G / D 权重 | 0.03 / 0.03 | **0 / 0** |
| training seed | 42 | 42 |
| max_steps | 300 iterations：60次G / 300次critic | 相同 |
| 初始化、teacher、训练数据 | 同一raw `causal_ode.pt`、Wan14B、clean_data | 相同 |
| block / gradient_num_blocks | 1 / 17 | 相同 |
| R1 / R2 | 0 / 0 | 相同 |
| 评测权重 | 最终step300 raw `generator` | 相同 |
| 视频生成 | 官方exporter，首1个latent frame做4步，后续20帧各1步 | 相同 |
| prompts / samples / dimensions | 固定Qwen944×5；原始文本评分；全16维 | 相同 |

GAN组相对官方config只共同覆盖 `seed=42`、`max_steps=300`；两组之间只允许 `gan_g_weight` / `gan_d_weight` 两项不同。使用原官方 `train.py`、Trainer、DataLoader、模型、FSDP、EMA保存和采样器，**源码逐文件零修改**。原代码在GAN与R1/R2权重均为0时不建立GAN判别分支，并继续训练DMD和fake-score去噪；不新增模型分支或自定义backward，不改更新比例、学习率或损失归一化。

仍按官方规则维护EMA，但评测不传 `--use_ema`。300是外层iterations，不是300次G更新。官方 `log_iters=50` 仍会保存中间权重，但只有step300进入VBench。官方GAN分支会额外消耗随机数；相同seed不等于两组逐次训练噪声相同，保留官方行为并披露，不通过改RNG流程来强行对齐。

## 准备与CPU核验

在独立ICLR checkout接入包含本文件的提交。不要对其它session正在使用的checkout做pull/reset。外部入口 `experiments.iclr.official_gan_ablation` 负责配置、日志、监控、分片和审计，不执行模型训练计算。实际训练和生成在下面独立官方worktree中执行。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export GAN300_OFFICIAL="$R/official_main_c9a2350_gan300"
export GAN300_PAIR="$R/official_gan_raw_s42_step300"
export WAN13="$R/assets/wan_models/Wan2.1-T2V-1.3B"
export VB_RUNNER="$R/tools/vbpy"

CUDA_VISIBLE_DEVICES='' "$OF_PY" -m pytest -q \
  tests/test_iclr_official_gan_ablation.py tests/test_iclr_official_frame_chunk.py

"$OF_PY" -m experiments.iclr.official_gan_ablation prepare \
  --official-root "$GAN300_OFFICIAL" --pair "$GAN300_PAIR"
"$OF_PY" -m experiments.iclr.official_gan_ablation check --pair "$GAN300_PAIR"
```

路径可映射；ODE_INIT、TEACHER、CLEAN_LMDB、WAN13使用已核实的同一套资产。prepare/check仅CPU操作，不启动训练。入口固定官方commit并按Git对象逐文件校验，检查两份完整YAML和prompt文件，拒绝已有输出、配置漂移或误用200步frame/chunk任务目录。

两组使用相同B200环境。记录版本、已有VBench wrapper源码/hash和资产来源；初始化记录完整SHA256，数据/teacher目录的size/mtime清单不能替代内容hash。官方代码在现有环境报错时保留日志并定位依赖问题，不退回ICLRTrainer或把review分支代码复制过去。

本机已通过GAN、frame/chunk与CD入口共27项CPU检查；在隔离官方checkout完成prepare/check、两组训练dry-run、两组8分片推理dry-run及原生train.py的帮助入口检查。原生infer.sh配置转换也已在CPU运行验证，GPU exporter由占位程序替代。尚未加载真实Wan执行训练或生成视频，不能把这些CPU检查当作B200训练已通过。

## 两组独立训练

先检查dry-run的cwd为GAN300_OFFICIAL，入口为官方 `train.py`，输出是独立GAN300_PAIR目录：

```bash
"$OF_PY" -m experiments.iclr.official_gan_ablation train \
  --pair "$GAN300_PAIR" --arm dmd_gan --python "$OF_PY" \
  --generator "$ODE_INIT" --teacher "$TEACHER" --data "$CLEAN_LMDB" --wan "$WAN13" \
  --dry-run
```

8卡空闲时将 `--dry-run` 换成 `--execute`。入口建立独立tmux，拒绝占用其它会话的GPU。**至少盯到第10步**，核对实际日志及GPU状态；外部监控写 `step10_health.json` 和 `step10_objectives.json`。GAN组G/D对抗损失应出现非零值；DMD-only组两项应始终为0，fake-score去噪和DMD仍有记录。R1/R2两组均为0。

```bash
"$OF_PY" -m experiments.iclr.official_gan_ablation monitor \
  --pair "$GAN300_PAIR" --arm dmd_gan --min-step 10
```

第一组完整300步结束、`training.done`通过并释放GPU后，将上述命令的 `--arm dmd_gan` 改为 `--arm dmd_only`，同样dry-run、execute并监控至少10步。两组实际完成计数须均为300 iterations / 60 G / 300 critic。

入口核对完整日志、GAN实际开关、最终checkpoint内部 `step=300`、raw参数键/形状/有限性和hash；仅有文件名不是完成证据。官方checkpoint不含完整optimizer/RNG恢复状态，因此不从已有200步权重接着训100步。中断保留产物，使用新输出目录从共同初始化重跑。

## 官方推理与终点评测

```bash
"$OF_PY" -m experiments.iclr.official_gan_ablation evaluate \
  --pair "$GAN300_PAIR" --arm dmd_gan --python "$OF_PY" \
  --vbench-python "$VB_RUNNER" --dry-run
```

两组都只加载 `checkpoint_model_000300/model.pt` 的raw generator。外部生成的推理YAML与官方 `scripts/infer.sh` 对训练YAML的转换一致：block=1、普通schedule=[1000]、首帧schedule=[1000,750,500,250]；沿用官方默认attention和noise实现。**不使用review的首4帧联合FFE或其强制attention-window配置。**

官方exporter无分片参数，因此固定8份连续prompt shard，每份118条、每条5样本。初始噪声seed=`5*global_prompt_index+sample_index`；首帧中间重新加噪使用官方每进程全局RNG。两组采用相同分片/seed/文本，不使用旧逐视频重置中间RNG的exporter。VBench仍由已验证的评分wrapper执行，生成过程完全使用官方代码。

dry-run核对后改 `--execute`；入口自动用独立tmux运行。GAN评测完成后运行DMD-only评测；顺序使用同一批空闲8卡。每组4720条视频，共9440条，21 latent / 81 RGB frames、16fps、全16维。

视频检查按 `eval/{dmd_gan,dmd_only}/visual_samples.json` 固定16对，实际查看结构、运动、冻结、闪烁和后段退化。仅能抽帧时写明范围并将完整动态检查标为待完成，不能用编码可读代替观看。保留异常视频和分数，不以“GAN必须更高”为通过条件。

原官方exporter遇已有视频会跳过，无法恢复中间RNG，故入口拒绝继续已有生成目录。失败时保留目录日志；只在评分失败时使用保存的 `scoring_command.json` 重跑评分，不重新生成全部视频。

```bash
"$OF_PY" -m experiments.iclr.official_gan_ablation report --pair "$GAN300_PAIR"
```

`summary.json` 给出两组16维、Total/Quality/Semantic及 **GAN−DMD** 百分点差值。交付pair/configs、共享资产与环境、launch、完整train.log、step10与objective审计、training.done、checkpoint路径/hash、export计划/完成记录、原始评分JSON、汇总和16对MP4及检查记录。权重/数据集不上传GitHub；轻量结果按总任务归档。此结果是300步官方代码消融，不能混用旧200步、EMA或旧FFE分数。
