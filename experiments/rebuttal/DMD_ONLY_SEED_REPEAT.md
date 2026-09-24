# DMD-only 换 seed 复现：仅训练一个模型

本次只重训 **DMD-only，training seed 固定为 1**。不重训 DMD+GAN，不增加
其它 seed，不调整参数来追求某个分数。旧 DMD-only seed 48491 的
83.515169 / 85.097562 / 77.185598（Total / Quality / Semantic）保留为比较基线。

## 固定条件

- 训练代码继续使用未修改的官方 `main@c9a2350e8740531562011fc9618e1a928d911ae0`。
- 官方 `config.yaml`，仅将 `seed` 设为 `1`、`gan_g_weight` 和 `gan_d_weight`
  设为 `0.0`。R1/R2 原本就是零，其它训练参数不变。
- 从上一次的纯 ODE checkpoint 重新开始；不加载训练过的 One-Forcing/DMD-only，
  不 resume。复用上一次的 14B teacher、Wan 1.3B 资源和 clean-latent LMDB。
- 8 张空闲 GPU、每卡 batch 1、200 次迭代、5:1 更新比（40 次 generator 更新）。
  训练 `[1000]`，每块 1 个 latent frame；**不使用 FFE 训练**。
- 保持上一次成功实验的 GPU 型号和软件环境（原实验 8×B200），避免同时改变 seed
  和环境。不要另装新版本，不添加 prompt cache、manual backward 或额外 offload。
- 评测沿用 Qwen、944 prompts × 5 samples、完整 16 维 VBench；FFE 首块 4 个
  latent frames 用 4 步，后续每块 1 帧用 1 步；21 latent / 81 RGB frames、
  16 fps、sink=0、generator 权重而非 EMA。
- **只改训练 seed，不改评测 seed**：生成 seed 仍是
  `prompt_index * samples_per_prompt + sample_index`。
- 继续遵守总计 48 小时硬上限。不要打断其它 session，不覆盖旧产物。

## 1. 独立代码目录与输出目录

不要在正在运行任务的 checkout 内 pull。使用 HTTPS 独立 clone（不需要 GitHub SSH）：

```bash
git clone --branch iclr2027-rebuttal-gan-ablation-audit \
  https://github.com/Aurora-edu/One-Forcing.git One-Forcing-dmd-seed1
cd One-Forcing-dmd-seed1
export REBUTTAL_ROOT="$PWD"
```

如果目标目录已经存在，先确认是否属于这次实验；不要删除、覆盖或从已有 checkpoint
自动恢复。先读之前的启动记录以复用相同输入和 Python 环境：

```text
/data/banyuanhao/iclr2027_ckpts/one_forcing_official_pair/runs/dmd/official_launch.json
```

从其中的 `python`、`asset_paths` 读取真实路径。以下 `/ABS/PATH` 必须替换；不要原样运行。
`VBENCH_PYTHON` 也要沿用上一次评测环境。确认旧任务结束且 GPU 空闲后再开始。

```bash
set -euo pipefail
export TRAIN_PYTHON=/ABS/PATH/previous-train-env/bin/python
export VBENCH_PYTHON=/ABS/PATH/previous-vbench-env/bin/python
export ODE_CKPT=/ABS/PATH/previous/causal_ode.pt
export TEACHER_DIR=/ABS/PATH/previous/Wan2.1-T2V-14B
export WAN13_DIR=/ABS/PATH/previous/Wan2.1-T2V-1.3B
export CLEAN_DATA=/ABS/PATH/previous/clean_data
export GPU_IDS=0,1,2,3,4,5,6,7

export REPEAT_ROOT=/data/banyuanhao/iclr2027_ckpts/one_forcing_official_dmd_seed1
export OFFICIAL_ROOT=/data/banyuanhao/iclr2027_ckpts/One-Forcing-main-c9a2350-dmd-seed1
export PAIR_DIR="$REPEAT_ROOT/configs"
export DMD_RUN_DIR="$REPEAT_ROOT/runs/dmd"
export DMD_EVAL_DIR="$REPEAT_ROOT/eval/dmd"

test ! -e "$REPEAT_ROOT"
test ! -e "$OFFICIAL_ROOT"
git worktree add --detach "$OFFICIAL_ROOT" \
  c9a2350e8740531562011fc9618e1a928d911ae0
mkdir -p "$REPEAT_ROOT"
"$TRAIN_PYTHON" experiments/rebuttal/official_main_pair.py prepare \
  --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR" --seed 1
"$TRAIN_PYTHON" experiments/rebuttal/official_main_pair.py check \
  --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR"
```

这里复用原来的配置生成器，会同时写出一份 full YAML 用于配置一致性检查；
**不启动 full 训练，也不评测 full**。只有 `dmd_only_official.yaml` 用于这次训练。

## 2. 只训练 DMD-only，并监控到至少第 10 步

```bash
"$TRAIN_PYTHON" experiments/rebuttal/launch_official_main_pair.py \
  --arm dmd --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR" \
  --run_dir "$DMD_RUN_DIR" --generator_ckpt "$ODE_CKPT" \
  --teacher_model_path "$TEACHER_DIR" --wan_1_3b_path "$WAN13_DIR" \
  --data_path "$CLEAN_DATA" --gpus "$GPU_IDS" --python "$TRAIN_PYTHON" \
  --session of_official_dmd_s1
```

训练默认通过 tmux 启动。集群如需 Slurm，沿用上次已验证的调度流程，在分配到的
8 卡计算节点执行相同命令；不要在登录节点开训练，也不要假装 tmux 启动就代表成功。
观察 `train.log` 和 GPU 使用情况，等打印到 step 10 后执行：

```bash
"$TRAIN_PYTHON" experiments/rebuttal/report_dmd_seed_repeat.py check-training \
  --run_dir "$DMD_RUN_DIR" --min_step 10
```

检查要求：step 连续、loss/梯度有限、GAN/R1/R2 loss 全为零、generator loss 等于
DMD loss、generator 在打印的 step 1/6/11/... 更新、有非零梯度。第 10 步应已记录
2 次 generator 更新和 10 次 critic 更新。继续监控至训练进程成功结束，并确认保存：

```bash
"$TRAIN_PYTHON" experiments/rebuttal/report_dmd_seed_repeat.py check-training \
  --run_dir "$DMD_RUN_DIR" --min_step 200
"$TRAIN_PYTHON" experiments/rebuttal/audit_noema_checkpoint.py \
  --checkpoint "$DMD_RUN_DIR/checkpoint_model_000200/model.pt" --expected_step 200
```

不要用 step 50/100/150 替代预先指定的 step 200，也不要因为分数不符合预期重抽 seed。

## 3. 同协议评测与视频检查

训练进程结束并释放 GPU 后，在 rebuttal checkout 中评测：

```bash
bash experiments/rebuttal/run_paper_aligned_vbench.sh \
  --name dmd_only_official_main \
  --checkpoint_path "$DMD_RUN_DIR/checkpoint_model_000200/model.pt" \
  --schedule ffe --output_root "$DMD_EVAL_DIR" --gpus "$GPU_IDS" \
  --python "$TRAIN_PYTHON" --vbench_python "$VBENCH_PYTHON"
```

不要更改 `--name`：审核工具使用这个结果文件名；新旧实验靠不同输出目录隔离。
保持相同评测软件/模型缓存和已记录的 B200 环境处理；若遇错误，报告错误，不缩减维度、
样本数或换推理方式。

必须实际查看新旧 DMD-only 的相同 prompt/sample 视频，至少包括上次检查的 5 个：
`a dog and a horse`、`a car and a motorcycle`、
`A cute happy Corgi playing in park, sunset, black and white`、
`a shark is swimming in the ocean, black and white`、
`Two pandas discussing an academic paper.`，均取 sample 0。
观察全段运动并检查开头/中间/结尾，记录运动幅度、结构、主体身份是否崩溃；不能只看指标，
不能把旧报告的视觉结论当作这次观察。视频及抽帧留在本地。

## 4. 单独汇总、上传小文件

```bash
"$TRAIN_PYTHON" experiments/rebuttal/report_dmd_seed_repeat.py report \
  --official_root "$OFFICIAL_ROOT" --pair_dir "$PAIR_DIR" \
  --run_dir "$DMD_RUN_DIR" --eval_root "$DMD_EVAL_DIR" \
  --output_prefix experiments/rebuttal/results/official_main_dmd_seed1
```

会生成 `official_main_dmd_seed1.json` 和 `.md`，报告 **DMD-only seed 48491 与
seed 1** 的 Total/Quality/Semantic、全部 16 维及差值。审核训练到 200 步、40 次
generator 更新，并验证与原实验的评测协议相同。它不需要 full 的新训练结果，
也不会把旧 full 与新 dmd 的差值写成配对 GAN 增益。

另写 `experiments/rebuttal/results/official_main_dmd_seed1_visual.md`，仅写实际观察、
实际代码 commit、环境/GPU、耗时、成功/失败状态；不要填预期分数。
在同一分支提交并 push 这三个小文件，保留原来的报告：

```bash
git add experiments/rebuttal/results/official_main_dmd_seed1.json \
  experiments/rebuttal/results/official_main_dmd_seed1.md \
  experiments/rebuttal/results/official_main_dmd_seed1_visual.md
git commit -m "Report official DMD-only training seed 1 repeat"
git push origin HEAD:iclr2027-rebuttal-gan-ablation-audit
```

不要上传权重、视频、数据、缓存或完整日志。HTTPS push 使用训练机已有凭据；如缺少
写权限，保留本地结果并报告，不安装 gh、不把 token 写进代码。若远端有新提交，先在
这份独立 checkout 合并远端更新，检查冲突后普通 push，不 force push。
