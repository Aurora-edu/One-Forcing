# 旧 frame/chunk 消融增加 training seed=2026

> 历史任务，已被[官方原代码 frame/chunk 对照](ICLR_OFFICIAL_FRAME_CHUNK_TASK.md)替代。新一轮执行官方零修改训练和推理、seed42、step200 raw，不再执行下文 ICLRTrainer/EMA 配方。下文保留供旧结果追溯。

用户要求：在旧 seeds=0/1 的 frame/chunk 消融基础上，只更换 training seed 为2026，其它设置不变。新增两个独立训练 run，沿用 ICLR 训练入口；不切换官方 main 入口，不采用后来 seed=42 的18帧反传窗口。本任务独立于 CD 和旧 seed=0 的 original-prompts 评测。

| 设置 | frame_roff_s2026 | chunk_roff_s2026 |
|---|---|---|
| Profile | main_block_ablation_s2026_v1 | 相同 |
| training seed / block大小 | 2026 / 1 | 2026 / 3 |
| gradient_num_blocks | 17 | 17 |
| 实际反传 latent 帧数 | 17 | 21（原实现裁到视频长度） |
| 正式训练 | 200 iterations：40次G、200次critic更新 | 相同 |
| R1 / R2 | 0 / 0 | 相同 |
| 终点评测 | step200、EMA、main_ffe、Qwen 944×5 | 相同 |

复用旧组同一 raw `causal_ode.pt` 初始化、14B teacher、clean LMDB；lr、GAN权重、判别器结构、EMA=0.99及第50步初始化等全部继承旧配方。固定 sampler seed仍为0，生成噪声仍使用旧 manifest；2026只改变训练随机 seed。`main_ffe`保留原 native block：frame首1帧四步，chunk首3帧所在块四步，后续一步。

只新增两组训练、两组最终VBench，共9440条视频。不补 original-prompts、raw/all1或中间 checkpoint 的 VBench。保存0/50/100/150/200步仅用于恢复和诊断。已有其它实验与结果目录保持独立。

另一个已确认任务 [GAN目标消融](ICLR_GAN_QWEN_EVAL_TASK.md) 会复用本组 `frame_roff_s2026` 的step200 **raw** 权重，并只新增一组DMD-only训练；两任务合计三组200步训练。该任务另做raw/旧版FFE/Qwen944×1评测，本表的EMA/main_ffe/944×5保持不变。两组训练启动前均接入本次初始化主干hash记录，保留同一代码快照；不要把复用理解为额外重训一遍framewise。

## 本地接手与生成配置

先接入GitHub `ICLR` 分支交接prompt指定的提交，保留其它session的工作，不执行破坏性reset。总入口为 [本轮训练与seed0日志上传](ICLR_SEED2026_TRAINING_TASK.md)；接手前确认有`SEED2026_PROFILE`及本文件，并优先上传旧seed0原始metrics。

沿用已验证的B200环境和VBench wrapper。以下路径来自旧训练机，实际位置不同时只映射路径。`OLD_ROOT`必须指旧seeds0/1实验，不指seed42目录。

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export OLD_ROOT="$R/iclr2027_main_200"
export SEED2026_ROOT="$R/iclr2027_main_s2026"
export VB_RUNNER="$R/tools/vbpy"

"$OF_PY" -m experiments.iclr.make_configs \
  --profile main_block_ablation_s2026_v1 \
  --base experiments/iclr/configs/base.yaml \
  --generator-ckpt "$ODE_INIT" --teacher "$TEACHER" --data "$CLEAN_LMDB" \
  --output "$SEED2026_ROOT/configs"

"$OF_PY" -m experiments.iclr.launch \
  --matrix "$SEED2026_ROOT/configs/matrix.json" --output "$SEED2026_ROOT/runs" \
  --python "$OF_PY" --dry-run
```

将生成配置与旧resolved_config逐项比较。除experiment_id、seed、iclr_profile及运行路径/记录外，科学设置必须一致。初始化文件内容hash、teacher及数据路径须与旧运行记录核对；同名文件不算核验通过。旧组配置如果有未纳入仓库的科学参数差异，先查明，不能悄悄套用新配方。不要将旧resolved_config中的logdir/config_name/resume字段带入新run。

## 训练与核验

仅在8卡全部空闲时启动，逐组执行，独立tmux；不停止其它session进程。每组观察到第10步，检查实际seed、loss/梯度和初始化；最终评测检查实际视频。正式训练头10步承担运行验证，不另增加正式训练组。

```bash
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$SEED2026_ROOT/configs/matrix.json" --output "$SEED2026_ROOT/runs" \
  --run frame_roff_s2026 --python "$OF_PY" --session iclr_frame_s2026 --execute

# 第一组结束且GPU释放后运行第二组。
"$OF_PY" -m experiments.iclr.launch \
  --matrix "$SEED2026_ROOT/configs/matrix.json" --output "$SEED2026_ROOT/runs" \
  --run chunk_roff_s2026 --python "$OF_PY" --session iclr_chunk_s2026 --execute

"$OF_PY" -m experiments.iclr.audit_initialization \
  --runs "$SEED2026_ROOT/runs" --profile main_block_ablation_s2026_v1
```

两组应从同一raw初始化独立训练，seed=2026的初始critic tensor hash相同。runtime_seed=2026；training.done均为200步、40/200更新。新组与旧seed0/1的generator初始hash应一致，critic初始hash无需跨seed一致。训练异常时保留日志，不自行更换seed、正则或训练预算。

## 最终评测和交付

直接复用旧Qwen prompt/rewrite/noise文件，不重新rewrite。先核对下面文件与旧Qwen实际export记录的路径及hash一致；路径不同则使用旧记录指向的真实文件。下列评测在独立tmux内顺序执行，两组训练完成且GPU空闲后启动。

```bash
for run in frame_roff_s2026 chunk_roff_s2026; do
  "$OF_PY" -m experiments.iclr.evaluate curve \
    --index "$SEED2026_ROOT/runs/$run/checkpoints.jsonl" \
    --config "$SEED2026_ROOT/runs/$run/resolved_config.yaml" \
    --prompts "$OLD_ROOT/manifests/main/vbench_prompts.txt" \
    --extended-prompts "$OLD_ROOT/manifests/main/vbench_qwen_rewrites.txt" \
    --manifest "$OLD_ROOT/manifests/main/vbench_qwen_5sample_seed0.jsonl" \
    --full-info experiments/rebuttal/assets/qwen_vbench/VBench_full_info.json \
    --samples-per-prompt 5 --output "$SEED2026_ROOT/evaluations/$run" \
    --python "$OF_PY" --vbench-python "$VB_RUNNER" \
    --gpus 0,1,2,3,4,5,6,7 --execute || break
done

"$OF_PY" -m experiments.iclr.summarize_seed2026_pair \
  --evaluations "$SEED2026_ROOT/evaluations" --output "$SEED2026_ROOT/summary"
```

生成前固定至少16个相同prompt/sample pairs，实际观看两组完整视频，检查运动、结构、冻结、闪烁和画面崩溃。只有抽帧时如实标明，不能只凭指标或成功解码宣布正常。

交付配置、来源commit及diff、初始化hash、runtime seed、完整训练日志和耗时、最终权重路径、export/protocol记录、原始16维VBench JSON、`points.csv`、`paired_comparison.csv`及实际视频检查记录。新seed单独列原始分数与frame−chunk差值；需要跨seed统计时纳入旧0/1及新2026全部三对，不能混入18帧窗口seed42，也不能只保留较好seed。
