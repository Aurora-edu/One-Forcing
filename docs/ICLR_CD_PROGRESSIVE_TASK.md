# 独立实验：Original / Rectified 确定性渐进蒸馏

本文件是本轮训练机的完整任务。新增独立输出目录，保留之前 CD、frame/chunk、GAN 消融及全部结果。只做下面这轮；不触发旧合并任务，不追加 Qwen。

## 0. 最新授权：持续解决问题，不停在等待确认

用户的目标是：**在真正面向一步生成的实验设置下，争取让 Rectified 的质量和 VBench 分数明显高于 Original，并查清此前一步收益没有体现出来的原因。** 用户关心的机制是：上轮1000步实验虽然组间参数匹配，但训练从四个噪声时刻随机取一个、直接回归相同端点；rectification 改变的中间输入在后续帧的一步推理中不会被直接调用。仅观察这些中间状态训练对纯噪声入口的间接影响，没有充分回答“拉直后的多步行为是否更容易压缩成一步”。这是本轮要实际检验和解决的实验设计问题，不是已经证实的唯一失败原因。

当前主线是 **Original/Rectified 四步起点 → 各自确定性4→2→1蒸馏 → 纯一步配对评测**。让中间状态参与 teacher 的实际两次更新，并将这两次更新蒸馏成 student 的一次更新。不要把任务退回“在旧多时刻端点回归上多训一会儿，再期待一步分数自动提高”。

**最新指令覆盖此前所有“先询问/等待用户确认恢复、修复或重试”的要求。** 用户授权训练机 AI 自主诊断、修复代码/环境/资产路径/采样器、安排必要的配对恢复或训练、复测并继续推进。报告方案和成本是进度记录，不是等待批准。视频检查失败、缺权重、导出错误、某个阶段失败，都应进入下面的诊断循环；不能仅回复“按文档停在这里，等待确认”就结束任务。

### 已知状态：从现有产物接着做

- 据用户转述，训练机已经恢复了 Original/Rectified 四步起点；不要因为新读到这份文档就再次恢复整对。
- 训练机报告确定性四步下 Original 为0/16可用，Rectified 为1/16可用。这是训练机报告，尚未由主线程检查视频；需保留判定依据，区分画面崩溃、语义不符、低清和静止场景。
- 4→2、2→1和最终VBench尚未启动；节点2807已释放，其它任务没有受影响。AI自行通过现有调度方式取得空闲资源，无需为正常申请推理或训练资源再问用户。
- 当前未知的是历史重新加噪的 `legacy4` 结果。`progressive_export` 的 source 阶段原本会生成 `legacy4` 和 `deterministic4` 两套视频；先查已有文件和 receipt，不重复申请GPU重导出完整64例。
- 历史四步采样每步重新加噪，新方案使用确定性 flow/DDIM 更新。**恢复旧训练配方并不保证换采样器后仍然可用**，这是优先诊断的兼容性问题；不能只凭0/16、1/16就认定恢复训练本身崩溃。

### 自主诊断、修复、复测循环

1. **先用现有证据定位。** 查找已完成 run、权重 hash、训练前10步与完整loss/gradient日志、实际代码与模块路径、runtime、source清单，以及两套已有预览。实际看视频并记录首帧、中段、末段，不能仅引用可用数量。
2. **用最小推理对照区分原因。** 如现有证据不足，复用两份300步raw，只取固定留出集前4例；通过历史原生 `CausalInferencePipeline.inference` 和 `experiments/rebuttal/configs/eval_all4.yaml` 生成对照。原生all4与自定义legacy4使用相同初始latent及起始RNG状态，比较各步latent和最终视频。
3. **按证据处理，不把分类报告当任务终点。** 原生all4正常而自定义legacy4失败，查新增导出/调度/缓存实现；两者正常而deterministic4失败，查确定性采样转换及teacher训练目标的兼容性；原生all4也失败，查恢复配方、初始化身份、权重加载、训练与生成历史条件、数据和VAE。初始化模型的基线用其自身已验证的原生采样协议，不假定它本来就能四步生成。
4. **修复并实际复测。** 能修代码就修代码，路径/环境问题自行解决；需要重新准备同源数据、配对恢复teacher或建立确定性teacher转换阶段时，先在本地记录假设、方案、两组共同设置与预计成本，再执行最小验证，无需询问。不能靠每次固定随机seed，就把原先含额外随机重加噪的teacher直接宣称为当前公式所需的确定性flow teacher。
5. **修复有效后接回主线。** 首4例通过后检查完整固定16例，再接4→2、两步预览、2→1、一步预览和最终评测。每次训练均用独立tmux并实际观察至少到第10步。某阶段出现问题，保留产物并继续该阶段的诊断循环；不直接把异常teacher传给下游。
6. **科学设置需要改动时也自主推进，但保持配对。** seed2026、300次更新/阶段、lr=2e-6为首轮基准。确有训练目标、采样兼容性或收敛证据要求改变窗口、日程、预算或目标时，创建独立版本，记录理由及两组相同修改，并同步更新代码、校验器、测试和manifest。现有v1代码硬性校验其配置，不能修改旧metadata或删除检查来冒用v1。额外实验的时间与更新数单独计入成本。

使用固定的开发/留出样例解决设置问题；正式方案和最终checkpoint选择规则确定后，运行相同的 original prompts 944×1。保留每个版本的两组结果及失败记录；不要通过单独削弱Original、选择有利seed/样例/权重或隐藏低分来制造差距。若最终差距仍小，继续分析最高噪声预测、teacher到student的质量损失和自回归误差；需要新的实验时基于具体假设设计成对验证，不反复盲跑同一失败设置。

**脚本的质量/来源校验阻止的是无效的下游计算，不是要求用户重新授权。** 遇到检查失败，AI应修复原因再复测，不能把 `usable=false` 直接改成true放行。空闲资源暂不可用时自行排队/等待，继续CPU审计、代码修复和结果整理，保持进度记录。不要终止其它session的进程。

## 1. 先找到并核实四步起点

历史对照见 [来源审计](ICLR_CD_PROVENANCE.md)、[历史配置](../experiments/rebuttal/configs/train_curvature_cd_small.yaml)、[历史汇总](../experiments/rebuttal/results/final_metrics.json)。该实验是 **seed2026、300 updates、前9 latent frames、纯相邻状态 CD**，两组仅全局 trajectory rectification 干预不同；训练 target EMA=0.95，评测 raw。

归档四步 Total 为 Original **61.16** / Rectified **71.11**。这些是历史汇总记录，尚不能替代本次权重身份和生成质量核验。本机没有找到可验证的这对历史权重及其完整运行记录。训练机优先查 `eval/reviewer/curvature_cd_small`、旧训练归档、原配置9帧重跑目录，而非按 `model.pt` 文件名判断身份。

**不得直接采用本轮1000步端点监督权重，也不得采用300步局部高噪声拉直权重。** `original` 对应旧命名 `curved`；`rectified` 必须对应历史全局中间状态拉直。核对同一初始化 hash、trajectory manifest hash、300步 raw 参数、配套 `run.intent.json` 和配置。只有同名路径、汇总分数或无来源的导出文件不够；不补造历史 metadata。

CPU 盘点示例：

```bash
"$OF_PY" -m experiments.iclr.progressive inventory \
  --roots "$R/iclr2027_main_200" /实际历史训练归档目录
```

该命令只列候选路径，**不会认定权重已经核验，也不会恢复训练**。不要把不存在的示例路径原样传入。复制 [来源清单模板](../experiments/iclr/configs/cd_progressive_sources.example.json) 到新实验目录以外，填写实际绝对路径、实测 SHA256 和真实记录路径；恢复重跑的权重必须将 `origin` 写成 `recovered_four_step_cd`。

若历史记录缺少初始化 hash，或原权重/原数据找不到，先记录缺口和下表成本，再自主执行有来源依据的恢复。不要为了通过校验改写历史 checkpoint 或日志。恢复同一9帧配方的新四步对照，应明确标为恢复重跑，不能称为找回了历史原权重。已有恢复产物优先复用和诊断。

| 情形 | 新增成本与下一步 |
|---|---|
| 找到可核验且生成正常的历史四步对照 | 直接做本轮4个阶段 run，共1200 updates |
| 缺失四步权重，但有 hash 匹配的初始化与共享轨迹 LMDB | 先恢复2个300步四步 teacher，**额外600 updates**；再检查四步视频，合计训练1800 updates |
| 初始化或配对轨迹也缺失 | 自主核实源模型与数据并优先复用；确需重建6505条48步轨迹，先记录这笔独立采样成本再执行 |
| 找到了权重但四步视频异常 | 保存异常产物，进入第0节诊断循环，自行修复、复测，合格后继续蒸馏 |

恢复和必要修复已获授权：方案及额外成本先写入进度记录，随后执行，不等待用户确认。恢复入口为 `experiments.iclr.cd_original`，详见 [9帧原配置任务](ICLR_CD_ORIGINAL_TASK.md)。只恢复确实缺失或已证实需替换的训练 run；本轮不因此追加旧任务的四个完整 VBench。可先做只读/CPU 预检：

```bash
"$OF_PY" -m experiments.iclr.cd_original \
  --data "$CD_DATA" --generator-ckpt "$CD_INIT" \
  --output /新的独立四步恢复目录 --preflight
```

## 2. 首轮基准设置

| 项目 | Original / Rectified 均相同 |
|---|---|
| Profile | `original_rectified_progressive_v1` |
| 顺序 | 各自核验过的4步 teacher → 2步 student → 1步 student |
| 训练预算 | **每阶段300次 optimizer updates**；2组×2阶段=4个阶段 run，共1200次 |
| Seed / 优化器 | 2026；AdamW lr=2e-6，betas=(0.9,0.999)，weight decay=0.01，clip=10 |
| Batch / 训练窗口 | 8卡×每卡1；沿用历史训练的**前9 latent frames**，block=1 |
| 数据 | 相同 clean latent LMDB；排除前64行及这些 prompt 文本的重复行 |
| 固定留出样例 | 前64行中 row=0,4,...,60 共16例，生成 seed=2026+row；两组、各阶段固定 |
| 权重 / 目标 | 冻结上一阶段的 raw teacher；student 同源初始化；新 AdamW；无 EMA target / EMA 导出 / GAN / DMD 损失 |
| 训练条件 | clean 过去帧 teacher forcing；推理使用自身生成的过去帧，不输入真实视频 |
| 推理长度 | 21 latent frames → 81视频帧，832×480，16fps，block=1 |
| 最终评测 | **raw、纯 all1、无 FFE、original prompts 944×1**，全16维 VBench |

留出仅针对本轮渐进训练：历史 teacher 可能见过这些数据，不能宣称是历史 teacher 的独立测试集。训练窗口取9帧是为了沿用历史四步训练窗口，避免额外改成21帧；正式生成仍与历史评测相同为21 latent frames。

在此基准版本中，Rectified 的含义由**四步起点的训练来源**决定。渐进阶段两组使用相同算法，不额外手工拉直或加入其它损失。需要基于诊断修改方案时，按第0节建立独立配对版本；结果无论方向如何均完整保留。正式VBench用于报告预先确定方案的结果，不用于反复挑seed、checkpoint或事后修改预算。

## 3. 确定性采样和训练目标

参考 [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) 的确定性采样逐次减半思想，在当前 flow 参数化中显式实现。令 `x_t=(1-t)x0+t*noise`，teacher 一次确定性更新为：

```text
x_s = (s/t) x_t + (1-s/t) D_teacher(x_t,t)
```

连续两次 teacher 更新得到 `x_end`，student 的 clean 目标为：

```text
x0_target = [x_end - (end/t) x_t] / [1 - end/t]
loss = mean((D_student(x_t,t) - stopgrad(x0_target))²)
```

两次 teacher 之间不重加噪；固定输入与 teacher 应给出可重复的 target。每个阶段第一步实际检查重复 teacher forward 与 RNG 不变。该保证针对**采样路径确定性**，不宣称 GPU backward 跨设备逐位一致。

使用实际 shifted sigma，**不再次 warp**：

```text
4步: 1, 0.9375, 5/6, 0.625, 0
2步: 1, 5/6, 0
1步: 1, 0
```

4→2 的两个粗区间交替，各150次更新；2→1 为同一区间300次更新。每步全局样本行、prompt、噪声 hash 累积入配对摘要；两个 arm 不一致时，停止使用该无效配对并自主修复数据/随机流问题，再开展有效配对。旧四步推理会在步间重新加噪，**不同于此次确定性 sampler**；因此起点同时输出旧 sampler 与确定性 sampler 的同一批视频，核实新 sampler 下的起点质量。

## 4. 训练机执行顺序

先保留机器本地改动，更新 `ICLR`，记录实际提交。优先独立 checkout/worktree，禁止 `reset --hard` 覆盖其它任务。沿用已跑通的 B200 环境和 `wan_models/` 资产链接，记录 torch/CUDA/flash-attn 版本。下面路径变量按实际环境填写，示例可参考：

```bash
source /data/data/cuijiaxing/oneforcing_iclr/tools/env.sh
export PD_ROOT="$R/cd_progressive_s2026_v1"
export PD_SPEC="$R/cd_progressive_s2026_sources.json"
export PD_DATA="$CLEAN_LMDB"
export PD_VB_PY="$R/tools/vbpy"
```

首次 `prepare` 时 `PD_ROOT` 必须尚不存在；`PD_SPEC` 在其外部。已有完整目录应从未完成的诊断或阶段继续，不重复prepare或重跑已完成训练。`OF_PY` 为训练环境 Python，`PD_DATA` 是 clean LMDB，不是 CD trajectory LMDB。配置和 manifest 生成后不可修改；修复改变了配置/代码hash时创建新目录并保留旧结果，显式引用仍有效的现有资产，避免无理由重复恢复teacher。

先 CPU 预检：

```bash
CUDA_VISIBLE_DEVICES='' "$OF_PY" -m pytest -q tests/test_iclr_progressive.py
"$OF_PY" -m experiments.iclr.progressive prepare \
  --source-pair "$PD_SPEC" --data "$PD_DATA" --output "$PD_ROOT"
```

准备只核查来源、数据、prompts、配置与代码 hash，**不构成视频质量通过**。GPU 操作默认 dry-run；以下 `--execute` 会检查8卡空闲并创建独立 tmux。已有进程占卡时等待或换时段，不停止其它会话。

### A. 四步起点视频

```bash
"$OF_PY" -m experiments.iclr.progressive preview --phase source \
  --output "$PD_ROOT" --python "$OF_PY" --gpus 0,1,2,3,4,5,6,7 \
  --session pd26_source --execute
```

共2组×2采样器×16例=64视频。在 `previews/source/` 中检查全部配对 MP4，关注首帧质量、后续结构、运动、闪烁、融化和重复静止；同时提供12帧 contact sheet 便于比较。检查完将 `review.template.json` 的副本写入 `reviews/source.json`：保留 `export_sha256`，逐个填写 `usable`、`method` 和有内容的 `notes`。`method` 只能是 `video_playback` 或 `temporal_frames`。确实只看逐时刻画面时必须保留 `dynamic_playback_pending=true`，明确说明未完成动态播放；不得声称“已播放”。机器可解码、分数有限或文件存在不能代替画面检查。

`usable` 指没有画面崩溃，不要求每例语义完美，也不要求 Rectified 的主观质量更高。若某个视频异常，不把 `usable` 改成 true 绕过检查。保留全部视频、说明失败样例和哪种 sampler 异常，立即进入第0节的自主诊断与修复循环。两组都在确定性四步下可用后进入 B；不把起点失败作为结束任务或等待确认的理由。

### B. 4→2，两组各300步

```bash
"$OF_PY" -m experiments.iclr.progressive train --stage 4to2 \
  --output "$PD_ROOT" --python "$OF_PY" --gpus 0,1,2,3,4,5,6,7 \
  --session pd26_4to2 --execute
```

两组依次占用8卡。训练机 AI 必须分别观察 Original 和 Rectified **至少到第10步**，查看 `logs/pd26_4to2.log`、`training/4to2/<arm>/metrics.jsonl` 和 `step10_health.json`，确认 loss/gradient 有限、teacher 重复检查通过、输入形状和实际更新数正常。不能启动后直接离开，也不能把自动 health 文件冒充人工观察记录。把观察到的 step、数值与时间写入 `operator_notes.md`。

用实际前10步训练耗时估算300步耗时，再加加载/保存时间；两组、下一阶段分别更新估计。不能直接给出“2–3天”。结束后：

```bash
"$OF_PY" -m experiments.iclr.progressive preview --phase 4to2 \
  --output "$PD_ROOT" --python "$OF_PY" --session pd26_preview2 --execute
```

输出32个两步视频。对照相同样例的四步 teacher，逐项记录 `reviews/4to2.json`。未检查就继续检查，明显崩溃则自主诊断、修复、复测；合格后进入下一阶段，不停在向用户询问是否继续。

### C. 2→1，两组各300步

```bash
"$OF_PY" -m experiments.iclr.progressive train --stage 2to1 \
  --output "$PD_ROOT" --python "$OF_PY" --session pd26_2to1 --execute
"$OF_PY" -m experiments.iclr.progressive preview --phase 2to1 \
  --output "$PD_ROOT" --python "$OF_PY" --session pd26_preview1 --execute
```

**第二条需等第一条训练完成后执行**；launcher 后台返回不等于训练完成。两组仍分别盯到第10步。student 从各自两步 raw 结果初始化，teacher 固定为该两步结果，优化器重置；不能交叉使用另一组权重。32个一步视频检查后填写 `reviews/2to1.json`，再进入 D。

### D. 终点正式 VBench

```bash
"$OF_PY" -m experiments.iclr.progressive evaluate \
  --output "$PD_ROOT" --python "$OF_PY" --vbench-python "$PD_VB_PY" \
  --session pd26_final --execute
```

导出 Original/Rectified 各944个视频，并依次计算全16维 VBench。最终仅这两个条件，无 FFE、无 Qwen、无 EMA；第一帧也只有一次 denoise，生成后的零时刻 cache 写入不属于额外去噪。生成 seed 按原始 prompt 全局序号0..943，两组一致；这是本轮固定协议，不声称逐视频复现旧 shard 的随机流。

最终视频在 `final/<arm>/videos`，rank 导出记录与 checkpoint hash 保存在 `final/`。评测完成可重验：

```bash
"$OF_PY" -m experiments.iclr.progressive report --output "$PD_ROOT"
```

留出预览共128视频，正式评测1888视频，总计2016视频。只对最终点跑 VBench；预览不是额外的 checkpoint 扫分。

## 5. 保留内容与执行边界

保留 source 清单、manifest、代码/配置/数据/权重 hash、所有4个阶段的 step300 raw、完整日志、10步观察笔记、每阶段所有预览 MP4/contact sheet/review、正式1888 MP4、16维原始 JSON 和 `report.json`。失败也保留，不覆盖目录、不自动 weights-only resume、不清理其它任务。自动 launcher 仅给自己子进程设置 `CAUSAL_DISABLE_FLEX_ATTENTION=1`，使用仓库已有 segmented attention fallback；不改 tmux 全局环境。两组同样设置，实际执行环境写入记录。

CPU 验证覆盖合并两步 teacher 的数学等价、teacher 无梯度/无采样 RNG、连续阶段微型优化、所有帧相同 NFE、真实生成帧 cache、来源拒绝规则、数据留出、raw lineage、视频检查门槛、MP4 编解码和原始944 prompts。CPU 合成视频只验证文件管线，**没有验证真实 Wan 输出质量或8卡 FSDP运行**；这些须在训练机执行上面的检查。

本轮基准代码新增了独立入口。GPU冒烟、来源或质量检查发现问题时，先记录错误与实际模块路径，在隔离工作目录自主修复并重做对应验证；必要的设置变更按第0节建立独立配对版本，不影响其它任务。训练机交付实际执行的代码提交、全部变更理由与成本、原始分数、配对差值和视频检查结论。任务的下一步由AI依据证据推进，不要求用户逐次回答恢复/诊断/修复/重试是否继续。
