# CD 1000步权重缺失时的重训流程

用户已授权此分支：先找已有 Original / Global Rectified 的 step1000 raw 权重，找不到就重训。完成后接 [FFE评测](ICLR_CURVATURE_1000_FFE_TASK.md)。这是 [本轮总任务](ICLR_CD_FFE_AND_OFFICIAL_S42_TASK.md) 的条件步骤，不增加新方案或参数搜索。

## 复用判断与固定配方

两份合格权重都存在则不重训。仅缺一组时，只有在初始化内容、轨迹数据、1000步配置及运行环境都能与保留组核对一致的情况下补训该组；无法证明时保留旧产物，另训完整一对。即使 seed 相同，新训练也不是旧运行本身；复用/重训状态须按组记录。

训练入口固定为 `experiments/rebuttal/train_curvature_ode.py`，原配置为 `experiments/rebuttal/configs/train_curvature_ode.yaml`。本轮不修改这两份文件，尤其不替换成 `cd`、`cd_original`、`cd_local` 或 `train_curvature_cd.py`。

| 项目 | 两组共同设置 |
|---|---|
| Seed / updates / world size | 2026 / 1000 / 8 |
| batch / latent frames / block | 每卡1 / 21 / 1 |
| 优化器 | AdamW，lr=2e-6，betas=(0.9,0.999)，weight decay=0.01，clip=10 |
| 输入采样 | 原代码从四个噪声时刻随机取一个；不重新加权 |
| 目标与历史 | 直接回归固定 teacher ODE 端点；clean latent teacher forcing |
| 干预 | curved 保留轨迹；rectified 按原 loader 做全局拉直 |
| 保存与评测 | 只保存最终1000步 raw；original944×1，main_ffe，全16维 |

没有相邻CD/target EMA损失、局部渐进拉直或额外端点权重。FFE 是训练后的推理设置，不写入训练配置。21帧由实际轨迹张量决定，不能只看 YAML 后假定训练用了21帧。

## 初始化和数据

先查旧 `curvature_control` 目录、迁移资产和旧 run.intent/training.done。成品权重丢失不要求重建轨迹；优先复用已经生成的共享 LMDB。

如果历史初始化/数据未找回，沿用此前用户已经指定并在训练机使用的 Causal-Forcing `framewise/ar_diffusion.pt` raw 和与其同源的 `cd_ar_data/data/shared_curvature_lmdb`。AR 权重 SHA256：`721ea4d8f6d23099ccbefc6f5741b170402f509738f5a7b2db10e91b76857825`。这时两组一起重训，报告为“原1000步配方的新配对重跑，历史输入未核实”，不能把它与旧四步分数当成同权重对照。不要误用 frame/chunk 的 `causal_ode.pt`。

沿用训练机 env.sh，设置真实路径：

```bash
export CD1000_INIT="$CD_INIT"
export CD1000_DATA="$CD_DATA"
export CD_REBUILD_ROOT="$R/curvature_1000_retrain_s2026"
export CD1000_GPUS=0,1,2,3,4,5,6,7
```

`CD1000_INIT` 必须是生成 `CD1000_DATA` 的同一份 raw 源模型，不能从变量名推断来源。启动前保存以下证据到新的重训目录：

- 模型完整 SHA256；LMDB 的 `curvature_dataset_manifest.json` 原文件及 hash、数据来源和校验记录。
- manifest 的 `generator_checkpoint_sha256` 与实际初始化相同；`use_ema=false`、`guidance_scale=6`。
- `latents_shape` 为 `[N,6,21,16,60,104]`，使用完整数据（既有数据通常N=6505）；`selected_timesteps` 为 `[1000,937.5,833.3333,625,0,null]`（允许浮点舍入）。
- 数据是原始 curved 轨迹，rectified 在 loader 中计算；噪声、ODE端点、独立clean history和prompt相同。可用 `analyze_curvature_intervention.py` 检查首64行。

旧 manifest 缺源模型hash时，从原轨迹文件/已有校验记录恢复证据，不能随便补写。如果不能证明现有数据同源，用已选 raw 初始化重新准备轨迹。优先复用原轨迹 `.pt` 重新 build；没有轨迹才在空闲8卡、独立tmux下执行下面的 **prepare 阶段**，不执行 `all` 或 `evaluate`：

```bash
bash experiments/rebuttal/run_curvature_causal_experiment.sh prepare \
  --ar_checkpoint "$CD1000_INIT" --clean_lmdb "$CLEAN_LMDB" \
  --output_root "$CD_REBUILD_ROOT" --gpus "$CD1000_GPUS" --python "$OF_PY"
export CD1000_DATA="$CD_REBUILD_ROOT/data/shared_curvature_lmdb"
```

这是一次共享数据准备，两组复用；不因某一组结果不好而另采一套数据。prepare 输出目录必须是新目录，保留已有数据。

原训练入口还会比较 manifest 的 `generator_ckpt` **解析后路径**。若迁移只改变路径，先验证内容hash，再优先使用安全的旧路径符号链接映射到实际初始化（不能覆盖已有文件）。无法建立映射时，可在新实验目录创建只读数据视图：只链接原 `data.mdb`，复制 manifest 并仅把 `generator_ckpt` 改为已核实的当前路径；另存原 manifest、前后hash和映射说明。不修改共享源 manifest，不改变源权重hash、数据内容、行顺序或其它字段。两组使用同一视图，报告中记录这次迁移。

## 训练命令与监控

在独立 tmux 内执行，每次启动前检查所选8卡没有其它会话的计算进程。下面先运行 curved；完成后将 `CD1000_ARM=rectified`，以相同命令启动第二组。只补训一组时选对应 arm，另一组保留并记录来源。

```bash
export CD1000_ARM=curved
"$OF_PY" -c 'import os; from experiments.rebuttal.run_sharded_inference import parse_gpus, require_idle_gpus; require_idle_gpus(parse_gpus(os.environ["CD1000_GPUS"]))'

CUDA_VISIBLE_DEVICES="$CD1000_GPUS" "$OF_PY" -m torch.distributed.run \
  --standalone --nproc_per_node=8 experiments/rebuttal/train_curvature_ode.py \
  --config_path experiments/rebuttal/configs/train_curvature_ode.yaml \
  --data_path "$CD1000_DATA" --generator_ckpt "$CD1000_INIT" \
  --output_dir "$CD_REBUILD_ROOT/training/$CD1000_ARM" \
  --intervention "$CD1000_ARM"
```

保存完整stdout/stderr。**每组至少盯到第10步**：检查实际 seed=2026、world size=8、输入21帧、loss/grad norm有限、GPU运行正常，保存观察记录。正式配方不增加10步预训练或smoke权重续训，直接观察正式运行的前10步。中断后原checkpoint没有完整optimizer/RNG状态，使用新空目录从相同初始化重跑，不拼接旧日志。

每组完成须有 `training.done`、完整1–1000步的 `metrics.jsonl` 和 `checkpoint_model_001000/model.pt`。审计 `experiment_metadata` 中的 arm/seed/1000步、raw generator，记录权重hash；两组配置、初始化hash和数据manifest须匹配。保留B200环境及包装脚本，环境修复不得变成科学配置修改。

## 准备评测目录并接 FFE

两组新训时令 `CURVATURE_ROOT="$CD_REBUILD_ROOT"`；只补一组时，在新的配对评测根目录按相同结构链接已验证的另一组最终checkpoint及来源记录，保留其原路径和hash，不搬走旧文件。

优先按字节复制历史 `manifests/vbench_prompts.txt` 和 `manifests/vbench_single_sample_seed0.jsonl`。两者也缺失时，在新的评测目录按仓库固定 VBench metadata 重建，原始prompts944条、sample=1、base seed=0：

```bash
export CURVATURE_ROOT="$CD_REBUILD_ROOT"
"$OF_PY" experiments/rebuttal/prepare_vbench_prompts.py \
  --full_info_path "$INFO" --output_path "$CURVATURE_ROOT/manifests/vbench_prompts.txt"
"$OF_PY" experiments/rebuttal/make_eval_manifest.py \
  --prompt_path "$CURVATURE_ROOT/manifests/vbench_prompts.txt" \
  --output_path "$CURVATURE_ROOT/manifests/vbench_single_sample_seed0.jsonl" \
  --base_seed 0 --num_samples_per_prompt 1 --naming vbench
```

重建前核对 `INFO` 与 `experiments/rebuttal/assets/qwen_vbench/VBench_full_info.json` 的hash相同；该JSON提供原始scoring prompts，不是Qwen改写文本。已有manifest不覆盖。

随后执行总任务中的 `evaluate_curvature_1000_ffe --dry-run` 和 `--execute`，只跑两组 main_ffe。完成真实视频检查、全16维汇总和交付；不自动加跑 all1/all4。新旧权重身份及训练来源分开记录，不以历史分数为新训练的验收阈值。
