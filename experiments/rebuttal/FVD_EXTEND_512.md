# 将现有 256-sample FVD 增量扩展到 512

本流程保留现有 256 个 real、主方法 fake 和 DMD-only fake，只从 LMDB 未使用的样本中再取
256 个。新增文件编号为 `0256`–`0511`，最终在评分阶段合并两个目录，不复制原视频。

代码固定沿用现有协议：step-200 主方法和 DMD-only checkpoint、FFE 推理、非 EMA 权重、
21 latent frames、16-frame I3D、`nearest_k=5`。两个模型使用完全相同的新增 prompt、seed
和 real set。

## 在实验机运行

拉取 `review` 分支后，先填写实验机的实际路径：

```bash
git checkout review
git pull --ff-only origin review

export CLEAN_LMDB=/path/to/clean_latent_lmdb
export MAIN200=/path/to/main/checkpoint_model_000200/model.pt
export DMD200=/path/to/dmd_only/checkpoint_model_000200/model.pt
export I3D=/path/to/i3d_torchscript.pt
export OLD_REAL=/path/to/eval/fvd/real_seed0
export OLD_MAIN=/path/to/eval/fvd/main_step200
export OLD_DMD=/path/to/eval/fvd/dmd_step200
export FVD512_OUT=/path/to/local/eval/fvd_512
export FVD_GPUS=0,1,2,3,4,5,6,7
```

机器有几张空闲卡就把 `FVD_GPUS` 写成几张，不要求一定是 8 卡。用 tmux 启动：

```bash
tmux new-session -s fvd512
mkdir -p "$FVD512_OUT"
bash experiments/rebuttal/run_fvd_extend_256_to_512.sh \
  --lmdb_path "$CLEAN_LMDB" \
  --existing_real_dir "$OLD_REAL" \
  --existing_main_fake_dir "$OLD_MAIN" \
  --existing_dmd_fake_dir "$OLD_DMD" \
  --main_checkpoint "$MAIN200" \
  --dmd_checkpoint "$DMD200" \
  --i3d_path "$I3D" \
  --output_root "$FVD512_OUT" \
  --gpus "$FVD_GPUS" \
  2>&1 | tee "$FVD512_OUT/run.log"
```

按 `Ctrl-b`、再按 `d` 可退出 tmux；用 `tmux attach -t fvd512` 返回。脚本可在中断后用同一条
命令继续，已完成且通过校验的视频不会重新生成。不要改变 GPU 列表或输入路径后复用同一个
`FVD512_OUT`。

完成后，本地保留所有视频，只需提交这两个指标文件：

```text
$FVD512_OUT/main_512.json
$FVD512_OUT/dmd_512.json
```

脚本会在启动时严格检查三个旧目录各有 256 个视频，并在评分前检查合并后的 real/fake 文件名
与 512 条 manifest 完全对应；不满足时直接报错，不会静默替换样本。
