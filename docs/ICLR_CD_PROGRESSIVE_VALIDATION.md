# 渐进蒸馏：本机 CPU 验证记录

日期：2026-11-11。任务与执行顺序见 [ICLR_CD_PROGRESSIVE_TASK.md](ICLR_CD_PROGRESSIVE_TASK.md)。

- 环境：Python 3.10，PyTorch 2.5.1+cu118，`CUDA_VISIBLE_DEVICES=''`，`OMP_NUM_THREADS=2`。
- 新增与历史 CD 相关测试共 **45 passed**，11.19秒；测试文件为 `test_iclr_progressive.py`、`test_iclr_cd_original.py`、`test_curvature_cd.py`。
- 旧两个测试的普通 CPU 收集触发了既有 `wan/modules/t5.py` 默认参数中的 `torch.cuda.current_device()`。回归重跑仅在测试解释器中将该查询替换为 `lambda: 0`；不初始化 CUDA，不修改模型或官方代码，不据此声称 GPU 验证通过。新增入口本身使用延迟模型导入，可直接做 CPU 验证。
- Python 编译检查、CLI help 和差异空白检查通过。
- CPU 微型模型执行完整的两个300步阶段，验证梯度、冻结 teacher 和继承关系；不训练 Wan。
- 合成 MP4 完整编码/解码81帧、16fps、832×480，检查生成时序 contact sheet。该视频仅是文件管线测试，不是实际模型生成质量证据。
- 原始 CPU 日志及合成测试产物保留在 `/data/fengjiaqi/One-Forcing/iclr_progressive_cpu_g24q_zfn/`，未上传大文件。

回归复核命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$OF_PY" - <<'PY'
import torch, pytest
torch.cuda.current_device = lambda: 0  # 仅旧 Wan 模块的导入默认参数
raise SystemExit(pytest.main(['-q', 'tests/test_iclr_progressive.py',
    'tests/test_iclr_cd_original.py', 'tests/test_curvature_cd.py']))
PY
```

真实四步起点、8卡 FSDP、B200 内核行为及实际 Wan MP4 尚待训练机核验。本机没有启动 GPU 训练，也没有中断其它会话。历史四步原权重未在本机找到；不能用已有1000步权重填补这一来源缺口。
