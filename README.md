<p align="center">
<h1 align="center">One-Forcing</h1>
<h3 align="center">Towards Stable One-Step Autoregressive Video Generation</h3>
</p>
<p align="center">
  <p align="center">
    Jiaqi Feng<sup>1,*</sup> · Justin Cui<sup>2,*</sup> · Yuanhao Ban<sup>2</sup> · Cho-Jui Hsieh<sup>2</sup><br>
    <sup>1</sup>Tsinghua University <sup>2</sup>UCLA <sup>*</sup>Equal contribution
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2605.23458">Paper</a> | <a href="https://aurora-edu.github.io/one-forcing/">Website</a> | <a href="https://huggingface.co/JiaqiFeng/OneForcing">Models/Data</a></h3>
</p>

---

  One-Forcing enables stable **1-step autoregressive video generation** by augmenting DMD-based causal distillation with a
  shared noised-latent adversarial critic, achieving **state-of-the-art** 1-step VBench performance and efficient framewise
  generation.

---

## Installation

```bash
conda create -n one_forcing python=3.10 -y
conda activate one_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Inference
Download the trained One-Forcing checkpoint:

```bash
hf download JiaqiFeng/OneForcing checkpoints/one_forcing.pt --local-dir .
```

```bash
bash scripts/infer.sh \
  --checkpoint_path checkpoints/one_forcing.pt \
  --prompt_path prompts/demos.txt \
  --output_folder outputs
```

## Training
### Dataset Preparation

```bash
hf download JiaqiFeng/OneForcing --include "clean_data/*" --local-dir .
```

### Download ODE initialized checkpoint
```bash
hf download JiaqiFeng/OneForcing checkpoints/framewise/causal_ode.pt --local-dir .
```
You can refer to [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) Stage1/2 to train your ODE initialized checkpoint

### Download Wan2.1 Base Models
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B \
  --local-dir wan_models/Wan2.1-T2V-14B
```

### One Forcing Training(200 steps recommended to converge)
```bash
torchrun --nproc_per_node=8 train.py \
  --config_path config.yaml \
  --generator_ckpt checkpoints/framewise/causal_ode.pt \
  --teacher_model_path wan_models/Wan2.1-T2V-14B \
  --data_path clean_data \
  --logdir runs \
  --disable-wandb \
  --no_visualize
```

## Evaluation

Export videos first, then run VBench with your local VBench installation:

```bash
python scripts/run_vbench.py \
  --videos_path outputs \
  --full_info_path VBench_full_info.json \
  --output_dir eval/vbench \
  --name one_forcing
```

## Citation

```bibtex
@article{feng2026oneforcing,
  title={One-Forcing: Towards Stable One-Step Autoregressive Video Generation},
  author={Feng, Jiaqi and Cui, Justin and Ban, Yuanhao and Hsieh, Cho-Jui},
  journal={arXiv preprint arXiv:2605.23458},
  year={2026},
  eprint={2605.23458},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.23458}
}
```

## Acknowledgements

This codebase builds on
[Causal Forcing](https://github.com/thu-ml/Causal-Forcing),
[Self Forcing](https://github.com/guandeh17/Self-Forcing),
[CausVid](https://github.com/tianweiy/CausVid), and the [Wan](https://github.com/Wan-Video/Wan2.1) model family.
