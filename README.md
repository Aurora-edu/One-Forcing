<p align="center">
<h1 align="center">One-Forcing</h1>
<h3 align="center">Towards Stable One-Step Autoregressive Video Generation</h3>
</p>
<p align="center">
  <p align="center">
    Jiaqi Feng<sup>1,*</sup> · Justin Cui<sup>2,*</sup> · Yuanhao Ban<sup>2</sup> · Cho-Jui Hsieh<sup>2</sup><br>
    <sup>1</sup>Tsinghua University <sup>2</sup>UCLA <sup>*</sup>Equal contribution
  </p>
  <h3 align="center"><a href="TODO">Paper</a> | <a href="https://aurora-edu.github.io/one-forcing/">Website</a> | <a href="https://huggingface.co/JiaqiFeng/OneForcing">Models/Data</a></h3>
</p>

---

One-Forcing is a one-step causal video distillation method built on Causal
Forcing and Wan2.1. It keeps the causal autoregressive generator and DMD
score-matching objective, and adds an adversarial noised-latent branch by
reusing the trainable fake-score network as both diffusion critic and
discriminator. The reported framewise model reaches 83.58 VBench total score in
the paper.

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
  --output_folder outputs \
  --use_ema
```

## Dataset Preparation

Following [CausVID](https://github.com/tianweiy/CausVid), we use the [MixKit Dataset](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/all_mixkit) (6K videos) as a toy example for distillation.

To prepare the dataset, follow these steps. You can also download the final LMDB dataset from [here](https://huggingface.co/tianweiy/CausVid/tree/main/mixkit_latents_lmdb)

```bash
# download and extract video from the Mixkit dataset 
python distillation_data/download_mixkit.py  --local_dir XXX 

# convert the video to 480x832x81 
python distillation_data/process_mixkit.py --input_dir XXX  --output_dir XXX --width 832   --height 480  --fps 16 

# precompute the vae latent 
torchrun --nproc_per_node 8 distillation_data/compute_vae_latent.py --input_video_folder XXX  --output_latent_folder XXX   --info_path sample_dataset/video_mixkit_6484_caption.json

# combined everything into a lmdb dataset 
python causvid/ode_data/create_lmdb_iterative.py   --data_path XXX  --lmdb_path XXX
```

## Training
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

### One Forcing Training
```bash
torchrun --nproc_per_node=8 train.py \
  --config_path config.yaml \
  --generator_ckpt checkpoints/framewise/causal_ode.pt \
  --teacher_model_path wan_models/Wan2.1-T2V-14B \
  --data_path mixkit_latents_lmdb \
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
  --name one_forcing_framewise
```

## Citation

```bibtex
@article{oneforcing2026,
  title={One-Forcing: Towards Stable One-Step Autoregressive Video Generation},
  author={TODO},
  journal={TODO},
  year={2026}
}
```

## Acknowledgements

This codebase builds on
[Causal Forcing](https://github.com/thu-ml/Causal-Forcing),
[Self Forcing](https://github.com/guandeh17/Self-Forcing),
[CausVid](https://github.com/tianweiy/CausVid), and the [Wan](https://github.com/Wan-Video/Wan2.1) model family.
