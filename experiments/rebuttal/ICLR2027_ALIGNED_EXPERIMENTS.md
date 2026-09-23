# ICLR 2027: priority-ordered Qwen-aligned follow-up

Source audited: `/home/jiaqi/paper/iclr2027.tex` on 2026-09-23. This is a
**rerun plan and executable code**, not a replacement result table. The paper's
current rebuttal-derived numbers must remain labeled with their actual old
protocol until the new runs finish. Do not adjust a score to make it match
83.76; the released headline checkpoint, random stream, and retrained
step-200 checkpoint are different experiments.

## What to run, in order

All scheduled VBench cells use the same pinned 944 original VBench prompts for
scoring/filenames, the same Qwen/Qwen2.5-7B-Instruct rewrites for model
conditioning, five generation seeds per prompt, 16 dimensions, official
normalized total/quality/semantic scoring, 21 latent/81 RGB frames, 16 fps,
zero attention sink, and the checkpoint's generator weights (not EMA). FFE is
four updates in the first block and one thereafter. `all4` is four updates
in every block. The schedule reuses **existing trained checkpoints**; it does
not train new models or evaluate a four-step model by switching a one-step
checkpoint's sampler.

| Priority | Paper location/question | Required cells | Existing aligned result? |
|---|---|---|---|
| Main 1 | Table `gan-ffe-ablation` (a), GAN contribution | full step 200 / DMD-only step 200, both FFE | No: old 75.27/80.36 used original prompts |
| Main 2 | Table `trajectory-rectification`, 4-step consistency distillation | curved / rectified step 300, both all4 | No: old 61.16/71.11 used original prompts |
| Appendix 1 | Table `extended-training` | full step 400 / 600, FFE; reuse Main 1 step 200 | No: old cells used original prompts |
| Appendix 2 | Table `ablation-extra-dimensions` | Reuse all 16-dimension scores above, plus existing FFE pair | No extra generation |

The paper's **83.76 headline** and **FFE pair (80.69/83.30)** were already
Qwen-conditioned and should not be generated again solely for prompt
alignment. The Qwen-matched 4-step comparison (83.85/83.46) is already done
on the historical one-sample protocol; do not rerun it as part of this
five-sample campaign. Its sample count must be disclosed separately.
LPIPS/recall use separate paired diversity/real-video selections rather than
the 944-prompt VBench table; latency and training curves are also separate
protocols. They do not become a VBench replication by replacing their prompt
file. The 20-second rollout already used the pinned Qwen rewrites, so video
generation is **not** scheduled again here.

There is a **paper-data conflict to resolve before updating the long-video
table**: `iclr2027.tex` prints Self-Forcing 75.30 and One-Forcing 78.52;
the committed 20-seed, no-sink Qwen report records Self-Forcing 69.08 and
One-Forcing 78.52, and says the prior Self-Forcing generation path behind
another score was not committed. The separate single-seed sink comparison
records yet another pair (79.86/80.14) under different attention settings.
These are not interchangeable. Locate and audit the raw export/config/result
behind 75.30, or correct the paper table using a single defensible paired
protocol. This is a provenance check, **not** authorization to synthesize or
substitute a favorable number.

## Training-machine execution

First inspect the plan without a GPU:

```bash
python experiments/rebuttal/run_iclr2027_priority.py --phase plan
python -m unittest -v tests/test_paper_vbench_protocol.py tests/test_iclr2027_priority.py
```

Find the **actual** pre-existing checkpoints. `full200`, `full400`, and
`full600` must come from the *same* 600-step DMD+GAN training trajectory;
`dmd200` is the paired DMD-only run; `curved300` and `rectified300` are the
paired 300-step consistency-distillation arms. Do not substitute the released
`checkpoints/one_forcing.pt` for any of them. Use absolute paths below, and
ensure the VBench Python environment has VBench dependencies installed.
`--gpus all` resolves the host's actual idle GPU inventory; it does not assume
eight A100s. Select explicit idle IDs if other sessions use some GPUs.

```bash
python experiments/rebuttal/run_iclr2027_priority.py \
  --phase main \
  --checkpoint full200=/ABS/PATH/full/checkpoint_model_000200/model.pt \
  --checkpoint dmd200=/ABS/PATH/dmd/checkpoint_model_000200/model.pt \
  --checkpoint curved300=/ABS/PATH/curved/checkpoint_model_000300/model.pt \
  --checkpoint rectified300=/ABS/PATH/rectified/checkpoint_model_000300/model.pt \
  --output_root /LOCAL/DISK/iclr2027_qwen \
  --gpus all --python /ABS/PATH/to/inference/python \
  --vbench_python /ABS/PATH/to/vbench/python
```

After the main comparison is audited and the generated clips have been
visually inspected, run the appendix phase with the **same output root**:

```bash
python experiments/rebuttal/run_iclr2027_priority.py \
  --phase appendix \
  --checkpoint full400=/ABS/PATH/full/checkpoint_model_000400/model.pt \
  --checkpoint full600=/ABS/PATH/full/checkpoint_model_000600/model.pt \
  --output_root /LOCAL/DISK/iclr2027_qwen \
  --gpus all --python /ABS/PATH/to/inference/python \
  --vbench_python /ABS/PATH/to/vbench/python
```

Completed cells are skipped only after re-auditing their prompt/rewrite and
seed manifest hashes, checkpoint path/size, generation settings, video count,
and all 16 VBench dimensions. A mismatched or incomplete existing directory
causes a hard failure; use a *new explicit output root* after investigation,
never overwrite or delete old results. The comparison JSON/Markdown is
written under the output root; videos, checkpoints and per-video data stay on
the local training machine. Publish only audited metric summaries once
approved. If a checkpoint is missing, stop and report it rather than
training a replacement under a different configuration.

Before relying on a cell, watch the run through completion and inspect
representative generated MP4s from **both sides** of each pair, including
the beginning, middle and end, for obvious collapse or motion failure.
Record the selected filenames and observations alongside the numeric summary.
