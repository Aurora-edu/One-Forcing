# Qwen-aligned VBench runner

For the paired GAN ablation using **the official GitHub `main` training code**
for both arms, use
[`OFFICIAL_MAIN_CODE_GAN_ABLATION.md`](OFFICIAL_MAIN_CODE_GAN_ABLATION.md).
The older [`MAIN_ALIGNED_GAN_ABLATION.md`](MAIN_ALIGNED_GAN_ABLATION.md)
matches main's YAML hyperparameters but runs the rebuttal-branch trainer;
it is not the official-code experiment.
The later FFE-trained variant is documented separately in
[`PAIRED_FFE_GAN_ABLATION.md`](PAIRED_FFE_GAN_ABLATION.md). The historical
fixed-rollout full200/DMD-only pair below answers a different training
configuration and is not the FFE-training-recipe ablation.

For the priority-ordered ICLR 2027 reruns, use
[`ICLR2027_ALIGNED_EXPERIMENTS.md`](ICLR2027_ALIGNED_EXPERIMENTS.md) and
`run_iclr2027_priority.py`. The examples below show the generic single-cell
runner; they are **not** an instruction to regenerate already aligned cells.

The original 48-hour rebuttal table mixed protocols: most short-video scores
used original VBench prompts as generator conditioning, while the paper says
its VBench prompts were rewritten with Qwen/Qwen2.5-7B-Instruct. It also mixed
newly trained step-600 and step-200 checkpoints, raw and EMA weights, and
one- and five-sample scoring. Those scores remain valid for their recorded
conditions but cannot be read as replications of the manuscript's main score.

Use `run_paper_aligned_vbench.sh` for new short-video VBench experiments. It
pins the 944 prompt order, exact Qwen rewrites, original-prompt filenames,
all 16 dimensions, 21 latent/81 RGB frames, 16 fps, no attention sink, and
raw One-Forcing generator weights; the Self-Forcing option explicitly loads
its released EMA-only weights with native three-frame chunkwise, four-step
generation. The `ffe` schedule uses four denoising
updates in the first block and one in each subsequent frame. It writes an
audited manifest with deterministic seeds; all conditions reuse the same
manifest bytes. Five samples per prompt is the official VBench protocol.
One-sample runs are labeled follow-ups and cannot be directly compared with a
five-sample manuscript number.

| Question | Matched conditions | Fixed setting |
|---|---|---|
| Manuscript score | released `one_forcing.pt`, `ffe` | already Qwen-conditioned in the paper; rerun only for a separate reproducibility question |
| FFE contribution | same checkpoint, `ffe` vs `all1` | already Qwen-conditioned in the paper |
| GAN contribution | retrained full vs DMD-only at step 200, both `ffe` | identical training budget and manifest |
| Training trajectory | retrained full step 200/400/600, all `ffe` | identical run and manifest |
| Four-step baseline | retrained One-Forcing framewise vs released Self-Forcing native chunkwise, both `all4` | Qwen and generation seeds paired; architecture and released weight type disclosed |

The pinned rewrite file is byte-identical to the historical Self-Forcing
comparison and the locally stored 20-second/long-video Qwen inputs. The exact
rewrite file and noise stream for the manuscript's headline run are not
archived here. This code therefore tests checkpoints under a shared,
reproducible paper-style protocol; it does not force reproduction of a
particular numeric total. The local ICLR 2027 paper reports 83.76 for its
headline checkpoint. Different manuscript versions and checkpoints should
not be conflated.

## Prepare and verify inputs without using GPUs

```bash
bash experiments/rebuttal/run_paper_aligned_vbench.sh \
  --name published_ffe \
  --output_root eval/paper_aligned/published_ffe \
  --prepare_only
python -m unittest -v tests/test_paper_vbench_protocol.py
```

The pinned assets live in `assets/qwen_vbench/`. Preparation fails if their
hashes or the 944 original/rewrite pairing differ. A separate output root is
required for each condition; old video directories with a different intent
are intentionally rejected.

## Run matched conditions

```bash
export VBENCH_PYTHON=/path/to/vbench/bin/python
export INFER_PYTHON=/path/to/one_forcing/bin/python
export GPU_IDS=all

bash experiments/rebuttal/run_paper_aligned_vbench.sh \
  --name published_ffe \
  --checkpoint_path checkpoints/one_forcing.pt \
  --schedule ffe \
  --output_root eval/paper_aligned/published_ffe \
  --gpus "$GPU_IDS" --python "$INFER_PYTHON" \
  --vbench_python "$VBENCH_PYTHON"

bash experiments/rebuttal/run_paper_aligned_vbench.sh \
  --name full_step200_ffe \
  --checkpoint_path /path/to/full/checkpoint_model_000200/model.pt \
  --schedule ffe \
  --output_root eval/paper_aligned/full_step200_ffe \
  --gpus "$GPU_IDS" --python "$INFER_PYTHON" \
  --vbench_python "$VBENCH_PYTHON"

bash experiments/rebuttal/run_paper_aligned_vbench.sh \
  --name dmd_step200_ffe \
  --checkpoint_path /path/to/dmd/checkpoint_model_000200/model.pt \
  --schedule ffe \
  --output_root eval/paper_aligned/dmd_step200_ffe \
  --gpus "$GPU_IDS" --python "$INFER_PYTHON" \
  --vbench_python "$VBENCH_PYTHON"

bash experiments/rebuttal/run_paper_aligned_vbench.sh \
  --name self_forcing_all4 \
  --model self_forcing \
  --checkpoint_path /path/to/self_forcing_dmd.pt \
  --schedule all4 \
  --output_root eval/paper_aligned/self_forcing_all4 \
  --gpus "$GPU_IDS" --python "$INFER_PYTHON" \
  --vbench_python "$VBENCH_PYTHON"
```

For FFE, run the **same** full checkpoint with `--schedule all1` in a fresh
output root. For checkpoint stability, run step 200/400/600 with `ffe`. For
four-step generation, use the separately trained four-step checkpoint with
`--schedule all4`. These conditions share the pinned Qwen conditioning and
seed rule, while their checkpoint or schedule differences remain explicit.
The historical Self-Forcing comparison uses its own two-stream RNG runner;
do not merge that 944×1 score with this 944×5 table. Use the new command
above for a five-sample baseline under the shared manifest. This baseline uses
the repository's `self_forcing_config.yaml` with native three-frame blocks.

Each completed condition writes `<name>_paper_protocol_summary.json` with
the 16 dimension scores, official normalized total/quality/semantic, source
checkpoint, schedule, prompt/rewrite/manifest hashes, and sample count.
Existing `RESULTS_SUMMARY.md` and `RUN_REPORT.md` retain the earlier numbers
and are marked as historical protocol results until these reruns finish.

After the paired runs finish, generate the replacement table with:

```bash
python experiments/rebuttal/summarize_paper_aligned_vbench.py \
  --run full=eval/paper_aligned/full_step200_ffe/full_step200_ffe_paper_protocol_summary.json \
  --run dmd=eval/paper_aligned/dmd_step200_ffe/dmd_step200_ffe_paper_protocol_summary.json \
  --comparison gan_gain=full,dmd \
  --output_json eval/paper_aligned/paired_results.json \
  --output_markdown eval/paper_aligned/paired_results.md
```

The summarizer rejects a comparison if prompt, rewrite, seed manifest, sample
count, or 16-dimension scores differ.

The 20-second long-video results already use the pinned Qwen rewrite set, but
their longer rollout and one- or twenty-seed scoring are separate protocols.
LPIPS/FVD evaluations use their own prompt and real-video selections; their
scores should be interpreted as paired ablations, not as replicas of the
manuscript VBench total.
