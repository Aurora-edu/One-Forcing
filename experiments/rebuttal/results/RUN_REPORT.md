# NeurIPS 2026 Rebuttal — Official Run Report

Generated programmatically from aggregated result JSONs only. Full numeric
tables: `final_metrics.json` / `final_metrics.csv` in this directory. No raw
videos, per-prompt/per-sample records, checkpoints, logs, or caches are
uploaded.

## 1. Hardware and timeline

- 48-hour clock started at the first environment/hardware precheck on the
  target machine: **2026-07-26T02:57:09Z** (deadline 2026-07-28T02:57:09Z).
  All work reported here finished ~2026-07-27T20:00Z, inside the window.
- Training node: `a3u50asia-a3u-44` (Slurm, exclusive), **8× NVIDIA H200
  141 GB**, driver 570.172.08, torch 2.5.1+cu124, flash-attn 2.7.4.post1.
  The README's 8×A100 speed assumptions were replaced by speeds measured on
  this machine (10-step prechecks of all three full configs + latency
  profiler) before any official run.
- All three training runs were launched via `launch_train.sh` inside tmux on
  the same 8 GPUs (IDs 0–7, verified idle), sequentially, and each was
  monitored past step 10 with finite losses before being left to finish.
- Evaluation was fanned out to identical 8×H200 nodes (one condition per
  node, per operator instruction to use more nodes); every condition still
  uses the repo's 8-way sharded inference and 8-rank VBench with the fixed
  manifests below. Eval nodes: a3u-12 (FFE + stability), a3u-21 (all1 +
  latency), a3u-11 (GAN7 pair), a3u-18 (OF-4step), a3u-36/a3u-2 (SF4),
  a3u-0 (diversity + FVD).

## 2. Training runs (all seed 0, single seed — disclosed)

| Run | Steps | s/step (measured cadence) | Wall clock | check_run |
|---|---:|---:|---:|---|
| Full One-Forcing (`train_1step_one_forcing`) | 600 | 23.98 | 4.09 h | PASS |
| DMD-only (`train_1step_dmd_only`) | 200 | 13.59 | 0.93 h | PASS |
| 4-step One-Forcing (`train_4step_one_forcing`) | 300 | 24.71 | 2.31 h | PASS |

- DMD-only run: `gan_d_loss`, `gan_g_loss`, `r1_loss`, `r2_loss` strictly 0
  at every logged step.
- Checkpoints every 100 steps (6 / 2 / 3), each ~11.6 GB, all present.
- No resume was used in any official run.

## 3. Official VBench (944 prompts × 5 samples = 4,720 videos per condition)

Official leaderboard aggregation (normalized, quality:semantic = 4:1):

| Condition | Total | Quality | Semantic |
|---|---:|---:|---:|
| Full One-Forcing step 600, FFE | **0.8165** | 0.8501 | 0.6820 |
| Full One-Forcing step 600, all1 (FFE off) | 0.7552 | 0.7925 | 0.6064 |
| 4-step One-Forcing step 300, all4 | **0.7981** | 0.8244 | 0.6927 |
| 4-step Self-Forcing (released `self_forcing_dmd.pt`), all4 | 0.7375 | 0.7638 | 0.6323 |

Method deltas (same manifest, frame count, generation seeds):

- **FFE contribution** (same step-600 checkpoint): +0.0613 total
  (0.8165 vs 0.7552).
- **4-step generalization**: One-Forcing +0.0606 total over Self-Forcing
  (0.7981 vs 0.7375). SF4 checkpoint sha256-verified against the official
  gdhe17/Self-Forcing release.

## 4. GAN ablation, paired at step 200 (7 dims, 326 prompts × 5 × 2)

Same manifest/seeds/schedule (FFE), both step-200 checkpoints:

| Dimension | Full (DMD+GAN) | DMD-only | Δ (full − DMD) |
|---|---:|---:|---:|
| subject_consistency | 0.5576 | 0.9386 | −0.3810 |
| background_consistency | 0.7911 | 0.9220 | −0.1309 |
| temporal_flickering | 0.9572 | 0.9730 | −0.0158 |
| motion_smoothness | 0.9609 | 0.9844 | −0.0236 |
| dynamic_degree | 1.0000 | 0.8944 | +0.1056 |
| aesthetic_quality | 0.3479 | 0.6196 | −0.2718 |
| imaging_quality | 0.5871 | 0.6861 | −0.0990 |

Reported as measured: at the shared 200-step budget the DMD-only variant is
ahead on 6/7 dimensions; the full method is ahead on dynamic_degree. The
stability sweep (below) shows the full method improves substantially from
step 200 to 600 (5-dim mean 0.853 → 0.948) — step-200 is early in its
schedule — but no cross-method comparison beyond step 200 is available in
this round because DMD-only was only trained to 200 per protocol.

## 5. Stability across training (same run, steps 100/200/400/600, 5 dims)

233 prompts × 5 samples per step; steps 200/600 reuse the GAN7/final results
(identical prompt→seed mapping by construction):

| Step | 5-dim mean |
|---:|---:|
| 100 | 0.8487 |
| 200 | 0.8533 |
| 400 | 0.9325 |
| 600 | 0.9483 |

Monotonic improvement; no post-200 collapse. This is one training run's
checkpoint trajectory (seed 0), not cross-seed variance.

## 6. Diversity and distribution coverage (step-200 pair)

- **LPIPS diversity** (100 prompts × 4 samples, lpips-vgg, 8 frames/video,
  bootstrap 95% CI):
  - Full method step 200: 0.4454 [0.4392, 0.4521]
  - DMD-only step 200: 0.6430 [0.6322, 0.6535]
- **FVD / precision / recall / density / coverage** (256 matched real/fake,
  I3D, nearest_k=5, same real manifest for both):
  - Full: FVD 2717.9; precision 0.9531, recall 0.0195, density 1.847,
    coverage 0.625
  - DMD-only: FVD 846.3; precision 0.750, recall 0.766, density 0.784,
    coverage 0.887
- **1-step Self-Forcing diversity baseline NOT RUN**: no 1-step Self-Forcing
  checkpoint exists on this machine or in the official gdhe17/Self-Forcing
  release (checkpoints: ode_init, dmd, gan, sid, sid_v2, 10s; configs:
  default/dmd/sid only). Reported as unavailable rather than substituted.

## 7. Latency (single H200, bf16, 21 latent frames, 20 trials, EMA, VAE incl.)

Mean over 20 timed trials (per-trial std in `final_metrics.json`,
`latency_h200_21latents`, together with NFE/context-update counts):

| Schedule / ckpt | first block ms | steady block ms | diffusion ms | VAE ms | total ms |
|---|---:|---:|---:|---:|---:|
| all1 (MAIN600) | 102.0 | 117.8 | 2460.7 | 1622.1 | 4133.2 |
| FFE (MAIN600) | 477.7 | 120.8 | 2533.7 | 1620.3 | 4190.4 |
| all4 (MAIN600) | 244.4 | 290.9 | 6065.3 | 1621.5 | 7723.1 |
| all4 (OF4-300) | 239.8 | 288.3 | 6009.0 | 1620.4 | 7665.8 |

GPU: NVIDIA H200 (measured on this machine; the README's A100 reference
numbers were not reused).

## 8. Deviations, fixes, and disclosures

1. **EMA export bug found and fixed before the official runs used here.**
   `EMA_FSDP.full_state_dict` filtered the gathered state dict against
   flat-param shadow keys and silently dropped every FSDP-wrapped block —
   saved `generator_ema` had 15/825 keys. Because the README mandates
   `--use_ema` for all official evaluation and the EMA shadow exists only
   in memory, an initial set of three training runs (identical protocol)
   was discarded and all three runs were retrained after the one-line fix
   (see `utils/distributed.py`). Every official checkpoint was verified to
   contain a complete 825-key `generator_ema` matching the generator keys.
   The fix was validated with a nested-FSDP numerical test before retraining.
2. **Hardware substitution**: 8×H200 141 GB instead of the README's 8×A100
   (no A100 available on this cluster). Memory and speed prechecks with the
   full configuration passed; H200 exceeds the A100 in memory and speed.
   All README speed estimates were re-measured on the target machine.
3. **Multi-node evaluation** (operator-directed): independent eval
   conditions ran concurrently on identical H200 nodes; each condition
   itself follows the repo's 8-GPU sharding with the shared fixed manifests
   and seeds. Training remained strictly serial on one node.
4. **Environment**: `opencv-python` → `opencv-python-headless` (cluster
   nodes lack libGL); all audited pins unchanged (torch 2.5.1, torchvision
   0.20.1, transformers 4.49.0, diffusers 0.31.0, accelerate 1.13.0,
   numpy 1.24.4). `detectron2` (git build, CUDA sm90) added to the VBench
   env — required by VBench GRiT dimensions but absent from
   `requirements-vbench.txt`. GRiT weights fetched from the official
   OpenGVLab/VBench_Used_Models HF mirror (the Azure origin URL errors);
   file size identical to the release listing.
5. **Assets**: `causal_ode.pt`, `clean_data` LMDB (6,505 clean-latent rows)
   and `one_forcing.pt` byte-exact from `JiaqiFeng/OneForcing`;
   Wan2.1-T2V-14B teacher and Wan2.1-T2V-1.3B from `Wan-AI`;
   `SF4 = self_forcing_dmd.pt` sha256
   `a0413986…cc8f56a3` matching the official release. `AGENTS.md` does not
   exist on any branch of the repository.
6. **Not run this round (per README §1)**: long-video / error-accumulation
   rollouts, curvature rectification causal intervention, action-conditioned
   generation, new human evaluation, GAN-only ablation, extra seeds, and
   1,200-step training. SF1 diversity baseline unavailable (see §6).
7. **Single seed**: every training run uses seed 0; VBench's 5
   samples/prompt and diversity's 4 samples/prompt are generation seeds.
   The stability curve is checkpoint-trajectory variance of one run, not
   cross-seed variance.

## 9. Post-deadline addendum: no-EMA ablation (operator-requested)

After the official EMA results above were finalized and pushed (inside the
48h window), the full evaluation suite was rerun with **raw generator
weights** (`--use_ema` removed; everything else identical — same manifests,
seeds, schedules, pipeline; outputs under separate `*_noema` roots; full
numbers in `final_metrics.json` → `noema_ablation`). SF4 no-EMA is
impossible: the released `self_forcing_dmd.pt` contains only
`generator_ema`.

| Metric | EMA (official) | no-EMA |
|---|---:|---:|
| VBench total, main-600 FFE | 0.8165 | 0.7899 |
| VBench total, main-600 all1 | 0.7552 | 0.7566 |
| VBench total, OF-4step-300 all4 | 0.7981 | 0.8201 |
| GAN7 7-dim mean @200: full / DMD-only | 0.7431 / 0.8598 | **0.8694 / 0.8194** |
| Stability 5-dim mean @100/200/400/600 | 0.849/0.853/0.933/0.948 | 0.875/0.952/0.865/0.821 |
| LPIPS diversity @200: full / DMD-only | 0.4454 / 0.6430 | 0.6440 / 0.6437 |
| FVD-256 @200: full / DMD-only | 2717.9 / 846.3 | 983.0 / 936.3 |

Observations (reported without alteration):

- The anomalous EMA step-200 results for the full method (§4, §6) are an
  **EMA artifact**: with raw weights the full method leads DMD-only on the
  GAN7 mean (+0.050), matches its diversity (0.644 vs 0.644), and has
  comparable FVD (983 vs 936). The EMA (decay 0.99, start step 50) at step
  200 still averages over early-training transients of the GAN branch.
- Conversely, at step 600 EMA helps the 1-step model (+0.027 VBench total
  with FFE), and the no-EMA stability curve degrades after step 200 while
  the EMA curve rises monotonically — EMA is what makes long training
  stable in weight space.
- The 4-step model is better without EMA at step 300 (+0.022).

No-EMA latency profiles match the EMA ones within noise (weights don't
affect timing); see `final_metrics.json`.

## 10. Ordered no-EMA follow-up experiments (NOEMA_FOLLOWUP.md)

One generated sample per official prompt (944 videos/condition), all 16
VBench dimensions, raw generator weights with fail-closed EMA provenance
audits. Deliberately labeled a one-sample protocol, not an official
five-sample submission. Full numbers: `final_metrics.json` →
`noema_followup_experiments`.

**Experiment 2 — raw step-200, framewise 1-step vs 4-step inference**
(same checkpoint, same manifest):

| Condition | Total | Quality | Semantic |
|---|---:|---:|---:|
| all1 | 0.7508 | 0.7773 | 0.6444 |
| all4 | 0.7776 | 0.8058 | 0.6646 |
| Δ (4-step − 1-step) | +0.0268 | +0.0285 | +0.0201 |

**Experiment 3 — complete VBench, raw FFE step-200 vs step-400:**

| Condition | Total | Quality | Semantic |
|---|---:|---:|---:|
| step 200 | 0.8036 | 0.8322 | 0.6892 |
| step 400 | 0.8073 | 0.8375 | 0.6868 |
| Δ (400 − 200) | +0.0037 | +0.0052 | −0.0024 |

Largest per-dimension movement at step 400: dynamic_degree −0.569 against
broad small gains elsewhere (per-dimension deltas in the JSON/CSV).

**Experiment 1 — controlled curvature intervention: in progress** at report
time (paired curved/rectified arms trained from identical raw AR teacher
trajectories, 6,505 pairs, 1,000 steps each; diff-in-diff readout
`curvature_causal_effect` to be appended when the chain completes).
Follow-up execution notes: trajectory generation was fanned to 28 nodes with
identical per-prompt seeding (byte-identical outputs to a single-node run);
three code-level fixes were required and are in the repo history
(device/dtype restoration after `assign=True` load; the repo's
`CAUSAL_DISABLE_FLEX_ATTENTION=1` segmented-flash path + bf16 autocast for
the full-sequence teacher-forcing forward, since eager flex attention
materializes a ~192 GiB score matrix; realpath-safe python wrappers for the
venv interpreters).

## 11. Sample sizes and pairing checklist (README §8)

- [x] Full method seed 0 → 600; DMD-only seed 0 → 200; 4-step seed 0 → 300.
- [x] All three runs tmux-launched via `launch_train.sh`, checked past
      step 10, `check_run.py` PASS at final step.
- [x] GAN comparison: MAIN200 vs DMD200, same FFE schedule, same
      `vbench_gan7_seed0.jsonl` manifest.
- [x] FFE on/off: both MAIN600.
- [x] 4-step comparison: both all4, real released Self-Forcing checkpoint.
- [x] Stability: steps 100/200/400/600 of one run; not presented as seed
      variance.
- [x] Official VBench: exactly 5 samples per prompt (944×5 per condition);
      subset scores are never reported as official totals.
- [x] Diversity: 4 samples/prompt × 100; FVD/coverage: 256 matched
      real/fake pairs, same real manifest for both fake sets.
- [ ] SF1 diversity condition — unavailable, disclosed above.
- [x] Long-video and curvature causal intervention marked as not covered.
- [x] Nothing unrun is reported as complete.
