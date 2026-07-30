# 20-second long-rollout evaluation with error bars (20 seeds per prompt)

Reviewer-facing summary of the long-video / error-accumulation study
(Reviewer Ga9M Q5 and the meta-review's "longer rollouts" item), with
across-seed uncertainty. Raw statistics and per-seed data:
`vbench_20s_nosink_20seed_errorbars.json`.

## Protocol

| Item | Value |
|---|---|
| Rollout length | 20 seconds (81 latent frames → 321 RGB frames, 16 fps) |
| Inference schedule | FFE: 4 denoising steps for the first 4-frame latent block, then **1 step per frame** |
| Attention | `sink_size = 0`, `local_attn_size = 21` — **identical for both methods**, verified in the recorded per-shard configs |
| Checkpoints | official releases: One-Forcing `one_forcing.pt` (raw/no-EMA), Self-Forcing `self_forcing_dmd.pt` (EMA — the only weights shipped) |
| Prompts | all 944 official VBench prompts + the shared Qwen-extended prompts |
| Seeds | 20 generation seeds per prompt, **identical seed manifest for both methods** (fully paired); 18,880 videos per method, 37,760 total |
| Scoring | each seed scored as an independent 944-video, 16-dimension VBench run (official leaderboard normalization); statistics computed across the 20 per-seed totals |
| Integrity | every video set passed exact filename-vs-manifest and 321-frame checks before scoring; 40/40 scoring runs succeeded |

## Results (VBench totals ×100, mean ± std across 20 seeds)

| Method | Total | Quality | Semantic | 95% CI (total) |
|---|---:|---:|---:|---:|
| **One-Forcing** | **78.52 ± 0.24** | 79.38 ± 0.23 | 75.11 ± 0.72 | ±0.10 |
| **Self-Forcing** | **69.08 ± 0.61** | 69.98 ± 0.73 | 65.46 ± 0.54 | ±0.27 |

**Paired per-seed gap (One-Forcing − Self-Forcing): +9.45 ± 0.63 total.
One-Forcing is ahead on 20 of 20 seeds.** Seed-level noise (σ ≤ 0.61) is an
order of magnitude smaller than the gap, so the long-rollout robustness
difference under this shared setting is unambiguous: without an attention
sink, One-Forcing degrades gracefully at 20 s while Self-Forcing drifts
(visually: color/structure collapse beginning ≈ 5–15 s, i.e., once frames
leave the 21-frame local window it was never trained to roll beyond).

Smaller-sample checkpoints of the same cells agree: 1 seed → 78.37 / 68.61;
3 seeds pooled → 78.53 / 69.28.

## Per-seed totals

| Seed | One-Forcing | Self-Forcing | Gap |
|---:|---:|---:|---:|
| 0 | 78.19 | 69.38 | +8.81 |
| 1 | 78.54 | 69.11 | +9.43 |
| 2 | 78.68 | 69.34 | +9.34 |
| 3 | 78.66 | 69.29 | +9.37 |
| 4 | 78.51 | 69.22 | +9.29 |
| 5 | 78.41 | 68.16 | +10.26 |
| 6 | 78.54 | 70.28 | +8.26 |
| 7 | 78.84 | 69.72 | +9.12 |
| 8 | 78.17 | 69.18 | +8.99 |
| 9 | 78.44 | 69.77 | +8.67 |
| 10 | 78.30 | 69.61 | +8.69 |
| 11 | 78.54 | 67.80 | +10.73 |
| 12 | 78.35 | 68.87 | +9.48 |
| 13 | 78.37 | 68.87 | +9.51 |
| 14 | 78.42 | 69.34 | +9.08 |
| 15 | 78.76 | 68.95 | +9.80 |
| 16 | 79.02 | 68.89 | +10.13 |
| 17 | 78.82 | 68.71 | +10.11 |
| 18 | 78.72 | 69.13 | +9.58 |
| 19 | 78.18 | 67.90 | +10.28 |
## Caveat (disclosed wherever these numbers are cited)

The Self-Forcing cell reflects this repository's committed framewise
rolling-window inference pipeline. The repository's earlier Self-Forcing
20-second figure (79.86) originated from an uncommitted generation path and
is not reproduced by any committed window/sink configuration we tested
(rolling-21 and full-81-frame attention both collapse); Self-Forcing's
native chunkwise multi-step mode is a different operating point and is
covered separately by the 4-step comparisons. This cell therefore answers:
"both methods under One-Forcing's long-video one-step protocol," which is
the pre-registered question for this ablation.
