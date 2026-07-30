# Rebuttal campaign — master results table

One-line-per-result index of every experiment in this campaign. Full numbers,
per-dimension scores, CIs, and protocols: `final_metrics.json` / `final_metrics.csv`,
`RUN_REPORT.md`, and the standalone JSONs in this directory. Unless noted:
seed 0 (single training seed), 944 official VBench prompts, hardware 8×H200
per node, "total" = official VBench leaderboard normalization (quality:semantic 4:1).

## Training (all seed 0, `check_run` PASS, EMA verified complete)

| Run | Steps | s/step | Wall clock |
|---|---:|---:|---:|
| Full One-Forcing | 600 | 23.98 | 4.09 h |
| DMD-only | 200 | 13.59 | 0.93 h |
| 4-step One-Forcing | 300 | 24.71 | 2.31 h |

## Official VBench, 944×5 samples (EMA weights)

| Condition | Total | Quality | Semantic |
|---|---:|---:|---:|
| Main step-600, FFE | **0.8165** | 0.8501 | 0.6820 |
| Main step-600, all1 (FFE off) | 0.7552 | 0.7925 | 0.6064 |
| 4-step One-Forcing step-300, all4 | **0.7981** | 0.8244 | 0.6927 |
| 4-step Self-Forcing (released), all4 | 0.7375 | 0.7638 | 0.6323 |

FFE contribution (same checkpoint): **+0.0613**. 4-step OF vs SF: **+0.0606**.

## No-EMA ablation (raw weights, 944×5)

| Condition | Total |
|---|---:|
| Main step-600, FFE | 0.7899 |
| Main step-600, all1 | 0.7566 |
| 4-step One-Forcing step-300, all4 | 0.8201 |
| 4-step Self-Forcing | n/a — release ships `generator_ema` only |

## DMD-only vs DMD+GAN, paired at step 200

| Metric (same init/data/budget/manifest) | DMD+GAN | DMD-only | Δ (GAN) |
|---|---:|---:|---:|
| 7-dim VBench mean, raw | **0.8694** | 0.8194 | **+0.050** |
| Complete 16-dim total, raw, 944×1 | **0.8036** | 0.7527 | **+0.0509** |
| 7-dim mean, EMA (step-200 EMA transient; see RUN_REPORT §9) | 0.7431 | 0.8598 | −0.117 |

## Stability across training (one 600-step run)

| Step | 5-dim mean (EMA) | 5-dim mean (raw) | Full-16 total |
|---:|---:|---:|---:|
| 100 | 0.849 | 0.875 | — |
| 200 | 0.853 | 0.952 | 0.8036 (raw, 944×1) |
| 400 | 0.933 | 0.865 | 0.8073 (raw, 944×1) |
| 600 | **0.948** | 0.821 | 0.8165 (EMA, official 944×5) |

No adversarial collapse anywhere; raw-weight late drift is confined to
dynamic_degree; EMA rises monotonically.

## Diversity (LPIPS-VGG, 100 prompts × 4 seeds, step-200 pair)

| Condition | Raw | EMA |
|---|---:|---:|
| DMD+GAN (full) | 0.6440 [0.6346, 0.6529] | 0.4454 |
| DMD-only | 0.6437 [0.6328, 0.6549] | 0.6430 |
| Causal-Forcing init (all1 / all4) | 0.6742 / 0.7080 | — |
| 1-step Self-Forcing | n/a — no official checkpoint exists | — |

GAN loss costs no measurable diversity (raw, overlapping CIs).

## FVD + precision/recall/density/coverage (raw @200, shared real set)

| N pairs | Method | FVD ↓ | P | R | D | C |
|---:|---|---:|---:|---:|---:|---:|
| 256 | DMD+GAN | 983.0 | 0.680 | 0.711 | 0.494 | 0.731 |
| 256 | DMD-only | 936.3 | 0.848 | 0.457 | 1.374 | 0.910 |
| 512 | DMD+GAN | 692.8 | 0.637 | 0.688 | 0.474 | 0.725 |
| 512 | DMD-only | **687.2** | 0.840 | 0.418 | 1.429 | 0.900 |

At 512 pairs the FVD gap closes (Δ 5.6); the recall-vs-precision profile
difference persists. (EMA @200 reference: 2717.9 / 846.3 — EMA transient.)

## Latency (1×H200, bf16, 21 latents, 20 trials, incl. VAE)

| Schedule | First block | Steady/frame | End-to-end |
|---|---:|---:|---:|
| all1 | 102 ms | 117.8 ms | 4.13 s |
| FFE | 478 ms | 120.8 ms | 4.19 s |
| all4 | 244 ms | 290.9 ms | 7.72 s |

## Inference-budget generalization (same raw step-200 checkpoint, 944×1)

| Schedule | Total |
|---|---:|
| all1 (1-step) | 0.7508 |
| all4 (4-step) | **0.7776** (+0.0268) |

## Qwen-matched 4-step comparison (historical protocol, SF videos reused, 944×1)

| Condition | Total |
|---|---:|
| One-Forcing 4-step (raw) | **0.8385** |
| Self-Forcing 4-step (existing official result) | 0.8346 |

Δ = +0.0039 (near parity; protocol-dependent magnitude, consistent direction).

## Curvature causal intervention (paired arms, identical except trajectory straightening)

| Budget | Gain all1 (primary) | Gain all4 (control) | DiD (causal effect) |
|---|---:|---:|---:|
| 1,000 steps | +0.0017 | +0.1285 | **−0.1268** |
| 300 steps (cd_small) | −0.0144 | +0.0995 | **−0.1139** |

Negative at both budgets → the original curvature-causes-one-step-failure
claim is **not supported**; paper revised to observational framing.

## 20-second long-video attention-sink grid (944×1, FFE, official ckpts)

| Total @20 s | No sink | Sink 3 |
|---|---:|---:|
| One-Forcing (raw) | 0.7837 | **0.8014** |
| Self-Forcing (EMA) | **0.7986** | 0.6861 |

Sink helps OF (+0.018) but breaks SF (−0.113; not trained with a sink).
Fair per-release-configuration pairing: OF 0.8014 vs SF 0.7986 (+0.0028).
60-second VBench-Long, OF sink-3: 0.7770.
