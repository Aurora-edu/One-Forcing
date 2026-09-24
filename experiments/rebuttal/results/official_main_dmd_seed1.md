# Official-main DMD-only: training-seed repeat

Training code: `Aurora-edu/One-Forcing main@c9a2350e8740531562011fc9618e1a928d911ae0`. Only DMD-only is retrained, from the original ODE initialization for 200 iterations (40 generator updates). Training seed changes from 48491 to 1; evaluation generation seeds stay fixed. Both evaluations use 944 × 5 Qwen-conditioned prompts, FFE, the generator weights (not EMA), and all 16 VBench dimensions.

| DMD-only run | Total | Quality | Semantic |
|---|---:|---:|---:|
| Original seed 48491 | 83.52 | 85.10 | 77.19 |
| Repeat seed 1 | 83.45 | 84.89 | 77.69 |
| Difference (1 − 48491) | -0.06 | -0.21 | 0.51 |

DMD+GAN was not retrained. This report does not estimate a same-seed GAN gain.

| Dimension (raw score) | Seed 48491 | Seed 1 | Difference |
|---|---:|---:|---:|
| aesthetic_quality | 0.649607 | 0.647825 | -0.001782 |
| appearance_style | 0.198093 | 0.199810 | +0.001717 |
| background_consistency | 0.949788 | 0.950254 | +0.000466 |
| color | 0.825660 | 0.801493 | -0.024167 |
| dynamic_degree | 0.672222 | 0.797222 | +0.125000 |
| human_action | 0.900000 | 0.912000 | +0.012000 |
| imaging_quality | 0.706984 | 0.659117 | -0.047867 |
| motion_smoothness | 0.988291 | 0.984183 | -0.004108 |
| multiple_objects | 0.803049 | 0.818293 | +0.015244 |
| object_class | 0.948259 | 0.951582 | +0.003323 |
| overall_consistency | 0.253974 | 0.255534 | +0.001561 |
| scene | 0.540843 | 0.535756 | -0.005087 |
| spatial_relationship | 0.761773 | 0.796185 | +0.034412 |
| subject_consistency | 0.966010 | 0.957781 | -0.008230 |
| temporal_flickering | 0.991856 | 0.990693 | -0.001163 |
| temporal_style | 0.240078 | 0.240372 | +0.000295 |
