# Visual inspection — official-main DMD-only, training seed 1 vs seed 48491

Real observations only. No expected/predicted scores are recorded here; see
`official_main_dmd_seed1.md`/`.json` for the numeric comparison.

## Provenance actually used

- Training code: `Aurora-edu/One-Forcing` main, commit
  `c9a2350e8740531562011fc9618e1a928d911ae0` (unmodified official `train.py`,
  via detached worktree `One-Forcing-main-c9a2350-dmd-seed1`).
- Rebuttal harness code: `Aurora-edu/One-Forcing`, commit `cdd427e` (branch
  `iclr2027-rebuttal-gan-ablation-audit`), independent HTTPS clone
  `One-Forcing-dmd-seed1` (`origin` = `https://github.com/Aurora-edu/One-Forcing.git`).
  The original seed-48491 checkouts (`One-Forcing-official-pair`,
  `One-Forcing-main-c9a2350`) were not touched.
- Arm trained: DMD-only only (`gan_g_weight`/`gan_d_weight` = 0.0). DMD+GAN
  was **not** retrained.
- Training seed: 1 (only change vs. the original pair). Same ODE
  initialization checkpoint, teacher, clean data, and 8×B200 GPUs as the
  seed-48491 run. Evaluation generation seeds are unchanged
  (`prompt_index * samples_per_prompt + sample_index`).
- Environment: same training machine/venvs as the seed-48491 run
  (`One-Forcing/.venv` for training and inference, `.venv-vbench` for
  scoring); `report_dmd_seed_repeat.py` recorded the same disclosed
  torch/torchvision version deviations as the original report
  (torch 2.11.0 found vs. 2.5.1 expected; torchvision 0.26.0 vs. 0.20.1) —
  identical deviations, not new ones introduced by this repeat.

## Jobs and elapsed time (Slurm, all on `slinky-5`, 8×B200)

- Training: job **8505**, COMPLETED, exit 0:0, 2026-09-24 18:58:00 →
  19:37:37 UTC, elapsed **00:39:37**. Detached tmux-shim session monitored
  live; step-10 health check and the step-200 checkpoint save were both
  explicitly verified (checkpoint `runs/dmd/checkpoint_model_000200/model.pt`,
  11,455,724,524 bytes — identical size to the seed-48491 DMD-only
  checkpoint).
- Evaluation: 3 submissions.
  - Job **8518**: FAILED, exit 1:0, 00:00:16 — missing local
    `wan_models/Wan2.1-T2V-1.3B` scaffolding (untracked symlink present in
    the original checkout, absent from this fresh independent clone; not
    a training-code or protocol issue).
  - Job **8519**: FAILED, exit 1:0, 00:00:16 — first fix attempt was
    structurally wrong (symlinked `wan_models` itself instead of nesting
    the `Wan2.1-T2V-1.3B` symlink inside a real `wan_models/` directory);
    same missing-model error recurred.
  - Job **8520**: COMPLETED, exit 0:0, 2026-09-24 19:55:48 → 20:56:18 UTC,
    elapsed **01:00:30**, after the symlink structure was corrected to
    match the original checkout exactly. 4,720/4,720 videos exported
    (944 prompts × 5 samples), `export.done` provenance PASS (raw/no-EMA
    generator weights confirmed), VBench scoring completed
    (`dmd_only_official_main_eval_results.json`,
    `..._paper_protocol_summary.json` both written).
- Training-log audit (`report_dmd_seed_repeat.py check-training`,
  `--min_step 200`): `last_step: 200`, `generator_updates_logged: 40`,
  `critic_updates_logged: 200` (5:1 cadence), `all_gan_losses_zero: true`,
  `all_logged_scalars_finite: true`. Matches the expected shape for a
  DMD-only run.
- `report_dmd_seed_repeat.py report`: `status: "pass"` — protocol hashes
  (`full_info`, `prompt`, `rewrite`, `manifest`), 944×5 sample count, and
  16-dimension shape all matched the seed-48491 baseline exactly; only the
  training seed (1 vs. 48491) and its downstream weights differ.
- Total wall-clock, training start to eval completion: **~2h**
  (18:58 → 20:56 UTC), well inside the 48-hour hard limit.

## Method

`ffprobe` confirmed both the new (seed 1) and original (seed 48491) clips
for all 5 prompts are 5.063s / 81 RGB frames @16fps. `ffmpeg -ss <t>
-frames:v 1` was used to grab frames at begin (~0.05s), mid (~2.53s), and
end (~4.9s), plus extra intermediate timestamps (3.2s/3.6s/4.0s/4.4s/4.7s)
for the one prompt where the end frame showed a real discrepancy, viewed
directly. All comparisons use sample index `-0` for both seeds, per the
runbook. Videos and extracted frames stay local
(`/tmp/visual_inspect_dmd_seed1/`, not committed).

## Per-prompt observations (seed 1, freshly and independently inspected — not copied from the seed-48491 report)

1. **"a dog and a horse"** (multi-object, dynamic) — **real discrepancy
   found, reported honestly below; this is not a shared-artifact match to
   the old report's stated conclusion.**
   - Seed 1 (new): at begin/mid (~0–2.5s) both animals are sharp and
     distinct. From ~3.2s onward the horse becomes strongly backlit/blown
     out (heavy rim-light, high-contrast highlight clipping not present
     earlier in the clip) and the dog progressively loses definition,
     shrinking into an indistinct, blurred tan/white blob behind the
     horse with no visible face, ears, or legs by the end frame (~4.9s).
     The horse's own body contour also looks locally warped near mid-body
     in the end frame. This is a real quality drop in the second half of
     the clip, not merely motion blur consistent with the gallop speed.
   - Seed 48491 (original, re-inspected now, not taken from the old
     written report): at begin/mid the dog is a normal light-colored dog.
     From ~3.2s the dog's proportions visibly grow and reshape toward a
     second, horse-like quadruped (longer legs, horse-like body length,
     flowing "mane/tail" texture); by ~4.7–4.9s it is rendered as an
     essentially horse-shaped white animal galloping beside the brown
     horse — i.e., an object-identity drift, not the "no collapse, no
     identity swap" characterization in the previously written report for
     this same prompt/sample.
   - **Conclusion for this prompt**: both training seeds show real
     degradation of the second subject (the dog) by the end of this
     specific clip, but of different character — seed 1 loses definition
     into blur/overexposure, seed 48491 morphs the dog toward a horse
     shape. Neither is "clean end-to-end" as the earlier written report
     stated; that characterization does not hold up under a fresh look at
     the same clips.

2. **"a car and a motorcycle"** (multi-object) — seed 1: both vehicles
   stay sharp and correctly shaped across begin/mid/end; the "motorcycle"
   is rendered as a small stationary model mounted behind the car's seats
   rather than a separate vehicle moving alongside, identical composition
   pattern to what was previously observed for both arms of the
   seed-48491 pair. No temporal collapse, no duplication, no drift.

3. **"A cute happy Corgi playing in park, sunset, black and white"**
   (dynamic) — seed 1: coherent pose/ears/tongue and stable sunset
   lighting across begin/mid/end; rendered in full color despite the
   black-and-white prompt (same style-adherence miss as previously seen
   in both seed-48491 arms). No structural collapse.

4. **"a shark is swimming in the ocean, black and white"** (dynamic) —
   seed 1: shark body/fins/teeth stay sharp and stable across
   begin/mid/end, water surface consistent; same black-and-white
   style-adherence miss (rendered in color). No deformation or collapse.

5. **"Two pandas discussing an academic paper."** (multi-object, dynamic
   hands/mouths) — seed 1: two distinct pandas holding two distinct books
   throughout, natural mouth motion, no melting, no limb/identity merging
   across begin/mid/end. No collapse.

## Overall assessment

4 of the 5 inspected prompts show no temporal collapse or identity drift
for the seed-1 repeat, consistent with the qualitative parity previously
reported for this official-main-code pair. The 5th prompt ("a dog and a
horse") shows genuine, independently-confirmed degradation of the
secondary subject toward the end of the clip in **both** training seeds,
manifesting differently in each (blur/overexposure for seed 1 vs.
identity drift toward a horse shape for seed 48491) — this nuances rather
than contradicts the earlier written report's overall qualitative-parity
claim (both seeds are still affected, so it is not something the training
seed change introduced), but the specific "no identity swap" wording for
this prompt in the earlier report does not hold up under this fresh
re-inspection of the same clip. No seed was redrawn and no parameters
were adjusted based on this or any other observation.

Extracted frames are kept locally at `/tmp/visual_inspect_dmd_seed1/` and
are not part of this commit, per the runbook's local-only instruction for
videos/frames.

Co-Authored-By: Claude Code <noreply@anthropic.com>
