# Provenance check: 20-second long-video table conflict (Self-Forcing 75.30)

Requested by `experiments/rebuttal/ICLR2027_ALIGNED_EXPERIMENTS.md`:

> There is a paper-data conflict to resolve before updating the long-video
> table: `iclr2027.tex` prints Self-Forcing 75.30 and One-Forcing 78.52; the
> committed 20-seed, no-sink Qwen report records Self-Forcing 69.08 and
> One-Forcing 78.52... The separate single-seed sink comparison records yet
> another pair (79.86/80.14)... Locate and audit the raw export/config/result
> behind 75.30, or correct the paper table using a single defensible paired
> protocol. This is a provenance check, **not** authorization to synthesize
> or substitute a favorable number.

This is a **provenance check only** — no numbers were changed, chosen, or
substituted, and no paper table was edited.

## What was checked

`iclr2027.tex` — searched for across the entire repository, both the current
working tree and the **full git history** (`git log --all -p -- '*.tex'`,
plus a full-tree `grep` for `75.3`/`75.30`/`iclr2027.tex`):

- **Not found.** No file named `iclr2027.tex` (or any `.tex` file at all)
  exists anywhere in this repository's working tree or commit history, and
  no other file in the repository contains the string `75.30` except the
  aligned-experiments document's own description of the conflict.

**Conclusion: the raw export/config/result behind the paper's 75.30 figure
cannot be audited from this repository.** The paper source (`iclr2027.tex`)
must live in a separate paper-writing repository, Overleaf project, or other
location not present on this training machine or in this git history. This
report cannot locate, and therefore cannot audit, that number's provenance —
and does not attempt to guess at or reconstruct one.

## What *is* available in this repository, for context

Three distinct Self-Forcing 20-second figures exist across this repo's
committed results, and their provenances differ:

| Figure | Self-Forcing | One-Forcing | Source | Protocol |
|---|---:|---:|---|---|
| Paper (`iclr2027.tex`) | 75.30 | 78.52 | **not found in this repo — unauditable from here** | unknown |
| 20-seed no-sink (this repo) | 69.08 ± 0.61 | 78.52 ± 0.24 | `LONGVIDEO_20SEED_ERRORBARS.md` / `vbench_20s_nosink_20seed_errorbars.json` | fully paired, identical seed manifest, identical sink=0/window=21 config for both methods, 20 seeds × 944 prompts, each method's own official checkpoint convention (SF EMA, OF raw) |
| Single-seed sink grid (this repo) | 79.86 | 80.14 | `RESULTS_SUMMARY.md` §"20-second long-video attention-sink grid" / `vbench_20s_sink_ablation_seed0.json`, `vbench_20s_self_vs_one_seed0.json` | single seed (seed 0), **not sink-matched between methods** (SF at sink=0/no-sink; OF figure shown is sink=3) |

Notable: **One-Forcing's number is identical (78.52) between the paper and
the 20-seed no-sink report**, which is at least internally consistent for
that method. Only the Self-Forcing figure conflicts across all three
sources.

The 69.08-vs-79.86 discrepancy (both computed in this repo) is already
explained in `LONGVIDEO_20SEED_ERRORBARS.md`'s caveat: the 79.86 figure
"originated from an uncommitted generation path and is not reproduced by any
committed window/sink configuration we tested (rolling-21 and full-81-frame
attention both collapse)." In other words, 79.86 is not reproducible from
any pipeline currently in this repository, whereas 69.08 is — it comes from
the fully paired, 20-seed, identical-protocol, committed run.

75.30 does not match either of this repo's two reproducible-or-explained
Self-Forcing figures (69.08 or 79.86), and its generating code/config is not
present here at all, so it cannot be assessed for methodological soundness
(sink size, seed count, chunkwise vs. framewise mode, window size, EMA vs.
raw weights, etc.) — nor ruled in or out as a third distinct, possibly valid,
operating point (e.g., Self-Forcing's native chunkwise multi-step mode,
which `LONGVIDEO_20SEED_ERRORBARS.md` notes is "a different operating point
... covered separately by the 4-step comparisons").

## Recommendation (decision left to the paper owner, not made here)

This report does not pick a number. Options, for whoever has access to
`iclr2027.tex` and its generating pipeline:

1. **Locate `iclr2027.tex` and its 75.30 source** (wherever the paper repo
   actually lives) and audit its config/export directly — the one action
   this report could not perform from here.
2. **Adopt the 20-seed no-sink pair (69.08 / 78.52)** as the paper table
   value: it is the only one of the three that is fully paired (identical
   seed manifest and attention config for both methods), uncertainty-
   quantified (20 seeds, σ ≤ 0.61), and fully reproducible from code
   currently in this repository.
3. If 75.30 turns out to reflect a genuinely different, defensible protocol
   (e.g., Self-Forcing's native chunkwise mode rather than the one-step
   rolling-window mode used here), **label it as such explicitly** in the
   table rather than presenting it alongside 78.52 as if paired under the
   same protocol.

No paper file was found or edited as part of this check.
