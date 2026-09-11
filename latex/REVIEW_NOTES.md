# Review notes — ResMemECG manuscript (BSPC submission)

Reviewer-style critique of `manuscript.tex` + the fixes applied. Kept here in
`latex/`, **not** inside the submission package (`latex_teacher_review/`).
Line numbers below refer to the Springer draft (`sn-article.tex`); the same
edits are in `manuscript.tex`.

---

## A. Author section

| # | Issue | Status |
|---|---|---|
| A1 | Author order contradicted the CRediT roles (Van Thong Nguyen = lead author but listed 6th). | **FIXED** — Van Thong Nguyen first; Manh Tuan Nguyen last + corresponding; CRediT reordered to match. |
| A2 | Corresponding author had no ORCID. | **FIXED** — Thien Khoi Nguyen (0009-0002-0036-523X) and Manh Tuan Nguyen (0000-0002-7158-6550) added; all 6 authors now have an ORCID. |
| A3 | ORCIDs rendered as literal bracketed text. | **FIXED** by the Elsevier conversion — linked `\orcidlink` iD icons for all six authors. |
| A4 | CRediT roles look padded for a 6-author single-institution paper. | Open — your call. "Literature investigation" (not a CRediT term) changed to "Investigation". |
| A5 | No clinical / biomedical co-author for a clinically framed paper. | Open — your call. |
| A6 | Funding stated twice, verbatim. | **FIXED** — single "Funding" statement; Acknowledgements now thanks UEH for computing only. |
| A7 | Code-availability is a promissory note. | Open — supply the repo URL before/at submission. |
| A8 | Prior conference presentation (INDEA 2026). | Not yet published, no formal disclosure triggered. Add a cover-letter line + footnote once INDEA proceedings appear. |

## B. Major scientific concerns

- **B1 — R2 + selective-prediction run on the deprecated RG-CV protocol** (half the
  Results, one whole contribution). **Softened, not removed** (you asked to keep
  the content): contribution #3 now reads "a diagnostic extension ... evaluated
  only on a legacy ... protocol ... reported as an exploratory sub-study, not a
  validated method"; abstract reworded; new Limitations paragraph states a
  confirmatory SG-CV run is future work.
- **B2 — the RMC cell has no multi-seed ablation and the DS1/DS2 win is
  recipe-confounded.** Softened: abstract says "exceeding ... baselines ... although
  that comparison shares the training recipe rather than isolating the
  architecture"; section 6.2 states the ablation reference (0.596) is an earlier
  config and RMC is "a design choice rather than a significance-tested source of
  improvement".
- **B3 — "pre-registered" (x11)** changed to "pre-specified" everywhere (revert
  only if a public OSF/registry link exists).
- **B4 — statistical claims vs. "not independent"**: added a sentence -- 5 folds /
  2 non-independent seeds, no significance tests, all fold comparisons
  descriptive.
- **B6 — two "flat under CV" F1-F numbers (0.0109 vs 0.053)**: section 5.4 now
  reconciles all three F-class numbers (0.0109 / 0.053 / 0.070) in one sentence.

If a reviewer **insists** on B1/B2 experiments (estimates on the RTX 3050 Ti,
4.7 h/run measured):
- same-recipe baselines (LSTM/BiLSTM, focal + cosine + FGSM): ~2 days incl. ~0.5 day code.
- multi-seed ablation (3 seeds x 5 variants): ~4-6 days GPU.
- R2 + selective under SG-CV: ~5-7 days, plus real risk the conformal analysis is
  degenerate on MIT-only sparse F folds (19/17/23 beats) and stays "exploratory".

## C. Consistency fixes (all applied)

C1 line 248 "RG-CV" to SG-CV; C2 record/subject terminology in contribution #4,
section 2.4, 6.1, 8; C3 hard "Section 7" to `\ref{sec:selective}` (+ 4 new
`\label`s); C4/C6 section 5.3 no longer pivots to 0.053 mid-para, INCART
preprocessing flagged legacy-only; C5 abstract recipe caveat; C7 Fig 1(b) removed
the duplicate h_{t-1} to lstm arrow, added the missing c_{t-1} to lstm; C8 "93% F
from 213" stated once; C9 "state-locked" jargon compressed; C10 ablation 0.596
vs 0.628 explained; C11 three F1-F numbers reconciled; C12 **verified** --
pixel-measured the noise figure at sigma=0.5 (BiLSTM ~0.398, ResMemECG ~0.388),
matches the text "0.396 vs 0.387"; C13 section 3.5 marked as diagnostic extension.

## D. Minor (applied)

D1 "Acknowledgements"; D4 Table 4 total support (2,780 S / 802 F); D5 keywords
+ selective prediction, conformal; D6 ethics: source-database IRB line; D7
contribution #4 "unseen subjects"; D8 "(R2)" glossed once. D2/D3 (forensic
figures to supplementary, unify patient/subject/record vocab) left as-is.

---

## Format conversion to BSPC / Elsevier (2026-09-09)

New submission source **`manuscript.tex`** -- Springer `sn-jnl` to Elsevier
`elsarticle [review,12pt]` (single column, double spaced, line numbers). The
Springer version stays at `latex/sn-article.tex` (do not edit it further --
`manuscript.tex` is canonical).

Added / changed for Elsevier compliance:
- `\begin{frontmatter}` with `\author[ueh]{}` + `\ead{}` + `\affiliation{organization=...}`, `\corref`/`\cortext`.
- ORCID via `\orcidlink` (linked iD icons).
- `\bibliographystyle{elsarticle-num}`.
- Back matter to Elsevier headings: CRediT authorship contribution statement, Declaration of competing interest, Funding, Ethics statement, Data availability, Acknowledgements. Springer-only items removed.
- **`highlights.tex`** created -- BSPC requires it (5 bullets, each <= 85 chars).
- `microtype` dropped (font clash in review mode); `lineno` added.

For the final/accepted version: switch `[review]` to `[5p]` and supply the repo URL.

---

## Overleaf "compile timed out" (2026-09-09)

The 3 `hyperref` "Token not allowed in a PDF string" warnings were **not** the
cause. They came from `\orcidlink` while hyperref built the PDF metadata.

- **Warnings fixed at the source**: explicit `pdftitle`/`pdfauthor` in
  `\hypersetup{}` (hyperref then never parses `\author{}`/`\title{}`) plus a
  `\pdfstringdefDisableCommands` guard. `\orcidlink` icons kept. Result: **0
  hyperref warnings, 0 errors.**
- **The document is healthy**: ~5 s per pass locally, 45 pages. The timeout is
  Overleaf infrastructure.
- **Trimmed to help Overleaf**: removed the dead `\iffalse ... \fi` TikZ block and
  the `tikz`/`pgf` load (figure is the pre-rendered
  `images/resmemecg_architecture.pdf`); package now carries only the 13 figures
  actually used (`images/` 9 MB to 2.1 MB); `manuscript.bbl` included so BibTeX
  can be skipped.
- **If it still times out**: (1) hit Recompile -- the first compile on a free
  server is slowest; (2) Overleaf Menu -> raise "Compile timeout" (paid plan);
  (3) for drafting use `\documentclass[preprint,12pt]{elsarticle}` and comment
  `\linenumbers` (`lineno` on a 45-page double-spaced doc is the main remaining
  cost), switch back for the one real submission compile; (4) compile locally
  (`pdflatex -> bibtex -> pdflatex x2`).
