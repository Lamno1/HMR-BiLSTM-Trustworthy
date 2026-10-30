# Journal revision handoff (written 2026-10-18, updated 2026-10-21)

Goal: finish the ResMemECG journal paper. Desk-rejected once at BSPC (no reviewer letter).
Supervisor's plan: target **Biomedical Engineering Letters** (Springer Nature, Q2) first;
MBEC next. Avoid Elsevier Q1 for now.

## Files
- **Base for submission: `latex/sn-article.tex`** (Springer template). Keep `latex/manuscript.tex`
  (BSPC/Elsevier) in sync -- every edit so far was applied to both with one script; `latex/bspc/`
  and `latex/overleaf_upload/` are stale copies.
- Conference (INDEA-2026) material is archived in `archive/conference_INDEA2026/` (left for later cleanup).
- Journal authors: Van Thong Nguyen, Dai Phong Luu Nguyen, Khoi Nguyen Tran, Dang Tho Le,
  Thien Khoi Nguyen, Manh Tuan Nguyen (corresponding).
- Build check: copy `latex/` (minus `bspc/` and aux files) to a scratch dir and run
  pdflatex/bibtex/pdflatex x2; the in-place `.aux` is stale and shows spurious undefined refs.

## Done (uncommitted)
- Seed table, extended ablation rows, data-source/preprocessing and baselines/recipe paragraphs,
  R2-vs-flat loss mismatch, per-class DS2 table (`tab:perclass`) + confusion matrix, SG-CV per-class
  table (`tab:sgcv-perclass`), many wording fixes.
- 2026-10-21 (self-review against an 8-question peer-review checklist):
  - `tab:literature` + Section "Comparison with published DS1/DS2 results" (`sec:literature`):
    de Chazal 2004, Llamedo 2011+, Mar 2011+, Huang 2014, Garcia 2017, Mousavi 2019 (+ = taken from
    Garcia 2017 Table 2). Numbers verified from primary PDFs/abstracts or Garcia's table; F1 derived.
    Message: ResMemECG S recall 19-44% << RR-feature methods 62-91%; V F1 0.85-0.87 within classical range.
  - "Input assumptions" paragraph (`sec:input`): annotated R peaks (no QRS detector), single beat, no RR.
  - Limitations paragraph on those assumptions and on 3 seeds.
  - Seed table caption fixed: archived seed-123/456 checkpoints used **patience 8**, not 15
    (verified from checkpoint config). "interaction-aware" toned down in Related Work/Methods.
  - SG-CV prevalence caveat (F1 not comparable across folds; use AUC columns).
  - New bib entries: garcia2017vcg, llamedo2011, mar2011, huang2014 (initials only, unverified first names).

## Matched-recipe controls (`train_matched_recipe.py`, official recipe, patience 15, seeds 42/123/456)
| arch | macro F1 per seed | mean +- SD |
|---|---|---|
| bilstm (nn.LSTM, same CNN/attention/head) | .460/.471/.489 | .473 +- .015 |
| normc (ResMemECG cell, RMC path removed) | .520/.586/.501 | .536 +- .045 |
| full (unchanged ResMemECG) | .628/.533/.515 | .559 +- .061 |
- DONE 2026-10-22. full seed 42 reproduces the archived checkpoint to 3 decimals (deterministic).
- full - normc per seed: macro +.108/-.053/+.013 (mean +.023 < seed SD); F1-N and F1-V higher in
  every seed, accuracy higher (.928 vs .895); S/F change sign; mean AUC-S/AUC-F LOWER than normc.
- Archived ResMemECG (seeds 123/456 patience 8): .628/.614/.511, mean .584 +- .064. The re-run val
  curves show patience 8 would NOT have stopped before the selected epoch, so the s123 .614 vs .533
  gap is run-to-run difference we cannot reconstruct (stated in paper). Paper now uses the re-runs.
- Independent earlier harness `modern_architectures/` (re-implemented ResMemECG, seeds 42/43/44,
  patience 8): .560/.495/.543 = **.533 +- .027** -- roughly equal to No-RMC. Must be reported.
  Same harness: CNN-Transformer .470 +- .045, pure Transformer .428 +- .012.
- No-RMC vs BiLSTM differ only in cell implementation (per-step dropout on h feeding the recurrence,
  forget-bias init 2, orthogonal W_h) yet gap .06 macro and AUC-F .90 vs .66.
- Learned blend gate beta on DS2 (archived seeds 42/123/456): .33/.21/.11 -- tracks macro F1
  (.628/.614/.511), n=3, direction of causality unknown.
- Gap decomposition idea for the paper (old BiLSTM .433 -> ResMemECG .628): recipe ~.04,
  cell implementation ~.06, RMC ~(full mean - .536), seed selection ~(.628 - full mean).

## Paper edits done 2026-10-22 (both .tex files; clean build: sn-article 33 pp, manuscript 58 pp)
- New `sec:matched` + `tab:matched` (before the ablation): matched controls, the independent
  harness .533, and the descriptive gap decomposition .04 recipe / .06 cell impl / .02 RMC / .07 seed.
- Abstract rewritten (210 words, BEL limit 250), Intro paragraph + contributions, baselines
  paragraph, main-result paragraph, tab:performance caption, seed paragraph + tab:seeds (re-runs),
  per-class paragraph + tab:perclass rows 123/456, literature text + rows, ablation text,
  SG-CV reference sentence, Discussion first subsection, hypothesis-ledger row "RMC improves rare
  classes ... Not supported", Conclusion first and last paragraphs. Broken `Table~\ref` fixed.

## Next steps
1. DONE: matched robustness (results/matched_recipe/analysis.json). Macro-F1 drop FGSM
   .056/.047/.044, PGD-20 subset .046/.055/.050 (bilstm/normc/full) = similar once recipe matched;
   conditional ASR lowest for full (FGSM .022, PGD .036). beta .33/.25/.23 (seeds 42/123/456).
   Both written into the paper; both files build clean.
2. Optional: supervisor decision on trimming R2/selective sections; code-availability URL; commit.

## 2026-10-24
- Multi-seed ablations added to `train_matched_recipe.py` (nointer, nohybrid, meanpool,
  {full,bilstm}_{noadv,ce,nocw}, full_nosmooth); user runs `run_matched_ablations.bat` in their own
  terminal (log: results/matched_recipe/ablation.log). Table/text not yet written.
- New Related Work subsection `sec:rmc-related` (RMC vs residual/highway LSTM, gate variants,
  multiplicative integration, attention) + design rationale paragraph in Methods + Discussion
  sentence; 7 new bib entries (pages for Kim 2017 / Zhang 2016 / Melis 2018 not verified -> arXiv ids).
- JOURNAL LENGTH PROBLEM: BEL Original Article <= 6,000 words INCLUDING references, legends and
  tables; max 8 figures, 8 tables, 60 references; abstract 150-250 words (Springer guideline snippet +
  2021 BEL editorial; official pages are behind a Springer login, verify by hand). sn-article now
  ~12,900 words (PDF text), 14 figures, 15 tables, 59 references -> must be cut roughly in half AT SUBMISSION TIME.
  MBEC: Original Research Article <= 18 pages. User 2026-10-24: draft stage, do NOT cut now.

## 2026-10-28: multi-seed ablations DONE and written into both .tex files
- `python -m validation.matched_ablation_table` -> results/tables/matched_ablation.{json,tex}.
- New `sec:ablation` + `tab:matched-ablation` replace the old "Architectural ablation" subsection;
  old single-seed `tab:resmemecg-ablation` kept for provenance (only source for "No CNN frontend").
- Findings (3 seeds each, paired vs same-seed reference):
  - nohybrid (beta=1) .488; below normc in EVERY seed (-.057/-.046/-.040), F1-V -.10 every seed
    -> RMC alone worse than standard update; only useful blended. Only consistent cell effect.
  - nointer .541, meanpool .539, nosmooth .541 (SD .002): mean diffs -.02, both signs -> no effect shown.
  - bilstm_ce .549 > bilstm .473 in EVERY seed: focal recipe penalises nn.LSTM. full_ce .575 (mixed).
    full_ce - bilstm_ce +.105/-.017/-.010 -> the ".06 cell implementation" gap is mostly loss x cell.
  - full_nocw .586: F1-F .387 (up every seed), F1-S .107 (down every seed) -> weights trade F vs S.
    full_nocw - bilstm_nocw +.091/+.064/+.095 (only consistent cell gap).
  - full_noadv .559 (SD .023): single-seed -.087 FGSM effect NOT reproduced.
- Edited: abstract (one sentence), contribution 2, Methods loss + smoothness sentences,
  sec:matched decomposition caveat, Discussion, 3 new ledger rows, Limitations, Conclusion.
  Build clean: sn-article 37 pp, manuscript 63 pp. Did NOT adopt best variant (test-set selection).

## Pitfalls to avoid
- Do not claim anything from one seed; report mean+-SD and per-fold values.
- Headline 0.628 is the best of 3 official seeds; say so.
- Do not call RG-CV subject-disjoint (201/202 cross folds; no INCART subject map).
- Never present R2 gains as caused by V1/RR alone (loss differs from flat).
- Verify numbers against `results/` JSON before writing; do not rely on memory.
- Literature numbers: never fill from memory; sciencedirect/mdpi/springer block automated fetches.
- Unfinished and still open: URL for code availability; sync of stale copies if submitting from them.
