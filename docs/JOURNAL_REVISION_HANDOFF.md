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

## 2026-10-30: leakage audit + workflow figure
- `python -m validation.leakage_audit` -> results/tables/leakage_audit.json. Re-extracted beats
  reproduce inter_{train,val,test}.npz exactly (diff 0, same labels/order); record sets disjoint;
  0 identical raw beats across splits; stored norm = train-only stats. SG-CV manifest: groups
  disjoint in all 5 folds, each tested once, 201/202 one group. Caveats now in paper:
  201 (DS1 train) / 202 (DS2 test) same subject -> DS1/DS2 "inter-patient"/record-disjoint, not
  strictly subject-disjoint; record 114 channel 0 = V5 in DS1/DS2 pipeline (1,878 train beats).
- New `\paragraph{Leakage checks}` (sec:leakage) after Data source; Fig. `fig:workflow` (TikZ) at
  start of Experimental Protocol; manuscript.tex now loads tikz.
- manuscript.tex / sn-article.tex / mau1.bib are no longer tracked by git (user: don't publish).
- BEL/MBEC fit review still needs papers from the user (Springer login).

## 2026-10-30: residual BiLSTM baseline DONE and written in
- `ResidualLSTMControl` (2 separate bidirectional nn.LSTM, h1+h2 before attention, 391,750 params
  = bilstm); `run_residual_lstm.bat`; rows reslstm / reslstm_ce in matched_ablation.{json,tex}.
- reslstm .481 +- .016 (vs bilstm +.011/+.003/+.011, F1-F .006); reslstm_ce .498 +- .054
  (vs bilstm_ce +.037/-.024/-.165; s456 F1-F .002 = no gain from CE, unlike plain BiLSTM).
- full - reslstm +.158/+.060/+.015; full_ce - reslstm_ce +.068/+.007/+.155; BUT normc - reslstm
  +.050/+.113/+.002 too -> gap is cell implementation vs nn.LSTM, does not isolate RMC.
- Paper: 2 table rows + caption, new paragraph in sec:ablation, pointer in sec:rmc-related,
  ledger row "Not supported", "thirteen" -> "fifteen configurations". Build clean: sn-article 39 pp,
  manuscript 67 pp.

## 2026-10-30: full-paper review + depth diagnostics (no GPU)
- Review fixes: literal TAB+"exttt" (2x) in sec:leakage; ensemble = earlier patience-8 ckpts
  (results/checkpoints/ensemble, .614/.511), not re-runs; recipe ref -> new \label{sec:recipe};
  "nine" -> "ten further variants"; temperature = 1 scalar fitted on 4 val records, seed-42 model.
- `python -m validation.matched_diagnostics` -> results/tables/matched_diagnostics.json +
  latex/images/confusion_matrix_seeds.png (3 seeds + sum; replaces single-seed confusion fig).
- Mechanism: nn.LSTM + class-weighted focal predicts F for 1,547/7,081/7,300 beats (388 true F);
  L (LBBB) beats = 9% of N but 27% of its false F; L->F 269/1,919/1,982 of 4,123 vs full 2-21,
  normc 18-104; CE 2/259/1,318; no class weights 1/0/3 -> the F class weight is the driver.
- Record-cluster bootstrap (22 DS2 records, B=2000, models fixed): intervals exclude 0 only for
  full-bilstm, bilstm_ce-bilstm, nohybrid-normc; full-normc [-0.001,+0.051]; record 213 most
  influential in 8/9 pairs. New tab:record-bootstrap. Build: sn-article 40 pp, manuscript 69 pp.
- Open depth items needing GPU: isolate which nn.LSTM-vs-custom-cell difference (recurrent
  dropout / forget bias / orthogonal init) causes the gap; BiLSTM + No-RMC under SG-CV.

## Pitfalls to avoid
- Do not claim anything from one seed; report mean+-SD and per-fold values.
- Headline 0.628 is the best of 3 official seeds; say so.
- Do not call RG-CV subject-disjoint (201/202 cross folds; no INCART subject map).
- Never present R2 gains as caused by V1/RR alone (loss differs from flat).
- Verify numbers against `results/` JSON before writing; do not rely on memory.
- Literature numbers: never fill from memory; sciencedirect/mdpi/springer block automated fetches.
- Unfinished and still open: URL for code availability; sync of stale copies if submitting from them.

## 2026-10-31: cell-implementation isolation DONE (run_cell_impl.bat)

3 seeds each, paired with normc (0.536):

| Variant | Macro F1 (SD) | Paired vs normc | Record bootstrap 95% |
|---|---|---|---|
| normc_nodrop (no in-cell dropout) | 0.573 (0.024) | +.031/+.013/+.068 | [+.009, +.061] |
| normc_nofb (forget bias 0) | 0.499 (0.039) | -.039/-.114/+.043 | [-.083, +.017] |
| normc_torchinit (nn.LSTM init) | 0.543 (0.028) | -.001/-.050/+.073 | [-.017, +.039] |
| normc_lstmlike (both) | 0.456 (0.077) | -.125/-.157/+.041 | [-.125, -.016] |

- lstmlike returns to the BiLSTM level (0.473): the normc - bilstm gap is implementation, not architecture.
- Effects are not additive (torchinit alone ~0, nodrop alone +, both together -), 3 seeds only.
- normc_nodrop 0.573 >= full 0.559; full - normc_nodrop = -0.014, bootstrap [-.033, +.007].
  ResMemECG also has the in-cell dropout -> fair test is full_nodrop (run_full_nodrop.bat, ~6 GPU h).
  NOT yet written into the paper; waiting for full_nodrop.

## 2026-11-01: full_nodrop DONE + written into both .tex

- full_nodrop 0.571 (0.059); vs full +.007/+.025/+.005 (bootstrap [-.006, +.028]);
  vs normc_nodrop +.084/-.041/-.049, mean -0.002, bootstrap [-.020, +.021].
- => RMC +0.023 exists only because the in-cell dropout hurts the No-RMC cell more. No evidence RMC helps.
- Paper: new sec:cellimpl (after the ablation section, before SG-CV) + tab:cell-impl (incl. L->F counts);
  6 rows added to tab:record-bootstrap; abstract, contribution 2, sec:matched, discussion, ledger (+1 row),
  conclusion updated. sn-article 42 pp, manuscript 72 pp, 0 errors / 0 undefined.

## 2026-11-01: depth / mechanism analyses (no retraining; saved checkpoints)

Scripts: validation/mechanism_lbbb.py, mechanism_lbbb_train.py, mechanism_transfer.py
(-> results/tables/mechanism_*.json).

1. Every core fits the training subjects perfectly: AUC(F vs N) on DS1-train = 0.996-1.000 for all
   11 configs x 3 seeds; L vs F also ~1.000 on train, L->F <= 9 beats.
2. On DS2 the F ranking collapses for some cores: AUC(F vs N) bilstm .741/.670/.593, normc .919/.933/.845,
   full .959/.915/.750. AUC(F vs L) for bilstm .445/.280/.423 (below chance) vs .97/.96/.84 without class weights.
3. L->F is almost entirely record 214 (one unseen LBBB patient; up to 1,913/2,001 beats). BUT removing 214
   leaves every macro-F1 gap unchanged (normc-bilstm +.063 -> +.060): L/214 is a symptom, not the cause.
   Gap normc-bilstm per-class share N/S/V/F = .22/.20/.17/.41 (spread over classes).
   -> the paper sentence "mainly a failure mode..." must be softened.
4. Validation is blind: across 60 runs Spearman(val macro F1, test macro F1) = 0.13 (p .33); across the 20
   config means rho = 0.03 (bootstrap 95% -0.45..0.52). By validation the plain BiLSTM ranks 3rd of 20
   (val .588, test .473). AUC(F vs N) val->test rho = -0.04 (33 model-seeds). Validation has 4 records,
   30 F beats, 0 L beats; DS1-train F = 372/384 from record 208.
   => Candidate central thesis: all cores interpolate the training subjects; DS2 differences are differences in
   cross-subject transfer set by implicit regularisers (in-cell dropout, init, class weights), which the
   validation set cannot observe -> architecture rankings on DS1/DS2 are not identifiable at selection time.

## 2026-11-01 SG-CV generality test (scripts ready, waiting for GPU runs)
- `validation/run_sgcv_arch.py`: matched-recipe cores (train_matched_recipe.ARCHS) under the locked
  C1_MIT_SUBJECT_CV_V1 folds (checked against results/controlled_mit_subject_cv/protocol_manifest.json).
  Saves best-state val + test logits. Out: results/sgcv_arch/<arch>/.
- `run_sgcv_arch.bat`: bilstm, bilstm_nocw, bilstm_ce (~1 h each), normc_nodrop, normc (~15 h each).
- `validation/sgcv_validation_blindness.py`: within-fold-seed val->test Spearman, val-pick hit rate +
  regret, paired test diffs. Existing flat runs = arch `full`.
- Note: inner val always contains the F-rich subject (record 208; val F ~377-388). Train F is 49-51 in
  folds 0-1 only; folds 2-4 train on ~400 F (record 213 in train) but test only 17-23 F.
- Flat full only, across folds: val->test Spearman +0.67 (n 10) -- fold difficulty, not a selection test.

## 2026-11-03 SG-CV generality test DONE (6 cores x 5 folds x 2 seeds = 60 runs)
Output: results/tables/sgcv_validation_blindness.json (python -m validation.sgcv_validation_blindness).
- Test macro F1 means (SD over fold-seeds ~0.08-0.11): full .475, normc .477, normc_nodrop .470,
  bilstm .449, bilstm_nocw .479, bilstm_ce .480. Val means: bilstm_ce highest .569, full lowest .523.
- Paired test diffs, fold-cluster bootstrap 95%: full-normc -.002 [-.015,.012] (RMC null replicates);
  normc-bilstm +.028 [-.011,.079]; full-bilstm +.026 [-.007,.074]; normc-normc_nodrop +.007 [.000,.011]
  (sign opposite to DS1/DS2 +.037 for nodrop); bilstm_nocw-bilstm [.003,.057].
- The bilstm deficit = one run: fold2 seed123 best_epoch=1 (val .539 at epoch 1 never exceeded,
  test .39 vs ~.64 for the others; S F1 .06, V->N 684). Without it normc-bilstm ~ +.003.
- Within fold-seed val->test Spearman over 6 cores: mean +0.08 (n 10); val pick = test best 2/10
  (chance 1.7); mean regret .016. Val picks bilstm_ce 5/10, test best for it 1/10.
- Across all 60 runs Spearman +0.62: val tracks fold (subject) difficulty, not core differences.
- F F1 ~0 everywhere under SG-CV (F-rich record 208 always in inner val) -> not informative for F.

## 2026-11-03 SG-CV + transfer findings written into both .tex (step 1 DONE)
- New script validation/ds2_validation_blindness.py -> results/tables/ds2_validation_blindness.json
  (60 runs rho .13 p .33; 20 config means rho .03 boot [-.46,.51]; record-214 removal gaps).
- Abstract: removed "48% of BBB beats" framing; added train-fit, DS2 val rho .03, SG-CV tie/RMC -0.002/rho .08.
- Mechanism paragraph: "mainly a failure mode" replaced by record-214 = symptom (gaps +.063->+.060, +.086->+.085).
- New \subsection sec:transfer (before SG-CV subject-composition) and sec:sgcv-arch + tab:sgcv-arch
  (before mixed-source RG-CV). sn-article: table* + resizebox (plain table + resizebox breaks sn-jnl threeparttable).
- Ledger +2 rows (DS1/DS2 ranking generalises: Not supported; validation identifies better core: Not supported);
  conclusion + limitation sentence; highlights item 4 replaced.
- Build: sn-article 44 pp, manuscript 76 pp, 0 errors, 0 undefined. Backups: scratchpad old_sgcv/.
- NEXT: step 2 = latex/bel-article.tex (<=6000 words) around the transfer/validation thesis; add to .gitignore.

## 2026-11-03 BEL version DONE (step 2)
- latex/bel-article.tex (12 pp; ~4,700 words incl. title page, abstract 248, captions, refs; 20 refs;
  5 tables, 2 figures). Title unchanged (user). Structured abstract Purpose/Methods/Results/Conclusion.
  Thesis: cores interpolate training subjects; differ only in transfer; validation (DS1 and SG-CV inner)
  cannot see it; RMC null. New fig latex/images/validation_agreement.png
  (validation/plot_validation_agreement.py, also prints the transfer-AUC table numbers).
- latex/bel-supplement.tex = Online Resource 1 (23 pp), generated from sn-article.tex by
  latex/bel_tools/build_supp.py (xr-hyper refs "M-" to bel-article labels). Regenerate after editing
  sn-article; build bel-article first (needs its .aux).
- New refs in mau1.bib (metadata checked online): bouthillier2021variance, cawley2010selection, varoquaux2018cv.
- .gitignore: bel-article.tex, bel-supplement.tex, their PDFs, latex/bel_tools/.
- Open: title vs content mismatch (user decision); venue BEL vs MBEC.

## 2026-11-03 Title changed + BEL expanded (user: "điểm 1 sửa đi", use the free budget)
- New title (bel-article, sn-article, manuscript, highlights, supplement): "A Controlled Evaluation of a
  Residual-Memory LSTM for Inter-Patient ECG Classification: Recurrent Cores Differ in Transfer to Unseen
  Patients That Validation Cannot Observe"; running title "Controlled Evaluation of Recurrent Cores for
  Inter-Patient ECG".
- BEL now 8 tables / 5 figures, ~5,270 words (limit 6,000): + baselines with per-class P/R/F1, literature
  table, record-bootstrap table, SG-CV per-fold P/R/F1/AUC; + DS2 concentration fig, confusion matrices,
  transfer-AUC fig (validation/plot_transfer_auc.py -> latex/images/transfer_auc.png).
- Supplement regenerated without the moved parts (18 pp): full ablation, single-seed ablation, RG-CV/R2,
  reliability, rare-class forensics, selective NO-GO.

## 2026-11-03 Full research record (user: "1 bản có full tất cả từ con số 0")
- latex/full-record.pdf (848 pp, XeLaTeX), generated by latex/full_record_tools/build_full_record.py
  (then xelatex x1, bibtex, xelatex x2). All local / gitignored (latex/full-record.*, latex/full_record_tools/).
- Parts: I conference paper (pdfpages) | II 16 Markdown logs/protocols/reports via pandoc + memory notes |
  III full sn-article body + BEL-only figures | IV every .tex/.csv/.json under results/ and outputs/ |
  V gallery of every distinct image | VI BEL article (pdfpages).
- Caps: JSON numeric arrays >50 -> n/min/mean/max; tables >400 rows -> first 40 rows + column summary;
  JSON with same-name CSV shown as CSV; byte-identical files once. Excluded: other authors' PDFs,
  certificates, raw data, checkpoints, console logs, older draft PDFs (advisor/BSPC).

## 2026-11-03 (late): second external review of bel-article, edits applied
- New: `python -m validation.sgcv_variance_decomposition` -> results/tables/sgcv_arch_runs.csv
  (60 SG-CV runs) + sgcv_variance_decomposition.json. Balanced 6 core x 5 fold x 2 seed SS split
  of outer-test macro F1: fold 85.0%, core 1.2%, core x fold 4.3%, seed-within-cell 9.5%;
  matches statsmodels two-way ANOVA (|diff| 9e-16). Descriptive only (runs share subjects).
- bel-article.tex: E1 (three-times sentence -> variance shares, + Methods sentence), E2 (Spearman
  "no better than chance" -> uninformative in this experiment, with CI), E3 ("no selection
  procedure" -> validation-based selection used here), Table 7 last row "Total / mean",
  B1 ("every model fits training" -> eleven re-scored configs, F-vs-N discrimination; abstract,
  contributions, 3.5, Fig. transfer caption, Discussion, Conclusion), B2 ("mainly in transfer",
  "in the evaluated configurations"), B3 (0.195 decomposition: along this sequence, no additivity).
- Now: 15 pp, abstract 248, text 5,555 (+~930 table-cell words). No GPU rerun (record-208 left
  to Limitations, user decision).
- Round 3 (same day): Conclusion sentence -> "apparent gain is accounted for, in the evaluated
  sequence of matched comparisons, ... rather than by a demonstrated independent benefit of RMC".
  Cut to 14 pp; words incl. table cells + refs: 5,838 (section counter) / 5,899 (bel_count +
  table cells) / 5,770 (pdftotext); abstract 248. Cuts only in Intro, Results text restating
  tables, Discussion, one caption note; Methods, tables, Limitations untouched.
- Supplement: new S2 "Variance decomposition of the SG-CV results" (formulas, df, SS, shares,
  non-independence caveat) from latex/bel_tools/supp_varcomp.tex; build_supp.py includes it.
- Committed (not pushed): sgcv_validation_blindness.py, sgcv_variance_decomposition.py
  (--from-csv), results/tables/sgcv_arch_runs.csv + two JSONs. ~80 other revision files still
  uncommitted.
