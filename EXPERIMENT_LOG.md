# EXPERIMENT_LOG.md

Append-only experiment knowledge base. Never overwrite a row. Each experiment:
ID / question / change / result / status / interpretation / next.
Status: KEPT / REJECTED / INCONCLUSIVE / RUNNING / PLANNED / DEFERRED.
Raw numbers live in `results/` and `docs/RARE_CLASS_PROGRESS.md` (legacy log).

---

## CROSS-EXPERIMENT PATTERNS (the reasoning, not the rows)

1. **Every architecture / loss change lost to the plain baseline** (~25 runs over
   ~1 month: DSTA-C + all gates, CAMP-Net, hierarchical head x3, aux losses,
   bigger models). Signature of "not enough signal / wrong layer", not "wrong
   model". -> bottleneck is NOT architecture.
2. **Feature additions gave modest real gains** (V1: N-vs-F transfer 0.90->0.96;
   rhythm-MLP: LOPO F1-S +0.14). -> representation/objective layer has more
   leverage than architecture, but the gains are small and S-only.
3. **`AUC_F 0.96` with `F1_F 0.24`, stable across every experiment.** The model
   RANKS F well and cannot COMMIT the argmax. -> decision-rule / measurement /
   label problem, not representation.
4. **Diminishing returns + instability**: DS1/DS2 F1-F swings 0.04<->0.45 by seed;
   "improvements" did not reduce the swing. LOPO then showed the DS1/DS2 numbers
   were a single-patient artifact. -> the evaluation was part of the problem.
5. **De-confounding (GRL) destroyed F.** The patient-signature and the F signal
   are inseparable in this data -> fusion morphology is intrinsically
   patient-specific (depends on the individual's own PVC shape).

---

## EXPERIMENTS

| ID | Question | Change | Result | Status | Interpretation |
|---|---|---|---|---|---|
| X01 | Is DSTA-C a better base than ResMemECG? | compile all runs | HMR mean macro .584+-.064; DSTA-C same-seed non-reproducible | REJECTED (DSTA base) | ResMemECG is the base; DSTA can't hold S+F together |
| X02 | Where is the F/S bottleneck? | CPU diagnostics on official ckpt | AUC_F .96 vs F1_F .24; 208-N vs 213-N AUC 1.000; V1 lifts transfer .90->.96; val = 30 F / 3 patients | KEPT (reframe) | measurement + decision-rule + confound, NOT features |
| X03 | Does pooling INCART fix cross-DB F/S? | MIT DS1 + INCART pooled | run1 FAILED (baseline wander + recipe drift); v2: S mean .33->.17, F .18->.31 but variance not reduced | REJECTED | INCART S = different domain; pooling as-is hurts S |
| X04 | Does a soft hierarchical decision head help? | flat -> hier head, 3 calibration variants | hier F .17/S .26; hier_la F .02; hier_ba F .10/S .22 - all worse | REJECTED (CLOSED) | decision head alone is not a big enough lever; double imbalance correction |
| X05 | Build a patient-honest evaluation | record-level 5-fold grouped CV + sentinel (`train_lopo_cv.py`, `docs/LOPO_PROTOCOL.md`) | harness built, sentinel fold defined | KEPT | the fixed DS1/DS2 F1-F/F1-S can never be "stable" - test-F is one patient |
| X06 | Sentinel screen of MultiPath rungs | flag combos (v1 / rhythm / grl / balanced) | R1b (+V1 residual) PASS; R2 (+rhythm) PASS big (213-F .24->.56, 232-S .03->.27); R3 (+GRL) F craters to .09; R3b (+balanced) trades F+N for S | KEPT: R2. REJECTED: GRL, balanced-as-stack | GRL scrubs F morphology; balanced trades classes |
| X07 | Does R2 beat flat on the HONEST LOPO (5 fold x 2 seed)? | full LOPO, R2 vs flat, best-epoch | F1-F 0.07+-0.09 (folds 2/3/4 ~0); F1-S ~0.38, seed-stable, fold 0.14-0.53. Paired dS_fold_mean +0.14 (folds 1 & 4 large, consistent across seeds); dF_fold_mean +0.017 (fold 0 flips sign by seed) | KEPT (gate 1: R2 wins S). F: refer-only | DS1/DS2 0.24 was a single-patient artifact; R2's V1 branch does NOT recover F on held-out patients |
| X08 | Can a calibrated selective layer make S auto-decidable? | `selective/run_p2.py` first cut (per-class temp calibration + Mondrian-S conformal + F frontier) | useful S auto-decision fold 4 only (auto_cov_S .24-.30 @ prec .93-.98); every other fold auto_cov_S ~0 (never an S singleton) | INCONCLUSIVE / gate 2 NOT passed | "effect on >1 fold" fails; need E1 to know if it's ranking or calibration |
| X09 | D1: R2 on official DS1/DS2, 3 seeds (literature table) | `validation/run_d1.py` | ~3.2 h/epoch observed (isolated 20-batch profile: ~8.1 min/epoch). Deliberately stopped at 9/45 epochs seed 42; no per-epoch checkpoint -> lost | REJECTED (this run) / PLANNED (rerun) | NO algorithmic bug in the train step; 24x wall-time gap UNPROVEN; rerun needs checkpoint + timing instrumentation |
| E1 | S: ranking vs calibration failure? F: label-ambiguous vs decision-rule errors? | analysis on `*_preds.npz` | S-AUC 0.73/0.98/0.95/0.95/0.82 (3/5>=0.85); F->N errors 48.7% in bottom-3 P(F) deciles, 4.5% top-3; F AP 0.001-0.05 despite AUC 0.79-0.86; fold-4 S = 55% patient 232 | DONE | RULED OUT: F-errors-as-argmax; S-as-global-calibration. F "separability" was an AUC illusion. |
| E2 | MIT-BIH F label ceiling? | literature + rule-vs-label proxy | no published F-specific inter-annotator number (MIT-BIH ships consensus only); rule proxy F1 0.051 kappa 0.022; same rule kappa -0.088 on 208 vs +0.169 on 213 | DONE | NO numerical ceiling derivable ("0.4" forbidden in paper). F is patient-specific at the signal level. |
| E5-A | Does S fold-variance collapse when conditioned on original sub-type? | join `data/raw/mitdb/*.atr` sub-symbols | MIT-S ~90%+ subtype A; A-only cross-fold AUC spread 0.31 (worse than aggregate 0.25); INCART-A spread 0.07 | DONE - H-S3 REJECTED | S fold-0/4 failure is patient-level, not subtype. Model CAN rank PAC (INCART, folds 1-3). |
| E5-B | Is F prediction patient-relative N/V geometry? Would patient-adaptation help? | geometry vs per-record N/V centroids (logits + pre-head embedding) | F near patient N -> pred N (94-99%); intermediate -> pred F (55-64%); robust across metrics/spaces and on 208 & 213. isolation<->perf rho 0.12-0.23, p 0.1-0.3, sign-unstable | DONE - H-F4 mechanism SUPPORTED, actionable REJECTED | ~1/3 of F geometrically N-coincident. No basis for E6. |
| P2-final | Does a pre-registered risk-controlled selective layer reach a useful S operating point? | `selective/run_p2_final.py` - continuous alpha sweep, patient-cluster bootstrap, MIT/INCART separate, F->referral | criterion (risk<=0.02 & auto-S cov>=0.15 & auto-S prec>=0.90) met by 0/3 reliable folds. fold2: prec 0.96/cov 0.026; fold3: cov 0.38/prec 0.30; fold1: no safe-zone. flat also 0/3. temperature calibration NOT the lever. | DONE - **NO-GO** | S selective automation collapses. Paper = LOPO evaluation-methodology + honest negative + F referral. |
| C1 | Does the flat model retain its DS1/DS2 rare-class result under controlled MIT-only subject-grouped CV? | 44 records/43 subject groups; bind 201+202; 5 folds x seeds 42/123; 45-epoch official recipe; train-N=37,208; no INCART | fold-first mean F1-F 0.0109, F1-S 0.2316; F1-S fold range 0.039-0.696; integrity recomputation PASS | DONE | Strengthens protocol/subject-composition sensitivity. Not a paired causal decomposition; replaces 0.053 as the primary flat SG-CV estimate while legacy RG-CV remains the R2/P2 protocol. |

**EXPERIMENTAL PHASE CLOSED 2026-09-05.** E3 (F-record scaling) and E4 (sequence
model) NOT run - evidence converged before they became worth the compute.
E6 (patient-adaptive F) NOT justified (E5-B). See RESEARCH_STATE FINAL VERDICT.

---

## FAILED HYPOTHESES (branch pruned)

- "Architecture / gate mechanism / bigger model solves F/S" - X01, X04, X06, plus
  ~15 legacy runs in `docs/RARE_CLASS_PROGRESS.md` section 5. Pruned.
- "Adversarial de-confounding recovers transferable F" - X06 (GRL). Pruned.
- "Cross-DB pooling as-is fixes S" - X03. Pruned.
- "The DS1/DS2 F1-F ~0.24 reflects patient generalization" - X07 (LOPO). Pruned.

## INCONCLUSIVE (needs a better experiment)

- X08: is S's fold-4-only selective utility a calibration artefact (fixable) or a
  real fold-generalization limit? -> E1.
- X09: real cause of D1 slowdown -> instrumented rerun (deferred).
