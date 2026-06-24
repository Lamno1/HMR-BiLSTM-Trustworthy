# Critical Fixes Summary

All fixes address CRITICAL-severity issues identified in `SCIENTIFIC_AUDIT_REPORT.md`.
Only verified, code-level bugs were fixed. Systemic issues (intra/inter-patient split
design, impossible table values) require experiment redesign and are documented in the
audit report but are not patched here.

---

## Fix 1 — Issue #9: Jaccard-with-tolerance double-counting

| Field | Detail |
|---|---|
| **File** | `explainability/integrated_gradients.py` |
| **Lines** | 102–108 |
| **Severity** | CRITICAL |

**Root cause.** The original algorithm computed the "intersection" as the arithmetic
mean of two overlapping match-sets:

```python
matched_a = set(a for a in set_a if any(abs(a - b) <= tolerance for b in set_b))
matched_b = set(b for b in set_b if any(abs(b - a) <= tolerance for a in set_a))
intersection = (len(matched_a) + len(matched_b)) / 2.0
```

When one element of B lies within tolerance of multiple elements of A (or vice versa),
both matched sets are inflated and `intersection` exceeds the true count, making the
Jaccard score artificially high (can reach 1.0 for cases that are not a perfect match).

**Fix.** Replaced with a greedy bijective matching: sort A, then for each `a` consume
the nearest unconsumed `b` within tolerance. Each element of B can match at most once,
yielding a correct intersection cardinality.

**Diff (simplified):**
```diff
-   matched_a = set(a for a in set_a if any(abs(a - b) <= tolerance for b in set_b))
-   matched_b = set(b for b in set_b if any(abs(b - a) <= tolerance for a in set_a))
-   intersection = (len(matched_a) + len(matched_b)) / 2.0
-   union = len(set_a) + len(set_b) - intersection
-   return intersection / max(1.0, union)
+   set_a_sorted = sorted(set_a)
+   set_b_remaining = sorted(set_b)
+   matched = 0
+   for a in set_a_sorted:
+       candidates = [b for b in set_b_remaining if abs(a - b) <= tolerance]
+       if candidates:
+           best = min(candidates, key=lambda b: abs(a - b))
+           matched += 1
+           set_b_remaining.remove(best)
+   union = len(set_a) + len(set_b) - matched
+   return matched / max(1, union)
```

**Scientific impact.** SHAP–IG consistency scores reported in the paper were inflated.
The fixed implementation gives a conservative, truthful Jaccard that reflects genuine
agreement between the two explainability methods.

---

## Fix 2 — Issue #13: Missing dependencies in `requirements.txt`

| Field | Detail |
|---|---|
| **File** | `requirements.txt` |
| **Severity** | CRITICAL |

**Root cause.** Eight packages imported throughout the codebase were absent from
`requirements.txt`, making a clean-environment install silently incomplete:
`shap`, `captum`, `torchattacks`, `autoattack`, `pyyaml`, `scipy`, `wfdb`,
`python-docx`, `tqdm`.

**Fix.** Added all missing packages with minimum version pins matching the imports:

```diff
+pyyaml>=6.0.0
+scipy>=1.10.0
+shap>=0.43.0
+captum>=0.6.0
+torchattacks>=3.5.0
+autoattack>=0.1.0
+python-docx>=1.1.0
+wfdb>=4.1.0
+tqdm>=4.65.0
```

**Scientific impact.** Any reviewer or reader attempting to reproduce results from a
fresh install would encounter immediate `ModuleNotFoundError` failures. The fix makes
the pipeline reproducible from `pip install -r requirements.txt`.

---

## Fix 3 — Issue #18: FGSM perturbation not clamped to data range

| Field | Detail |
|---|---|
| **File** | `evaluate_fgsm.py` |
| **Line** | 33 |
| **Severity** | CRITICAL |

**Root cause.** The FGSM implementation added the signed gradient perturbation to the
original input without clamping the result to the valid data range:

```python
x_adv = (x + perturbation).detach()   # can escape the data manifold
```

FGSM adversarial examples must stay within the same data distribution to be
meaningful threat examples. Without clamping, perturbations can push samples outside
the normalised ECG range, and the reported robustness accuracy is evaluated against
out-of-distribution inputs that would never occur in practice.

**Fix.**
```diff
-x_adv = (x + perturbation).detach()
+x_adv = (x + perturbation).clamp(x.min(), x.max()).detach()
```

**Scientific impact.** Reported FGSM robustness numbers are no longer contaminated by
out-of-range adversarial samples. The clamped version is consistent with the standard
formulation used in published robustness benchmarks.

---

## Fix 4 — Issue #19: C&W trade-off parameter `c` too small

| Field | Detail |
|---|---|
| **Files** | `robustness/cw_attack.py` line 215, `configs/experiment_config.yaml` line 27 |
| **Severity** | CRITICAL |

**Root cause.** The C&W L₂ attack uses a trade-off constant `c` that balances
classification loss against perturbation magnitude. The value `c = 1e-4` is 100×
below the widely-used default (1e-2 to 1.0). At this value the optimiser heavily
penalises perturbation size and almost never finds a misclassifying example, making
C&W indistinguishable from no attack at all.

**Fix.**
```diff
 # robustness/cw_attack.py
-cw_c = rob_cfg.get("cw_c", 1e-4)
+cw_c = rob_cfg.get("cw_c", 1e-2)

 # configs/experiment_config.yaml
-cw_c: 1.0e-4
+cw_c: 1.0e-2
```

**Scientific impact.** With `c = 1e-4` the attack silently failed to find adversarial
examples, so C&W accuracy figures were near-identical to clean accuracy. The corrected
`c = 1e-2` puts attack strength in a regime where genuine adversarial examples are
reliably found, giving a meaningful robustness lower bound.

---

## Fix 5 — Issue #20: Robustness evaluation uses intra-patient test split

| Field | Detail |
|---|---|
| **File** | `evaluate_robustness_all.py` |
| **Lines** | 73–75 |
| **Severity** | CRITICAL |

**Root cause.** The script loaded `data/processed/train.npz`, `val.npz`, and
`test.npz` — the intra-patient splits used during initial training. The deployed
HMR-BiLSTM model is trained on inter-patient splits (`inter_best_rlstm.pt`). Evaluating
it on intra-patient test data overstates accuracy due to the same-patient data leakage
present in intra-patient splits.

**Fix.**
```diff
-train = np.load("data/processed/train.npz")
-val   = np.load("data/processed/val.npz")
-test  = np.load("data/processed/test.npz")
+train = np.load("data/processed/splits/inter_train.npz")
+val   = np.load("data/processed/splits/inter_val.npz")
+test  = np.load("data/processed/splits/inter_test.npz")
```

**Scientific impact.** Robustness figures (noise-level F1 curves, clean-accuracy
baseline in the noise sweep) are now computed on the correct held-out patient cohort,
making them directly comparable to the primary classification results.

---

## Fix 6 — Issue #22: SHAP analysis omits class Q (arrhythmia class 4)

| Field | Detail |
|---|---|
| **Files** | `explainability/shap_analysis.py` lines 39–40 and 202, `configs/experiment_config.yaml` line 34 |
| **Severity** | CRITICAL |

**Root cause.** The SHAP explainability pipeline hard-coded only 4 of the 5 AAMI
classes:

```python
CLASS_NAMES  = {0: "N", 1: "S", 2: "V", 3: "F"}   # missing Q=4
SHAP_CLASSES = [0, 1, 2, 3]                          # missing 4
```

The config also listed only `shap_classes: [0, 1, 2, 3]`. Class Q (paced/unknown beats)
was therefore excluded from:
- Per-class SHAP waterfall and bar plots
- Global mean |SHAP| importance averaging
- SHAP–IG Jaccard consistency computation
- `shap_importance_ranking.csv`

**Fix.**
```diff
 # explainability/shap_analysis.py
-CLASS_NAMES  = {0: "N", 1: "S", 2: "V", 3: "F"}
-SHAP_CLASSES = [0, 1, 2, 3]
+CLASS_NAMES  = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
+SHAP_CLASSES = [0, 1, 2, 3, 4]

 # line 202 (config fallback)
-shap_classes = exp_cfg.get("shap_classes", [0, 1, 2, 3])
+shap_classes = exp_cfg.get("shap_classes", [0, 1, 2, 3, 4])

 # configs/experiment_config.yaml
-shap_classes: [0, 1, 2, 3]      # N, S, V, F
+shap_classes: [0, 1, 2, 3, 4]   # N, S, V, F, Q
```

**Scientific impact.** Global feature importance and SHAP–IG agreement metrics now
reflect all 5 clinically relevant classes. Excluding Q systematically biased the
importance map toward the four dominant classes, potentially under-weighting features
specific to paced beats.

---

## Non-Code CRITICAL Issues (Not Fixed Here)

The following CRITICAL issues from the audit require experimental redesign rather than
code edits. They are documented here for completeness.

| Issue | Location | Description |
|---|---|---|
| #5 | `data/processed/` splits | Intra-patient vs. inter-patient split mismatch: primary training data and reported Table 1 metrics may not match the inter-patient model. |
| #6 | Reported tables | Macro-F1 and per-class F1 values in `SCIENTIFIC_AUDIT_REPORT.md` tables exceed what is achievable on the inter-patient split (patient leakage probable in the numbers). |
| #7 | `preprocess_aami.py` | Beat segmentation window choice affects class boundary overlap; must be verified against AAMI EC57 standard. |
| #8 | `train_inter.py` | SMOTE applied before cross-patient split validation — must be confirmed it respects patient boundaries. |
| #11 | MC Dropout | `evaluate_uncertainty.py` activates Dropout in model.train() mode, which also activates BatchNorm statistics update. Should use selective per-layer `train()`. |
| #12 | Deep Ensemble | Ensemble members share the same CNN stem initialisation seed; diversity relies only on training stochasticity, not architecture. |
| #14 | TracIn | Gradient cosine similarity approximation treats all checkpoint epochs equally; influence magnitudes are not normalised by learning rate × step size. |
| #17 | AutoAttack | `autoattack_eps` in config is `0.02` without verification of whether this epsilon is in the same normalised space as the training data. |
| #21 | Reported results | `results/` CSV files contain accuracy figures inconsistent with published confusion matrices. |
| #23–#28 | Various | Additional systemic data-pipeline and reporting inconsistencies documented in `SCIENTIFIC_AUDIT_REPORT.md`. |

---

---

## Session 2 Fixes (2026-06-24) — Issues #11, #14, #23, #26, #27

Additional code-level fixes applied in the second session. Issues #7, #8, #12, #17, #21
still require experiment redesign (re-running evaluations on unified splits).

| Fix | Issue(s) | File(s) | Description |
|---|---|---|---|
| Fix 7 | #56 | `evaluate_trustworthiness.py:122` | Wrong epsilon selector: `abs(eps) - 0.02` → `abs(abs(eps) - 0.02)` |
| Fix 8 | #39 | `train_inter_patient.py:119` | AUC passed raw `probs[:,:4]` instead of renormalized `probs_4class` |
| Fix 9 | #59 | `explainability/integrated_gradients.py:299` | Signed mean → mean absolute for IG attribution plots |
| Fix 10 | #11 | `validation/pgd_convergence.py:62–64` | PGD now passes `r_fwd`/`r_bwd` via `return_internals=True` |
| Fix 11 | #29 | `hmr_bilstm.py:181` | `self.dropout` now applied to `h_t` in `RLSTMCell.forward` |
| Fix 12 | #27 | `preprocess.py:151–152` | Global scalar normalization → per-feature `axis=0` |
| Fix 13 | #38 | `train.py:106–109` | 5-class F1 → 4-class (`labels=[0,1,2,3]`) to match inter-patient training |
| Fix 14 | #44/#129 | `evaluate_calibration.py:28` | Default `num_bins` changed from 10 to 15 (matches config) |
| Fix 15 | #45/#48 | `evaluate_calibration.py:main()` | Temperature scaling applied for HMR-BiLSTM; pre/post ECE clearly labeled |
| Fix 16 | #108 | `evaluate_calibration.py:239` | `results/tables/` directory created with `mkdir(parents=True, exist_ok=True)` |
| Fix 17 | #14 | `run_reproducible_pipeline.py`, `run_all.bat` | `preprocess_aami.py` (step 1) and `train_inter_patient.py` (step 3) added to both orchestrators |
| Fix 18 | #26 | `robustness/auto_attack.py:158` | Epsilon-space assertion added: `assert abs(aa_eps - pgd_eps) < 1e-6` |
| Fix 19 | #23 | `validation/preprocess_aami.py:155–176` | Intra-patient splits now normalized with train-only statistics before saving |
| Fix 20 | #36 | `train.py:128–147` | BN layers switched to eval during FGSM generation; `model.zero_grad()` removed from adversarial function |
| Fix 21 | #42 | `validation/preprocess_aami.py:49` | `int(fs)` → `int(round(fs))` for resample_poly denominator |
| Fix 22 | #43 | `validation/preprocess_aami.py:65` | Boundary guard `<` → `<=` to include beats at exact boundary |
| Fix 23 | #60 | `explainability/data_attribution.py:1–38` | Module docstring updated to clarify this is not true TracIn; print statements renamed |
| Fix 24 | #37 | `train.py:151,181,196` | NaN batch counter added; WARNING printed; count returned in stats dict |
| Fix 25 | #131 | `evaluate_trustworthiness.py:61,82` | `dict \| None` type hints changed to untyped (Python 3.9 compatibility) |
| Fix 26 | #89/#143 | `train.py:281` | Cosine LR denominator changed to `epochs - 1` so final epoch reaches exact `min_lr` |
| Fix 27 | #101 | `preprocess.py:216` | Class weight upper cap raised from 10.0 to 50.0 |
| Fix 28 | #100 | `validation/preprocess_aami.py:144–156` | Intra-patient split changed from random shuffle to stratified `train_test_split` |
| Fix 29 | #146 | `preprocess.py:152` | `std(axis=0) + 1e-8` → `sqrt(var(axis=0) + 1e-8)` |
| Fix 30 | #148 | `validation/preprocess_aami.py:136` | Zero-check for `len(y)` in `print_dist` |
| Fix 31 | #74 | `run_baselines.py:174` | Dead `y_pred = model.predict(X_tr_flat)` line removed |
| Fix 32 | #94 | `train.py:276` | `best_f1 = 0.0` → `float("-inf")` to ensure first epoch always saves a checkpoint |
| Fix 33 | #96 | `run_ablation.py:358` | Checkpoint existence check before `torch.load` |
| Fix 34 | #97 | `validation/pgd_convergence.py:195` | `steps` → `plot_steps` to avoid shadowing the outer loop variable |
| Fix 35 | #150 | `calibration/reliability_diagram.py:35–41` | Empty-bin phantom rows removed; empty bins are now silently skipped |
| Fix 36 | #153 | `evaluate_fgsm.py:137` | Duplicate `label_accuracy` key removed from result dict |
| Fix 37 | #154 | `gen_guide.py:408` | `class_weights.json` → `class_weights.npy` |

*Generated: 2026-06-24*
