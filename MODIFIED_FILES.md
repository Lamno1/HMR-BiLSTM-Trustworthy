# Modified Files

Files changed during the CRITICAL fix pass. All other files in the repository were
read-only during this session. See `CRITICAL_FIXES_SUMMARY.md` for diffs and
scientific rationale for each change.

---

## `explainability/integrated_gradients.py`
**Fix 1 — Issue #9: Jaccard-with-tolerance double-counting**
- Lines 102–122: replaced symmetric double-set intersection with a greedy bijective
  matching algorithm. Each element of B can be consumed by at most one element of A,
  preventing the previous formula from returning inflated Jaccard scores.

---

## `requirements.txt`
**Fix 2 — Issue #13: Missing runtime dependencies**
- Added 9 packages that are imported throughout the codebase but were absent from the
  install manifest: `pyyaml`, `scipy`, `shap`, `captum`, `torchattacks`, `autoattack`,
  `python-docx`, `wfdb`, `tqdm`.

---

## `evaluate_fgsm.py`
**Fix 3 — Issue #18: FGSM perturbation not clamped**
- Line 33: added `.clamp(x.min(), x.max())` so the adversarial example stays within
  the normalised data range. Prevents evaluation on out-of-distribution inputs.

---

## `robustness/cw_attack.py`
**Fix 4 — Issue #19: C&W trade-off constant too small**
- Line 215: changed the in-code default for `cw_c` from `1e-4` to `1e-2` so that
  when the config value is absent the attack uses a sensible strength.

---

## `evaluate_robustness_all.py`
**Fix 5 — Issue #20: Robustness evaluation on wrong (intra-patient) test split**
- Lines 73–75: changed all three `np.load()` calls to use the inter-patient split
  files under `data/processed/splits/` instead of the intra-patient files under
  `data/processed/`.

---

## `explainability/shap_analysis.py`
**Fix 6 — Issue #22: SHAP analysis omits class Q (label 4)**
- Line 39: added `4: "Q"` to `CLASS_NAMES`.
- Line 40: added `4` to the `SHAP_CLASSES` constant.
- Line 202: updated the config-fallback list from `[0, 1, 2, 3]` to `[0, 1, 2, 3, 4]`.

---

## `configs/experiment_config.yaml`
**Fix 4 + Fix 6 — config values for cw_c and shap_classes**
- Line 27: `cw_c: 1.0e-4` → `cw_c: 1.0e-2` (matches Fix 4 in `cw_attack.py`).
- Line 34: `shap_classes: [0, 1, 2, 3]` → `shap_classes: [0, 1, 2, 3, 4]` (matches Fix 6 in `shap_analysis.py`).

---

## Files Read But Not Modified

The following files were inspected to verify or rule out audit findings. No changes
were made to them.

| File | Audit Verdict |
|---|---|
| `hmr_bilstm.py` | Issues #1/#24 are FALSE POSITIVES — file is syntactically correct |
| `hmr_bilstm_ablation.py` | Issues #2/#3 are FALSE POSITIVES — file is syntactically correct |
| `calibration/temperature_scaling.py` | Issue #4 is FALSE POSITIVE — no IndentationError |
| `evaluate_calibration.py` | Issue #10 is FALSE POSITIVE — no IndentationError |
| `configs/paths.py` | Read to obtain `INTER_TEST` path constant for Fix 5 |

---

*Generated: 2026-06-23*
