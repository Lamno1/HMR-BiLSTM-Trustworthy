# Calibration Analysis Report
**Generated:** 2026-06-10T20:14:09.010373
**Checkpoint Hash:** sha1_dbab046b958c40af1d6504b95468e8d7070237c5
**Run ID:** v1.0_20260610_200745

## Checkpoint Verification
- Test Accuracy: 0.8988
- Test F1 (macro): 0.3993
  *(Verify these values match model training logs)*

## Calibration Metrics (Before → After Temperature Scaling)

### Expected Calibration Error (ECE)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| ECE (lower is better) | 0.0300 | 0.0160 | **-0.0140** |

### Maximum Calibration Error (MCE)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| MCE (lower is better) | 0.0654 | 0.0915 | **+0.0261** |

### Negative Log Likelihood (NLL)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| NLL (lower is better) | 0.3111 | 0.3017 | **-0.0094** |

### Brier Score
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Brier (lower is better) | 0.1581 | 0.1576 | **-0.0005** |

## Summary
**Optimal Temperature:** 0.8469

### Interpretation
- **ECE improvement** ✓: Temperature scaling successfully reduced ECE from 0.0300 to 0.0160
- **MCE increase** ⚠: MCE increased from 0.0654 to 0.0915. Temperature scaling optimizes NLL/ECE but may degrade worst-case calibration.
- **Brier stability** ✓: Brier score improved slightly from 0.1581 to 0.1576

### Per-Class Conditional ECE (After Scaling)
**Conditional ECE** measures calibration only on samples the model actually predicted for each class.

| Class | Conditional ECE | N Predicted | Interpretation |
|-------|-----------------|-------------|-----------------|
| N (C0) | 0.0410 | 43386 | Well-calibrated |
| S (C1) | 0.5408 | 1371 | Poorly calibrated |
| V (C2) | 0.0658 | 3415 | Well-calibrated |
| F (C3) | 0.6667 | 1496 | Poorly calibrated |
| Q (C4) | None | 0 | Too few predictions (< min_bin), insufficient data |

## Recommendation for Paper
- Report **all three metrics** (ECE, MCE, Brier) to provide transparent view
- Note: Temperature scaling optimizes NLL/ECE but trades off MCE
- Use Conditional ECE per-class for detailed calibration narrative
- Class 4 (Q) may not benefit from temperature scaling if rarely predicted