"""Leakage-free, denominator-explicit metrics for selective ECG prediction."""
from __future__ import annotations

import numpy as np


def accept_mask(scores: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Accept samples whose uncertainty score is at most the threshold.

    ``threshold`` may be scalar or a length-K vector indexed by predicted class.
    Lower scores must always mean more confidence.
    """
    scores = np.asarray(scores)
    return scores <= threshold


def selective_metrics(y_true, y_pred, accepted, classes=(1, 3)) -> dict:
    """Return class-specific selective metrics with their exact denominators.

    ``auto_coverage[c]`` = P(accept | Y=c).
    ``auto_recall[c]`` = P(accept and prediction=c | Y=c).
    ``auto_precision[c]`` = P(Y=c | accept and prediction=c).
    The last is the operational quantity for an auto-decision labelled c; it is
    deliberately not interchangeable with coverage or recall.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accepted = np.asarray(accepted, dtype=bool)
    if not (len(y_true) == len(y_pred) == len(accepted)):
        raise ValueError("y_true, y_pred, and accepted must have equal length")

    out = {"n": int(len(y_true)), "accepted_n": int(accepted.sum()),
           "selective_risk": float(np.mean(y_true[accepted] != y_pred[accepted])) if accepted.any() else np.nan,
           "auto_coverage": {}, "auto_recall": {}, "auto_precision": {},
           "referral_rate_by_true_class": {}, "denominators": {}}
    for c in classes:
        truth = y_true == c
        predicted_and_accepted = accepted & (y_pred == c)
        n_truth = int(truth.sum())
        n_predicted_and_accepted = int(predicted_and_accepted.sum())
        out["denominators"][int(c)] = {
            "true_class_n": n_truth,
            "accepted_predicted_class_n": n_predicted_and_accepted,
        }
        out["auto_coverage"][int(c)] = float(accepted[truth].mean()) if n_truth else np.nan
        out["auto_recall"][int(c)] = float((accepted & (y_pred == c) & truth).sum() / n_truth) if n_truth else np.nan
        out["auto_precision"][int(c)] = (
            float((truth & predicted_and_accepted).sum() / n_predicted_and_accepted)
            if n_predicted_and_accepted else np.nan
        )
        out["referral_rate_by_true_class"][int(c)] = float((~accepted)[truth].mean()) if n_truth else np.nan
    return out


def coverage_at_precision(y_true, y_pred, score, target_precision, class_id, thresholds=None):
    """Best auto-coverage subject to auto-precision for one predicted class.

    This is a *descriptive* sweep.  Threshold selection must be performed on a
    tuning split, never on the returned outer-test data.
    """
    score = np.asarray(score)
    if thresholds is None:
        thresholds = np.unique(score)
    candidates = []
    for threshold in np.asarray(thresholds):
        metrics = selective_metrics(y_true, y_pred, accept_mask(score, threshold), (class_id,))
        precision = metrics["auto_precision"][class_id]
        coverage = metrics["auto_coverage"][class_id]
        if not np.isnan(precision) and precision >= target_precision:
            candidates.append((float(coverage), float(threshold), metrics))
    return max(candidates, key=lambda item: item[0]) if candidates else None
