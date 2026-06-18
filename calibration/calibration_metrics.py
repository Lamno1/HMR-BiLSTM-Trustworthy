import numpy as np

def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) for multi-class classification.
    
    probs: numpy array of shape (N, C)
    labels: numpy array of shape (N,)
    num_bins: number of bins
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    n_samples = len(labels)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if a prediction falls into the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if bin_lower == 0.0:
            in_bin = in_bin | (confidences == 0.0)
            
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def compute_mce(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    """Computes Maximum Calibration Error (MCE) for multi-class classification.
    
    probs: numpy array of shape (N, C)
    labels: numpy array of shape (N,)
    num_bins: number of bins
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    max_error = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if bin_lower == 0.0:
            in_bin = in_bin | (confidences == 0.0)
            
        if in_bin.sum() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            if error > max_error:
                max_error = error
                
    return float(max_error)

def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Negative Log Likelihood (NLL).
    
    probs: numpy array of shape (N, C)
    labels: numpy array of shape (N,)
    """
    # Clip probs to prevent log(0)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    n_samples = len(labels)
    nll = -np.sum(np.log(probs[np.arange(n_samples), labels])) / n_samples
    return float(nll)

def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Brier Score for multi-class classification.
    
    probs: numpy array of shape (N, C)
    labels: numpy array of shape (N,)
    """
    num_classes = probs.shape[1]
    y_onehot = np.eye(num_classes)[labels]
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
    return float(brier)

def compute_all_metrics(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> dict:
    """Computes ECE, MCE, NLL, Brier Score, classwise ECE (OvR), and conditional ECE."""
    return {
        "ece": compute_ece(probs, labels, num_bins=num_bins),
        "mce": compute_mce(probs, labels, num_bins=num_bins),
        "nll": compute_nll(probs, labels),
        "brier": compute_brier(probs, labels),
        "classwise_ece": compute_classwise_ece(probs, labels, num_bins=num_bins),
        "conditional_ece": compute_conditional_ece(probs, labels, num_bins=num_bins)
    }

def compute_classwise_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> dict:
    """Classwise ECE: ECE per class using one-vs-rest scheme.
    Returns dict {class_idx: ece_value}.
    Paper self-reported missing this metric.
    """
    num_classes = probs.shape[1]
    classwise = {}
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for c in range(num_classes):
        binary_conf = probs[:, c]          # P(c | x)
        binary_labels = (labels == c).astype(int)

        ece_c = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (binary_conf > bin_lower) & (binary_conf <= bin_upper)
            if bin_lower == 0.0:
                in_bin = in_bin | (binary_conf == 0.0)

            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = binary_labels[in_bin].mean()
                avg_confidence_in_bin = binary_conf[in_bin].mean()
                ece_c += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        classwise[c] = float(ece_c)
    return classwise

def compute_conditional_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15, min_bin: int = 10) -> dict:
    """Conditional ECE per class: calibration only on samples model predicted as class c (argmax==c).
    Adaptive (quantile) binning. Returns {class_idx: {"ece": float|None, "n": int}}.
    """
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out = {}

    for c in range(probs.shape[1]):
        mask = (preds == c)
        n_c = int(mask.sum())

        if n_c < min_bin:
            out[c] = {"ece": None, "n": n_c}
            continue

        conf_c = conf[mask]
        acc_c = (labels[mask] == c).astype(float)
        n_bins_c = min(num_bins, max(1, n_c // min_bin))

        edges = np.quantile(conf_c, np.linspace(0, 1, n_bins_c + 1))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9

        ece_c = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (conf_c >= lo) & (conf_c < hi)
            if in_bin.sum() == 0:
                continue
            ece_c += abs(conf_c[in_bin].mean() - acc_c[in_bin].mean()) * in_bin.mean()

        out[c] = {"ece": float(ece_c), "n": n_c}

    return out
