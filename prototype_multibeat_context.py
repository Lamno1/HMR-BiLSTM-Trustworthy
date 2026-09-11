"""
Prototype: does adding neighboring-beat context help Fusion-class F1?

Motivation: Fusion (F) beats are, by AAMI definition, a morphological blend
of a Normal and a Ventricular wavefront -- the signal that distinguishes a
Fusion beat from its neighbors lives partly in the *relationship* to the
surrounding beats, not only in the isolated 187-sample window the current
pipeline feeds the model. Recent literature (DeepArrhythmia, arXiv:2605.16441,
2026) reports a large Fusion-F1 gain from incorporating multi-beat rhythm
context instead of classifying beats in isolation.

This script builds a SINGLE-VARIABLE A/B test on top of the paper's own
already-extracted beat pool (data/processed/splits/all_extracted_beats.npz,
same DS1/DS2 patient-disjoint protocol, same RLSTMClassifier, same loss,
same hyperparameters, same epoch budget) -- the ONLY thing that changes is
the input:

  control   ("single"):  X = [curr_beat]                       (187 samples)
  treatment ("context"):  X = [prev_beat, curr_beat, next_beat]  (561 samples)

Edge beats (first/last beat of a record) repeat the current beat as the
missing neighbor rather than zero-padding, to avoid an artificial discontinuity.

This is a QUICK PROTOTYPE (moderate epoch budget, no adversarial training,
no cosine schedule) meant to answer one question -- does F1(Fusion) move at
all under identical training conditions -- not to produce paper-final numbers.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

from hmr_bilstm import RLSTMClassifier, RLSTMLoss

BEATS_PATH = "data/processed/splits/all_extracted_beats.npz"
NUM_CLASSES = 5
CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# Same DS1/DS2 patient-disjoint protocol as the paper (Experimental Protocol section).
DS1 = {101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230}
DS1_VAL_HOLDOUT = {122, 124, 205, 223}
DS1_TRAIN = DS1 - DS1_VAL_HOLDOUT
DS2 = {100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234}

EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 128
SEED = 42


def load_beat_pool():
    """Load the pooled beat set, grouped by patient, in original (chronological) order."""
    data = np.load(BEATS_PATH, allow_pickle=True)["data"]
    by_patient = {}
    for entry in data:
        pid = int(entry["patient"])
        by_patient.setdefault(pid, []).append(entry)
    return by_patient


def build_dataset(by_patient, patient_ids, mode):
    """
    mode='single'  -> X: (N, 187)   just the current beat
    mode='context' -> X: (N, 561)   [prev, curr, next] concatenated
    """
    X, y = [], []
    for pid in sorted(patient_ids):
        beats = by_patient[pid]
        n = len(beats)
        for i in range(n):
            curr = np.asarray(beats[i]["x"], dtype=np.float32)
            if mode == "single":
                X.append(curr)
            else:
                prev = np.asarray(beats[i - 1]["x"], dtype=np.float32) if i > 0 else curr
                nxt = np.asarray(beats[i + 1]["x"], dtype=np.float32) if i < n - 1 else curr
                X.append(np.concatenate([prev, curr, nxt]))
            y.append(int(beats[i]["y"]))
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def normalize(X_train, *others):
    mean = X_train.mean(axis=0)
    std = np.sqrt(X_train.var(axis=0) + 1e-8)
    out = [(X_train - mean) / std]
    for X in others:
        out.append((X - mean) / std)
    return out


def class_weights_from(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


def train_and_eval(name, X_train, y_train, X_val, y_val, X_test, y_test, device, batch_size=BATCH_SIZE):
    print(f"\n{'='*70}\n[{name}]  X shape: train={X_train.shape} val={X_val.shape} test={X_test.shape}  batch_size={batch_size}\n{'='*70}")

    torch.manual_seed(SEED)
    cw = class_weights_from(y_train).to(device)

    model = RLSTMClassifier(
        input_size=1, hidden_size=96, dropout=0.25,
        num_classes=NUM_CLASSES, cnn_out_channels=64, num_layers=2,
    ).to(device)
    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw, use_focal=True, focal_gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def to_tensor(X):
        return torch.from_numpy(X).unsqueeze(-1)  # (N, T) -> (N, T, 1)

    # Keep val/test on CPU and move per-eval-chunk to GPU -- with T=561 (context
    # mode) the full val/test set resident on GPU at once was contributing to the
    # OOM alongside the training batches; evaluation isn't time-critical here.
    Xtr_t, ytr_t = to_tensor(X_train), torch.from_numpy(y_train)
    Xva_t, yva_t = to_tensor(X_val), torch.from_numpy(y_val)
    Xte_t, yte_t = to_tensor(X_test), torch.from_numpy(y_test)

    def eval_in_chunks(X_cpu, chunk=batch_size):
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_cpu), chunk):
                xb = X_cpu[i:i + chunk].to(device)
                preds.append(model(xb).argmax(-1).cpu())
        return torch.cat(preds).numpy()

    n_train = len(Xtr_t)
    best_f1, best_state, patience_cnt = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            logits, internals = model(xb, return_internals=True)
            loss, _ = criterion(logits, yb, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            del xb, yb, logits, internals, loss

        model.eval()
        val_preds = eval_in_chunks(Xva_t)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    test_preds = eval_in_chunks(Xte_t)

    y_true = yte_t.numpy()
    acc = accuracy_score(y_true, test_preds)
    f1_macro = f1_score(y_true, test_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, test_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)

    print(f"\n  [{name}] best val F1_macro={best_f1:.4f}  train_time={train_time:.1f}s")
    print(f"  [{name}] TEST accuracy={acc:.4f}  F1_macro={f1_macro:.4f}")
    print(f"  [{name}] TEST per-class F1: " +
          "  ".join(f"{CLASS_NAMES[i]}={f1_per_class[i]:.4f}" for i in range(4)))
    print(classification_report(y_true, test_preds, labels=[0, 1, 2, 3],
                                 target_names=["N", "S", "V", "F"], zero_division=0, digits=4))

    return {"accuracy": acc, "f1_macro": f1_macro,
            "f1_per_class": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(4)},
            "train_time_sec": train_time}



# T=561 (context) needs roughly 3x the per-sample activation memory of T=187
# (single) for the same batch -- this GPU is a shared 4GB card that already
# OOM'd once transitioning into context mode at batch_size=128.
BATCH_SIZE_BY_MODE = {"single": BATCH_SIZE, "context": 32}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading beat pool...")
    by_patient = load_beat_pool()
    print(f"  {len(by_patient)} records loaded")

    results_path = Path("results/logs/multibeat_context_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode, name in [("single", "Single-beat (control, 187 samples)"),
                        ("context", "Multi-beat context (treatment, 561 samples)")]:
        print(f"\nBuilding {mode} dataset...")
        X_train, y_train = build_dataset(by_patient, DS1_TRAIN, mode)
        X_val, y_val = build_dataset(by_patient, DS1_VAL_HOLDOUT, mode)
        X_test, y_test = build_dataset(by_patient, DS2, mode)
        X_train, X_val, X_test = normalize(X_train, X_val, X_test)

        results[mode] = train_and_eval(name, X_train, y_train, X_val, y_val,
                                        X_test, y_test, device,
                                        batch_size=BATCH_SIZE_BY_MODE[mode])

        # Persist after each mode so a crash in "context" doesn't also lose
        # the already-finished "single" result (happened once already).
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  [OK] partial results -> {results_path}")

        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(" SUMMARY: single-beat control vs. multi-beat context treatment ")
    print("=" * 70)
    print(f"{'Metric':<20} {'Single (control)':>20} {'Context (treatment)':>22} {'Delta':>10}")
    for key, label in [("accuracy", "Accuracy"), ("f1_macro", "F1_macro")]:
        a, b = results["single"][key], results["context"][key]
        print(f"{label:<20} {a:>20.4f} {b:>22.4f} {b - a:>+10.4f}")
    for cls in ["N", "S", "V", "F"]:
        a = results["single"]["f1_per_class"][cls]
        b = results["context"]["f1_per_class"][cls]
        print(f"{'F1-' + cls:<20} {a:>20.4f} {b:>22.4f} {b - a:>+10.4f}")


if __name__ == "__main__":
    main()
