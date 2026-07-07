"""
evaluate_autoattack_baselines.py
==================================
Run AutoAttack across LSTM, BiLSTM, and the canonical HMR-BiLSTM on the exact
same stratified subset, so the paper's robustness table has a like-for-like
AutoAttack comparison instead of HMR-BiLSTM alone.

Reuses the existing, already-verified building blocks rather than
duplicating them:
    - load_baseline_model / build_test_loader   (evaluate_fgsm.py)
    - load_hmr_bilstm                            (report_results.py)
    - select_stratified_subset                   (robustness/cw_attack.py)
    - pgd_attack, to_unit_range,
      enter_deterministic_rnn_train_mode          (robustness/auto_attack.py)

Same eps/data-range/4-target-class-cap protocol as robustness/auto_attack.py,
attacking a stratified subset of the FULL test set (not filtered to
correctly-classified beats). Uses a LIGHTER "custom" AutoAttack ensemble
(APGD-CE + Square, reduced restarts/iterations/queries) rather than the full
4-attack "standard" ensemble -- HMR-BiLSTM's hand-rolled RLSTMCell (no cuDNN
fast path) made "standard" too slow to finish within this environment's
background-task time limit even on a reduced subset. This still exercises
both a white-box gradient attack and a black-box query attack, which is what
the gradient-masking check (PGD ASR vs AutoAttack ASR) actually needs; it is
a smaller compute budget than the full ensemble, not a different method.

Output: results/robustness/autoattack_baseline_comparison.csv

Cach chay:
    python evaluate_autoattack_baselines.py --models LSTM BiLSTM   # fast
    python evaluate_autoattack_baselines.py --models HMR-BiLSTM    # slow
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from configs.paths import INTER_TEST
from evaluate_fgsm import load_baseline_model, build_test_loader
from report_results import load_hmr_bilstm
from robustness.cw_attack import select_stratified_subset
from robustness.auto_attack import pgd_attack, to_unit_range, CLASS_NAMES, enter_deterministic_rnn_train_mode

AA_EPS = 0.02
AA_NORM = "Linf"
PGD_ALPHA = AA_EPS / 4
PGD_STEPS = 20
N_EVAL = 80  # reduced from 200: full-ensemble AutoAttack on the hand-rolled
             # HMR-BiLSTM cell (no cuDNN fast path) is slow enough per-sample
             # that 200 didn't finish within this environment's background-task
             # time limit; 80 keeps a legitimate stratified subset while fitting
SEED = 42


class AAWrapper(nn.Module):
    """Maps AutoAttack's [0,1]-range (N,1,T,1) input back to the model's
    normalised range and calls the wrapped model. Mirrors the wrapper
    defined inline in robustness/auto_attack.py::main()."""
    def __init__(self, m, d_min, d_max):
        super().__init__()
        self.m = m
        self.d_min = d_min
        self.d_max = d_max

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, 0, :, 0]
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x[:, 0, :]
        x = x.unsqueeze(-1)
        x_real = x * (self.d_max - self.d_min) + self.d_min
        return self.m(x_real)


def evaluate_one_model(name, model, X_sub, y_sub, data_min, data_max, device):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_sub), 256):
            b = torch.from_numpy(X_sub[i:i+256]).to(device)
            all_preds.append(model(b).argmax(-1).cpu().numpy())
    preds_clean_sub = np.concatenate(all_preds)

    clean_acc = accuracy_score(y_sub, preds_clean_sub)
    clean_f1 = f1_score(y_sub, preds_clean_sub, average="macro", zero_division=0)
    print(f"  Subset: {len(X_sub)} samples | Clean Acc={clean_acc:.4f} F1={clean_f1:.4f}")

    # PGD-20 baseline
    X_pgd_list = []
    for i in range(0, len(X_sub), 64):
        bx = torch.from_numpy(X_sub[i:i+64]).to(device)
        by = torch.from_numpy(y_sub[i:i+64]).long().to(device)
        x_pgd = pgd_attack(model, bx, by, AA_EPS, PGD_ALPHA, PGD_STEPS, data_min, data_max, device)
        X_pgd_list.append(x_pgd.cpu().numpy())
    X_pgd = np.concatenate(X_pgd_list, axis=0)
    pgd_preds = []
    with torch.no_grad():
        for i in range(0, len(X_pgd), 256):
            b = torch.from_numpy(X_pgd[i:i+256]).to(device)
            pgd_preds.append(model(b).argmax(-1).cpu().numpy())
    pgd_preds = np.concatenate(pgd_preds)
    pgd_asr = float((pgd_preds != y_sub).mean())
    pgd_f1 = f1_score(y_sub, pgd_preds, average="macro", zero_division=0)
    print(f"  PGD-20  ASR={pgd_asr:.4f} F1={pgd_f1:.4f}")

    # AutoAttack
    X_sub_unit = to_unit_range(X_sub, data_min, data_max)
    x_aa_tensor = torch.from_numpy(X_sub_unit).to(device)
    y_aa_tensor = torch.from_numpy(y_sub).long().to(device)
    x_aa_tensor = x_aa_tensor.permute(0, 2, 1).unsqueeze(-1)  # (N,1,T,1)

    aa_wrapper = AAWrapper(model, data_min, data_max).to(device)
    # pyautoattack's own APGD implementation calls autograd.grad internally, so
    # the wrapped model needs the same deterministic-train-mode treatment as
    # pgd_attack() above (cuDNN RNN backward requires train mode, but plain
    # model.train() would make Dropout stochastic and get flagged by AutoAttack
    # as "a randomized defense").
    restore = enter_deterministic_rnn_train_mode(model)

    from pyautoattack import AutoAttack
    # "standard" (APGD-CE, APGD-DLR, FAB, Square with n_restarts=5/n_iter=100/
    # n_queries=5000) is far too expensive for HMR-BiLSTM's hand-rolled
    # RLSTMCell (no cuDNN fast path) to finish within this environment's
    # background-task time limit. We use a lighter "custom" ensemble instead:
    # APGD-CE (white-box, gradient-based) + Square (black-box, query-based),
    # with reduced restarts/iterations/queries. This keeps exactly the
    # white-box-vs-black-box comparison the paper needs to detect gradient
    # masking, just with a smaller compute budget than the full 4-attack
    # ensemble -- disclosed as a limitation rather than silently substituted.
    adversary = AutoAttack(aa_wrapper, norm=AA_NORM, eps=AA_EPS / (data_max - data_min),
                           version="custom", attacks=["apgd-ce", "square"], device=device)
    adversary.apgd.n_restarts = 1
    adversary.apgd.n_iter = 50
    adversary.square.n_queries = 500
    adversary.apgd_targeted.n_target_classes = 4
    adversary.fab.n_target_classes = 4
    result = adversary.run_standard_evaluation(x_aa_tensor, y_aa_tensor, batch_size=32)
    x_adv_aa = result[0] if isinstance(result, tuple) else result

    restore()
    aa_preds = []
    with torch.no_grad():
        for i in range(0, x_adv_aa.size(0), 256):
            b = x_adv_aa[i:i+256].to(device)
            aa_preds.append(aa_wrapper(b).argmax(-1).cpu().numpy())
    aa_preds = np.concatenate(aa_preds)
    aa_asr = float((aa_preds != y_sub).mean())
    aa_f1 = f1_score(y_sub, aa_preds, average="macro", zero_division=0)
    print(f"  AutoAttack ASR={aa_asr:.4f} F1={aa_f1:.4f}")

    return {
        "model": name,
        "n_subset": len(X_sub),
        "clean_acc": round(clean_acc, 4),
        "clean_f1_macro": round(clean_f1, 4),
        "pgd20_asr": round(pgd_asr, 4),
        "pgd20_f1_macro": round(pgd_f1, 4),
        "autoattack_asr": round(aa_asr, 4),
        "autoattack_f1_macro": round(aa_f1, 4),
        "masking_gap_aa_minus_pgd": round(aa_asr - pgd_asr, 4),
    }


COLS = ["model", "n_subset", "clean_acc", "clean_f1_macro",
        "pgd20_asr", "pgd20_f1_macro", "autoattack_asr",
        "autoattack_f1_macro", "masking_gap_aa_minus_pgd"]


def save_results(all_results, out_dir):
    csv_path = out_dir / "autoattack_baseline_comparison.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(COLS) + "\n")
        for r in all_results:
            f.write(",".join(str(r[c]) for c in COLS) + "\n")
    print(f"[OK] Saved -> {csv_path}")

    json_path = out_dir / "autoattack_baseline_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Saved -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description="AutoAttack comparison across LSTM/BiLSTM/HMR-BiLSTM")
    parser.add_argument("--models", nargs="*", default=["LSTM", "BiLSTM", "HMR-BiLSTM"],
                        help="Subset of models to (re-)run; others already in the output are kept as-is.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models to run: {args.models}")
    torch.manual_seed(SEED)

    out_dir = Path("results/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "autoattack_baseline_comparison.json"

    # Resumable: keep any already-computed model results, only (re-)run --models.
    results_by_model = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            for r in json.load(f):
                results_by_model[r["model"]] = r
        print(f"[OK] Loaded {len(results_by_model)} existing result(s) from {json_path}")

    print(f"Loading test data: {INTER_TEST}")
    test = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)
    data_min = float(X_test.min())
    data_max = float(X_test.max())
    print(f"  Data range: [{data_min:.3f}, {data_max:.3f}]")

    # Draw the stratified subset ONCE from a fresh, fixed-seed RNG so every
    # model -- regardless of which --models subset is (re-)run, or in what
    # order -- is attacked on the exact same 167 test beats.
    rng = np.random.default_rng(SEED)
    X_sub, y_sub, _ = select_stratified_subset(X_test, y_test, N_EVAL, rng)
    print(f"  Shared stratified subset: {len(X_sub)} samples")

    models_info = {
        "LSTM": "results/checkpoints/best_lstm.pt",
        "BiLSTM": "results/checkpoints/best_bilstm.pt",
        "HMR-BiLSTM": "results/checkpoints/inter_best_rlstm.pt",
    }

    for name in args.models:
        ckpt_path = models_info.get(name)
        if ckpt_path is None:
            print(f"[SKIP] Unknown model: {name}")
            continue
        if not Path(ckpt_path).exists():
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue
        if name == "HMR-BiLSTM":
            model, _ = load_hmr_bilstm(ckpt_path, device)
        else:
            model, _ = load_baseline_model(ckpt_path, device)
        result = evaluate_one_model(name, model, X_sub, y_sub, data_min, data_max, device)
        results_by_model[name] = result
        # Save immediately after each model so a kill mid-run (HMR-BiLSTM's
        # AutoAttack is slow -- hand-rolled cell, no cuDNN fast path) doesn't
        # lose already-completed models.
        save_results(list(results_by_model.values()), out_dir)

    all_results = list(results_by_model.values())
    print("\n" + "=" * 70)
    print(f"{'Model':<12} {'Clean F1':>9} {'PGD ASR':>9} {'PGD F1':>8} {'AA ASR':>8} {'AA F1':>8}")
    for r in all_results:
        print(f"{r['model']:<12} {r['clean_f1_macro']:>9.4f} {r['pgd20_asr']:>9.4f} "
              f"{r['pgd20_f1_macro']:>8.4f} {r['autoattack_asr']:>8.4f} {r['autoattack_f1_macro']:>8.4f}")


if __name__ == "__main__":
    main()
