"""
train_rr_prototype.py
======================
Prototype: does adding RR-interval features improve F1(S) for HMR-BiLSTM?

Trains ONE model (full HMR-BiLSTM + RR side input) on data/processed/splits_rr/
(produced by validation/preprocess_aami_rr.py) using the exact same recipe
(hyperparameters, class weights, focal loss, adversarial training, cosine LR,
early stopping) as the "full" variant in run_ablation.py, so the comparison
against the existing F1(S)=0.267 baseline is controlled — the only thing that
changes is the RR input.

Does NOT touch data/processed/splits/, results/ablation/, or any existing
checkpoint/result file. Results are written to results/rr_prototype/.

Cach chay:
    python train_rr_prototype.py
"""

import argparse
import json
import math
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)

from hmr_bilstm_rr import RLSTMClassifierRR
from hmr_bilstm_ablation import RLSTMLoss

CONFIG = {
    "data_dir":                "data/processed/splits_rr",
    "seed":                    42,
    "batch_size":              128,
    "hidden_size":             96,
    "dropout":                 0.25,
    "learning_rate":           1e-3,
    "min_lr":                  1e-5,
    "weight_decay":            1e-4,
    "lambda_smooth":           0.003,
    "epochs":                  45,
    "early_stopping_patience": 8,
    "grad_clip":               1.0,
    "num_classes":             5,
    "input_size":              1,
    "cnn_out_channels":        64,
    "num_layers":              2,
    "use_focal_loss":          True,
    "focal_gamma":             1.5,
    "adversarial_training":    True,
    "adv_epsilon":             0.02,
    "adv_ratio":               0.3,
    "use_class_weights":       True,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(epoch, total_epochs, base_lr, min_lr):
    progress = epoch / max(1, total_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def load_data(data_dir, batch_size):
    def make_loader(split, shuffle=False):
        d = np.load(f"{data_dir}/inter_{split}.npz")
        ds = TensorDataset(
            torch.from_numpy(d["X"]).float(),
            torch.from_numpy(d["rr"]).float(),
            torch.from_numpy(d["y"]).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        make_loader("train", shuffle=True),
        make_loader("val"),
        make_loader("test"),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for X, rr, y in loader:
        X, rr = X.to(device), rr.to(device)
        logits = model(X, rr)
        all_logits.append(logits.cpu())
        all_y.append(y)

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(dim=-1).numpy()

    metrics = {
        "accuracy":        float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "f1_weighted":     float(f1_score(y_true, preds, labels=[0, 1, 2, 3], average="weighted", zero_division=0)),
    }
    try:
        valid_idx = np.isin(y_true, [0, 1, 2, 3])
        y_true_4c = y_true[valid_idx]
        y_prob_4c = probs[valid_idx][:, :4]
        present_labels = sorted(set(y_true_4c.tolist()) & {0, 1, 2, 3})
        if len(present_labels) >= 2:
            row_sums = y_prob_4c.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-9, row_sums)
            y_prob_4c = y_prob_4c / row_sums
            metrics["auc_ovr"] = float(roc_auc_score(
                y_true_4c, y_prob_4c, multi_class="ovr", average="macro", labels=present_labels,
            ))
        else:
            metrics["auc_ovr"] = 0.0
    except Exception as e:
        print(f"  Warning: AUC calculation failed: {e}")
        metrics["auc_ovr"] = 0.0

    per_class = f1_score(y_true, preds, average=None, zero_division=0)
    for i, cls in enumerate(["N", "S", "V", "F", "Q"]):
        metrics[f"f1_{cls}"] = float(per_class[i]) if i < len(per_class) else 0.0

    metrics["_preds"] = preds
    metrics["_y_true"] = y_true
    return metrics


def fgsm_attack_train(model, x, rr, y, epsilon, criterion):
    """FGSM on the raw waveform only; RR features are left unperturbed
    (they are computed, not sensor-observed, so perturbing them has no
    physical attack meaning here)."""
    x_adv = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x_adv, rr)
        loss, _ = criterion(logits, y, r_fwd=None, r_bwd=None)
        model.zero_grad()
        loss.backward()
    return (x + epsilon * x_adv.grad.sign()).detach()


def train_one_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss, n = 0.0, 0

    adv_training = cfg["adversarial_training"]
    adv_epsilon  = cfg["adv_epsilon"]
    adv_ratio    = cfg["adv_ratio"]

    for X, rr, y in loader:
        X, rr, y = X.to(device), rr.to(device), y.to(device)

        if adv_training and adv_epsilon > 0:
            split = int(len(X) * (1 - adv_ratio))
            X_adv = fgsm_attack_train(
                model, X[split:], rr[split:], y[split:], adv_epsilon, criterion
            )
            X = torch.cat([X[:split], X_adv], dim=0)
            y = torch.cat([y[:split], y[split:]], dim=0)
            rr = torch.cat([rr[:split], rr[split:]], dim=0)

        optimizer.zero_grad()

        logits, internals = model(X, rr, return_internals=True)
        loss, _ = criterion(
            logits, y,
            r_fwd=internals["r_fwd"],
            r_bwd=internals["r_bwd"],
        )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="Train HMR-BiLSTM + RR-interval prototype")
    parser.add_argument("--data-dir", default=CONFIG["data_dir"],
                        help="Directory with inter_{train,val,test}.npz (X,y,rr). "
                             "Use data/processed/splits_rr_smote for the SMOTE variant.")
    parser.add_argument("--out-dir", default="results/rr_prototype",
                        help="Where to write checkpoints/ and rr_prototype_results.json")
    parser.add_argument("--class-weights-dir", default=None,
                        help="Compute class_weights from THIS directory's inter_train.npz "
                             "instead of --data-dir. Use this to keep pre-SMOTE class weights "
                             "when training on an oversampled --data-dir (e.g. "
                             "--data-dir data/processed/splits_rr_smote "
                             "--class-weights-dir data/processed/splits_rr) so oversampling "
                             "and inverse-frequency loss weighting don't cancel each other out.")
    args = parser.parse_args()

    cfg = {**CONFIG, "data_dir": args.data_dir}
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Data dir: {cfg['data_dir']}")

    data_dir = Path(cfg["data_dir"])
    if not (data_dir / "inter_train.npz").exists():
        print(f"[ERROR] {data_dir}/inter_train.npz not found. "
              f"Run: python validation/preprocess_aami_rr.py first.")
        return

    print("\n[Loading data]")
    train_loader, val_loader, test_loader = load_data(cfg["data_dir"], cfg["batch_size"])

    class_weights = None
    if cfg["use_class_weights"]:
        cw_source = args.class_weights_dir if args.class_weights_dir else cfg["data_dir"]
        print(f"Class weights computed from: {cw_source}"
              + ("  (kept separate from training data-dir)" if args.class_weights_dir else ""))
        y_inter_tr = np.load(f"{cw_source}/inter_train.npz")["y"]
        counts = np.bincount(y_inter_tr, minlength=cfg["num_classes"]).astype(np.float64)
        counts = np.where(counts == 0, 1e-9, counts)
        cw_arr = counts.sum() / (float(cfg["num_classes"]) * counts)
        cw_arr = np.clip(cw_arr, 0.5, 50.0).astype(np.float32)
        class_weights = torch.from_numpy(cw_arr).float().to(device)
        print(f"Class weights (inter_train): {cw_arr}")

    set_seed(cfg["seed"])

    model = RLSTMClassifierRR(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
        cnn_out_channels=cfg["cnn_out_channels"],
        num_layers=cfg["num_layers"],
        n_rr_features=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = RLSTMLoss(
        lambda_smooth=cfg["lambda_smooth"],
        class_weights=class_weights,
        use_focal=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )

    ckpt_path = ckpt_dir / "best_rlstm_rr.pt"
    best_f1, best_epoch, patience_cnt = 0.0, 0, 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        lr = cosine_lr(epoch - 1, cfg["epochs"], cfg["learning_rate"], cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg)
        val_m = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        marker = ""
        if val_m["f1_macro"] > best_f1:
            best_f1, best_epoch, patience_cnt = val_m["f1_macro"], epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_f1_macro": best_f1,
                "n_params": n_params,
            }, ckpt_path)
            marker = " <-- best"
        else:
            patience_cnt += 1

        history.append({
            "epoch": epoch, "lr": lr, "train_loss": train_loss,
            "val_f1_macro": val_m["f1_macro"], "val_accuracy": val_m["accuracy"],
        })

        print(f"  ep{epoch:3d} | lr={lr:.5f} | loss={train_loss:.4f} | "
              f"val_f1={val_m['f1_macro']:.4f} acc={val_m['accuracy']:.4f} | "
              f"{elapsed:.0f}s{marker}")

        if patience_cnt >= cfg["early_stopping_patience"]:
            print(f"\n  [Early stop] ep={epoch} best={best_epoch} F1={best_f1:.4f}")
            break

    print(f"\n  [Loading best checkpoint — epoch {best_epoch}]")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    test_m = evaluate(model, test_loader, device)

    report = classification_report(
        test_m["_y_true"], test_m["_preds"],
        target_names=["N", "S", "V", "F", "Q"], zero_division=0, digits=4,
    )

    print(f"\n  Test F1 macro: {test_m['f1_macro']:.4f}")
    print(f"  Test Accuracy: {test_m['accuracy']:.4f}")
    print(f"  Test AUC:      {test_m['auc_ovr']:.4f}")
    print(f"  Test F1(S):    {test_m['f1_S']:.4f}   (no-RR: 0.2674 | RR-ratio only: 0.4314)")
    print(f"  Test F1(V):    {test_m['f1_V']:.4f}   (no-RR: 0.8689 | RR-ratio only: 0.8457)")
    print(f"  Test F1(F):    {test_m['f1_F']:.4f}   (no-RR: 0.2958 | RR-ratio only: 0.1841)")
    print(f"\n  Per-class:\n{report}")

    clean_test = {k: v for k, v in test_m.items() if not k.startswith("_")}
    result = {
        "variant": "hmr_bilstm_rr",
        "label": "HMR-BiLSTM + RR-interval (prototype)",
        "data_dir": cfg["data_dir"],
        "class_weights_dir": args.class_weights_dir if args.class_weights_dir else cfg["data_dir"],
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_metrics": clean_test,
        "report": report,
        "history": history,
        "baseline_no_rr": {"f1_S": 0.2674, "f1_V": 0.8689, "f1_F": 0.2958, "f1_macro": 0.5964},
        "rr_ratio_only": {"f1_S": 0.4314, "f1_V": 0.8457, "f1_F": 0.1841, "f1_macro": 0.6043},
    }
    with open(out_dir / "rr_prototype_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Saved -> {out_dir / 'rr_prototype_results.json'}")


if __name__ == "__main__":
    main()
