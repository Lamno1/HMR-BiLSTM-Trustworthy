"""
Buoc 1-2 cua ke hoach da chot (2026-08-21): dung cau hinh C_neither
(lambda_cm=0, lambda_mix=0, lambda_rep=0) lam nen, doi backbone tu nn.LSTM
chuan sang BiRLSTM (keep/add softmax blend + beta-gated hybrid path -- giong
het RLSTMClassifier chinh thuc), xem F1-F co phuc hoi khong.

Co so: tong hop TOAN BO thi nghiem trong du an (baseline don gian, DSTA A-D,
CAMP-Net ban nn.LSTM) cho thay F1-F chi vuot 0.15 khi model CO co che RMC
hybrid-blend + beta-gate trong cell hoi quy. CAMP-Net (nn.LSTM, khong co co
che nay) bi chan duoi 0.11 (C_neither, tot nhat trong 3 bien the CAMP-Net).

Neu lan nay van that bai (F1-F khong phuc hoi ro ret) -> xem xet lai bien
the C cua DSTA (per_dim blend + extra_edge, F1_macro=0.6357, F1-F=0.4322)
lam huong chinh cho paper thay vi CAMP-Net.
"""

import json
import sys
from pathlib import Path

import torch

from camp_net import (
    load_split, class_weights_from,
)
from camp_net_ablation import train_one_variant

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

REFERENCE = {
    "official_hmr_bilstm": {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}},
    "dsta_c": {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}},
    "camp_net_c_neither_plain_lstm": {"f1_macro": 0.4985, "f1_per_class": {"N": 0.9221, "S": 0.2049, "V": 0.7562, "F": 0.1108}},
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    result = train_one_variant(
        "RMC_backbone_neither", dict(lambda_cm=0.0, lambda_mix=0.0, lambda_rep=0.0),
        X_train, y_train, X_val, y_val, X_test, y_test, device,
    )

    print("\n" + "=" * 100)
    print(" SUMMARY: CAMP-Net + RMC backbone vs cac moc tham chieu ")
    print("=" * 100)
    print(f"{'Model':<32} {'F1_macro':>10} {'F1-N':>8} {'F1-S':>8} {'F1-V':>8} {'F1-F':>8}")
    for name, r in REFERENCE.items():
        fc = r["f1_per_class"]
        print(f"{name:<32} {r['f1_macro']:>10.4f} {fc['N']:>8.4f} {fc['S']:>8.4f} {fc['V']:>8.4f} {fc['F']:>8.4f}")
    fc = result["f1_per_class"]
    print(f"{'CAMP-Net RMC backbone':<32} {result['f1_macro']:>10.4f} "
          f"{fc['N']:>8.4f} {fc['S']:>8.4f} {fc['V']:>8.4f} {fc['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/camp_net_rmc_test_results.json")
    with open(out_path, "w") as f:
        json.dump({"reference": REFERENCE, "camp_net_rmc_backbone": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
