"""
RMC backbone (da xac nhan giup F1-F: 0.1108 -> 0.3007) + Confusion-Margin
(tung giup F1-S: 0.1966 -> 0.4299 khi dung voi nn.LSTM + classifier da sua).
Muc tieu: xem 2 cai cong lai co giu duoc ca 2 loi ich khong, hay lai xung
dot nhu Mixture-Decomposition da tung lam voi F.

lambda_mix=0, lambda_rep=0 -- chua dua Mixture tro lai (da xac nhan gay hai
cho F o moi cau hinh tung thu).
"""

import json
import sys
from pathlib import Path

import torch

from camp_net import load_split
from camp_net_ablation import train_one_variant

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

REFERENCE = {
    "official_hmr_bilstm": {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}},
    "dsta_c": {"f1_macro": 0.6357, "f1_per_class": {"N": 0.9660, "S": 0.2547, "V": 0.8899, "F": 0.4322}},
    "camp_net_rmc_neither": {"f1_macro": 0.5905, "f1_per_class": {"N": 0.9634, "S": 0.1973, "V": 0.9005, "F": 0.3007}},
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    result = train_one_variant(
        "RMC_backbone_cm_only", dict(lambda_cm=0.3, lambda_mix=0.0, lambda_rep=0.0),
        X_train, y_train, X_val, y_val, X_test, y_test, device,
    )

    print("\n" + "=" * 100)
    print(" SUMMARY: CAMP-Net RMC backbone + Confusion-Margin vs cac moc tham chieu ")
    print("=" * 100)
    print(f"{'Model':<32} {'F1_macro':>10} {'F1-N':>8} {'F1-S':>8} {'F1-V':>8} {'F1-F':>8}")
    for name, r in REFERENCE.items():
        fc = r["f1_per_class"]
        print(f"{name:<32} {r['f1_macro']:>10.4f} {fc['N']:>8.4f} {fc['S']:>8.4f} {fc['V']:>8.4f} {fc['F']:>8.4f}")
    fc = result["f1_per_class"]
    print(f"{'CAMP-Net RMC + CM':<32} {result['f1_macro']:>10.4f} "
          f"{fc['N']:>8.4f} {fc['S']:>8.4f} {fc['V']:>8.4f} {fc['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/camp_net_rmc_cm_results.json")
    with open(out_path, "w") as f:
        json.dump({"reference": REFERENCE, "camp_net_rmc_cm": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
