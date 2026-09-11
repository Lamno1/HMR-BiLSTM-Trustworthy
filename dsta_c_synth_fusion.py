"""
"Dot bien": tong hop du lieu F gia lap bang cach tron song N that + V that
theo ty le alpha ngau nhien, cong voi 1 dau du doan alpha CO GIAM SAT (khac
Mixture-Decomposition cua CAMP-Net -- khong giam sat, phai tu hoc alpha tren
384 mau F that, khong on dinh).

Kiem chung truoc bang CPU (khong can model): AUC(F gia lap vs F that) ve do
rong QRS = 0.527 (~0.5 = gan nhu khong phan biet duoc) -- mau tong hop hop
ly ve mat thong ke, khong phai rac.

Kien truc: DSTALiteClassifier bien the C (per_dim, extra_edge, softmax --
KHONG doi) + 1 dau alpha nho (Linear->ReLU->Linear->sigmoid) tren embedding
da pool. Loss = focal task loss (tren CA that lan gia lap) + MSE(alpha du
doan, alpha that) CHI tren mau gia lap.

Du lieu gia lap: tron X_train[N] va X_train[V] that (khong dung tu val/test
-- tranh leak), alpha ~ U(0.3, 0.7) (tranh gan 0/1 -- se giong het N hoac V
thuan, khong con la "blend"). n_synth co the dieu chinh -- mac dinh gap ~13x
so mau F that (384) de F khong con qua hiem trong batch.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from hmr_bilstm import AttentionPooling, ECGFeatureExtractor, RLSTMLoss
from hmr_dsta_lite import BiDSTALite

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

NUM_CLASSES = 5
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
N_IDX, S_IDX, V_IDX, F_IDX = 0, 1, 2, 3


class DSTALiteClassifierSynthFusion(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 96,
        dropout: float = 0.25,
        num_classes: int = 5,
        cnn_out_channels: int = 64,
        num_layers: int = 2,
        attn_dim: int = 24,
        num_heads: int = 2,
        rank: int = 4,
    ):
        super().__init__()
        self.cnn = ECGFeatureExtractor(
            input_channels=input_size, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )
        self.birlstm = BiDSTALite(
            cnn_out_channels, hidden_size, num_layers=num_layers, dropout=dropout,
            attn_dim=attn_dim, num_heads=num_heads, rank=rank,
            extra_edge=True, gamma_mode="softmax", blend_mode="per_dim",
        )
        embed_dim = 2 * hidden_size
        self.attention_pool = AttentionPooling(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        # Dau alpha CO GIAM SAT -- khac MixtureDecompositionHead cua CAMP-Net
        # (khong giam sat). Chi dung de du doan ty le tron cho mau gia lap.
        # FIX (v2): nhan h_embed.detach() -- KHONG duoc chia se gradient nguoc
        # ve backbone voi classifier nua (loi "sai vi tri" phat hien duoc:
        # ban dau alpha_head va classifier dung CHUNG embedding, loss alpha
        # ep backbone hoc bieu dien tot cho viec doan ty le tron cua du lieu
        # GIA LAP, khong dam bao tot cho phan loai F THAT -- giong het xung
        # dot CM-vs-Mixture da gap o CAMP-Net). Gio alpha_head la 1 nhanh doc,
        # chi hoc rieng no, khong con anh huong nguoc len CNN/DSTA gate.
        self.alpha_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, return_internals: bool = False):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features)
        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_embed = self.dropout(self.layer_norm(h_pooled))
        logits = self.classifier(h_embed)
        alpha_pred = torch.sigmoid(self.alpha_head(h_embed.detach())).squeeze(-1)

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "alpha_pred": alpha_pred,
                             "attention_weights": attn_weights}
        return logits


class SynthFusionLoss(nn.Module):
    def __init__(self, lambda_smooth=0.003, class_weights=None, focal_gamma=1.5, lambda_alpha=0.2):
        super().__init__()
        self.task_loss_fn = RLSTMLoss(lambda_smooth=lambda_smooth, class_weights=class_weights,
                                       use_focal=True, focal_gamma=focal_gamma)
        self.lambda_alpha = lambda_alpha

    def forward(self, logits, targets, internals, alpha_true, is_synth):
        total, comps = self.task_loss_fn(logits, targets, internals["r_fwd"], internals["r_bwd"])
        if is_synth.sum() > 0:
            l_alpha = F.mse_loss(internals["alpha_pred"][is_synth], alpha_true[is_synth])
        else:
            l_alpha = logits.new_zeros(())
        total = total + self.lambda_alpha * l_alpha
        comps["alpha"] = l_alpha.detach()
        comps["total"] = total.detach()
        return total, comps


def synthesize_fusion_batch(X_train, y_train, n_synth, rng, alpha_lo=0.3, alpha_hi=0.7):
    """Tron N that + V that tu chinh X_train (KHONG dung val/test -- tranh leak)."""
    N_idx = np.where(y_train == N_IDX)[0]
    V_idx = np.where(y_train == V_IDX)[0]
    n_pick = rng.choice(N_idx, n_synth)
    v_pick = rng.choice(V_idx, n_synth)
    alphas = rng.uniform(alpha_lo, alpha_hi, n_synth).astype(np.float32)
    X_synth = alphas[:, None, None] * X_train[n_pick] + (1 - alphas)[:, None, None] * X_train[v_pick]
    y_synth = np.full(n_synth, F_IDX, dtype=np.int64)
    return X_synth.astype(np.float32), y_synth, alphas


def class_weights_from(y_train, num_classes=NUM_CLASSES):
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (num_classes * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


# =============================================================================
#  SMOKE TEST -- CPU, dummy data
# =============================================================================
def _smoke_test():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    B, T, D = 16, 187, 1

    X_train = np.random.randn(200, T, D).astype(np.float32)
    y_train = rng.integers(0, 5, 200)
    y_train[:40] = N_IDX
    y_train[40:80] = V_IDX

    X_synth, y_synth, alpha_synth = synthesize_fusion_batch(X_train, y_train, n_synth=8, rng=rng)
    print(f"X_synth shape: {X_synth.shape}, y_synth all F? {(y_synth == F_IDX).all()}")
    print(f"alpha range: [{alpha_synth.min():.3f}, {alpha_synth.max():.3f}]")

    X_real = torch.from_numpy(X_train[:B])
    y_real = torch.from_numpy(y_train[:B])
    X_s = torch.from_numpy(X_synth)
    y_s = torch.from_numpy(y_synth)
    alpha_s = torch.from_numpy(alpha_synth)

    x_batch = torch.cat([X_real, X_s], dim=0)
    y_batch = torch.cat([y_real, y_s], dim=0)
    alpha_batch = torch.cat([torch.zeros(B), alpha_s], dim=0)
    is_synth = torch.cat([torch.zeros(B, dtype=torch.bool), torch.ones(8, dtype=torch.bool)], dim=0)

    model = DSTALiteClassifierSynthFusion(input_size=D, hidden_size=96, num_classes=5, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    logits, internals = model(x_batch, return_internals=True)
    print(f"logits: {tuple(logits.shape)}  alpha_pred: {tuple(internals['alpha_pred'].shape)}")

    cw = class_weights_from(y_train)
    criterion = SynthFusionLoss(class_weights=cw)
    loss, comps = criterion(logits, y_batch, internals, alpha_batch, is_synth)
    print(f"loss components: " + "  ".join(f"{k}={v.item():.4f}" for k, v in comps.items()))
    loss.backward()

    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print(f"Params khong nhan grad: {n_none}  |  grad=NaN: {n_nan}")
    print("Smoke test PASSED" if n_none == 0 and n_nan == 0 else "Smoke test FAILED")


if __name__ == "__main__":
    _smoke_test()
