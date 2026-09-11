"""
DSTA bien the C (per_dim blend, extra_edge, gamma softmax -- backbone tot
nhat da co, F1_macro=0.6357) + RR-interval noi vao TRUOC classifier cuoi
cung, KHONG di qua DSTA gate/beta nhu lan thu truoc (hmr_dsta_rr.py).

Ly do doi chien luoc: lan truoc dua RR vao z_t ben trong DSTALiteGate, nhung
duong di c_t = beta*c_rmc + (1-beta)*c_proposal voi beta khoi tao ~0.007
chan gan het tin hieu RR (rr.grad ~0.000001 luc init) -- ket qua "khong ro
rang" (inconclusive), khong phai bang chung RR vo dung. Lan nay RR di thang
toi classifier, khong qua cong nao ca -- duong gradient sach, khong bi
throttled.

RR di qua 1 MLP nho (rr_encoder) truoc khi noi, khong noi 3 so tho -- de co
du dung luong bieu dien canh tranh voi 192 chieu embedding chinh.

Backbone (BiDSTALite) GIU NGUYEN 100% cau hinh bien the C, khong doi gi --
single-variable so voi ket qua C da co (F1_macro=0.6357, F1-S=0.2547,
F1-F=0.4322).
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

from hmr_bilstm import AttentionPooling, ECGFeatureExtractor
from hmr_dsta_lite import BiDSTALite
from hmr_bilstm import RLSTMLoss

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

NUM_CLASSES = 5
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
S_IDX = 1


class DSTALiteClassifierRRShallow(nn.Module):
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
        rr_dim: int = 3,
        rr_hidden: int = 32,
    ):
        super().__init__()
        self.cnn = ECGFeatureExtractor(
            input_channels=input_size, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )
        # Bien the C: per_dim blend + extra_edge + gamma softmax -- KHONG doi
        self.birlstm = BiDSTALite(
            cnn_out_channels, hidden_size, num_layers=num_layers, dropout=dropout,
            attn_dim=attn_dim, num_heads=num_heads, rank=rank,
            extra_edge=True, gamma_mode="softmax", blend_mode="per_dim",
        )
        embed_dim = 2 * hidden_size
        self.attention_pool = AttentionPooling(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_dim, rr_hidden),
            nn.GELU(),
            nn.Linear(rr_hidden, rr_hidden),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + rr_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor, return_internals: bool = False):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features)
        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_pooled = self.dropout(self.layer_norm(h_pooled))

        rr_feat = self.rr_encoder(rr)
        combined = torch.cat([h_pooled, rr_feat], dim=-1)
        logits = self.classifier(combined)

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "attention_weights": attn_weights, "rr_feat": rr_feat}
        return logits


# =============================================================================
#  SMOKE TEST -- CPU, dummy data
# =============================================================================
def _smoke_test() -> None:
    torch.manual_seed(42)
    B, T, D, RR_DIM = 16, 187, 1, 3
    HIDDEN = 96

    x = torch.randn(B, T, D)
    rr = torch.randn(B, RR_DIM)
    y = torch.randint(0, NUM_CLASSES, (B,))

    model = DSTALiteClassifierRRShallow(input_size=D, hidden_size=HIDDEN, num_classes=NUM_CLASSES,
                                         num_layers=2, rr_dim=RR_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (DSTA-C goc: 603,118)")

    logits, internals = model(x, rr=rr, return_internals=True)
    assert logits.shape == (B, NUM_CLASSES)
    print(f"logits: {tuple(logits.shape)}  OK")

    rr_req = rr.clone().requires_grad_(True)
    logits2, internals2 = model(x, rr=rr_req, return_internals=True)
    criterion = RLSTMLoss(lambda_smooth=0.003)
    loss, comp = criterion(logits2, y, internals2["r_fwd"], internals2["r_bwd"])
    loss.backward()
    print(f"loss: {loss.item():.4f}")
    print(f"rr.grad abs sum: {rr_req.grad.abs().sum().item():.6f}  (nen LON HON nhieu so voi "
          f"~0.000001 cua lan thu truoc dua RR vao gate -- vi lan nay khong qua cong beta)")

    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print(f"Params khong nhan grad: {n_none}  |  grad=NaN: {n_nan}")
    print("Smoke test PASSED" if n_none == 0 and n_nan == 0 else "Smoke test FAILED")


if __name__ == "__main__":
    _smoke_test()
