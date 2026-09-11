"""
One-vs-rest (OvR): thay vi 1 softmax dung chung cho N/S/V/F (buoc cac lop
canh tranh gradient qua chuan hoa -- da chung minh bang toan hoc la nguyen
nhan goc cua MOI that bai truoc do, ke ca co che "s-boost" tuong tu duoc co
lap o forward-value nhung khong co lap o gradient), dung 4 dau sigmoid DOC
LAP, moi dau co BCE loss RIENG (khong chia se chuan hoa) -- gradient cho
lop N khong con bi buoc phai doi nguoc voi gradient cho lop F nua.

Backbone giu nguyen y het "C khong RR" (CNN + BiDSTALite + pool). Classifier
cuoi la 4 unit sigmoid doc lap thay vi 1 softmax 4-lop.

Du doan cuoi: argmax cua 4 xac suat sigmoid (khong chuan hoa lai) -- neu
nhieu lop cung "tu tin" thi lay lop co xac suat cao nhat; day la lua chon
suy luan don gian, co the tinh chinh sau.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr_bilstm import AttentionPooling, ECGFeatureExtractor

NUM_CLASSES = 4
N_IDX, S_IDX, V_IDX, F_IDX = 0, 1, 2, 3


class DSTACOvR(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 96,
        dropout: float = 0.25,
        cnn_out_channels: int = 64,
        num_layers: int = 2,
        attn_dim: int = 24,
        num_heads: int = 2,
        rank: int = 4,
    ):
        super().__init__()
        from hmr_dsta_lite import BiDSTALite
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
            nn.Linear(embed_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_CLASSES),  # 4 logit DOC LAP (khong qua softmax chung)
        )

    def forward(self, x, return_internals=False):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features)
        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_embed = self.dropout(self.layer_norm(h_pooled))
        logits = self.classifier(h_embed)  # dung nhu BCE-per-class, KHONG qua softmax

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "attention_weights": attn_weights}
        return logits


class OvRLoss(nn.Module):
    """4 BCE DOC LAP (mot cho moi lop) + focal modulation rieng cho tung
    lop + smoothness loss (giu quy uoc cu). KHONG dung softmax/CE chung."""

    def __init__(self, pos_weights=None, focal_gamma=1.5, lambda_smooth=0.003):
        super().__init__()
        self.register_buffer("pos_weights", pos_weights if pos_weights is not None else torch.ones(NUM_CLASSES))
        self.focal_gamma = focal_gamma
        self.lambda_smooth = lambda_smooth

    @staticmethod
    def _smooth(r_seq):
        if r_seq.size(1) < 2:
            return r_seq.sum() * 0.0
        diff = r_seq[:, 1:, :] - r_seq[:, :-1, :]
        return (diff ** 2).sum(dim=-1).mean()

    def forward(self, logits, y, r_fwd, r_bwd):
        y_onehot = F.one_hot(y, NUM_CLASSES).float()  # (B,4)
        probs = torch.sigmoid(logits)
        pt = torch.where(y_onehot.bool(), probs, 1 - probs).clamp(min=1e-6, max=1 - 1e-6)
        focal_w = (1 - pt) ** self.focal_gamma
        bce = F.binary_cross_entropy_with_logits(
            logits, y_onehot, reduction="none", pos_weight=self.pos_weights,
        )
        l_main = (focal_w * bce).sum(dim=-1).mean()

        l_smooth = (self._smooth(r_fwd) + self._smooth(r_bwd)) / 2.0
        total = l_main + self.lambda_smooth * l_smooth
        return total, {"main": l_main.detach(), "smooth": l_smooth.detach(), "total": total.detach()}


def class_pos_weights(y_train, num_classes=NUM_CLASSES):
    """pos_weight cho BCEWithLogits: neg_count/pos_count cho tung lop (chuan
    cach can bang nhi phan, khac inv-freq da dung cho softmax truoc do)."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    n_total = len(y_train)
    pw = (n_total - counts) / np.clip(counts, 1, None)
    pw = np.clip(pw, 0.5, 100.0).astype(np.float32)
    return torch.from_numpy(pw)


# =============================================================================
#  SMOKE TEST
# =============================================================================
def _smoke_test():
    torch.manual_seed(42)
    B, T, D = 16, 187, 1
    x = torch.randn(B, T, D)
    y = torch.randint(0, 4, (B,))

    model = DSTACOvR(input_size=D, hidden_size=96, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    logits, internals = model(x, return_internals=True)
    print(f"logits: {tuple(logits.shape)}")

    pw = torch.tensor([0.5, 5.0, 2.0, 10.0])
    criterion = OvRLoss(pos_weights=pw, focal_gamma=1.5)
    loss, comps = criterion(logits, y, internals["r_fwd"], internals["r_bwd"])
    print("loss comps:", {k: round(v.item(), 4) for k, v in comps.items()})
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print(f"Params khong nhan grad: {n_none}  |  grad=NaN: {n_nan}")
    print("Smoke test PASSED" if n_none == 0 and n_nan == 0 else "Smoke test FAILED")


if __name__ == "__main__":
    _smoke_test()
