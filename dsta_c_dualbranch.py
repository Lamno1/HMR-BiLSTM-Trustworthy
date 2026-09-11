"""
Kien truc tach-nhanh: CNN RIENG cho MLII va CNN RIENG cho kenh phu (V1...),
hop nhat o TANG DAC TRUNG (sau CNN, truoc DSTA-gate) -- khac han 2 thu
nghiem truoc:
  (1) 2-kenh-day-du (dsta_c_rr_template_2ch_train.py): 1 CNN DUY NHAT nhan
      ca 2 kenh lam input tho -- ep chia se filter ngay tu dau, da xac nhan
      LAM S SUP (AUC N-vs-S rieng benh nhan: MLII=0.74/0.61, V1=0.59/0.55 --
      V1 tu no da YEU HON cho S, tron chung filter keo S xuong theo).
  (2) V1-chi-1-dac-trung (chua code) -- an toan cho S nhung rut V1 ve 1 so
      duy nhat, mat het chi tiet hinh dang, kho giup F dat muc tot nhat
      (ban 2-kenh-day-du dat F=0.144 nho CNN hoc TOAN BO dang song V1).

Ban nay: MOI kenh co bo loc CNN RIENG (khong chia se trong so) -- tranh
dung "cach can bang bat buoc" da hai S -- nhung van giu DAY DU 187 diem
chi tiet cua V1 (khong rut gon ve 1 so) truoc khi hop nhat, giu co hoi cho
F dat muc tot. Chi 1 CONG DSTA DUY NHAT sau khi hop nhat (khong nhan doi
bat on dinh CP-decomposition da chung minh o file memory).

Cong them: balance regularizer (dsta_balance_reg.py) vao loss -- pha vo doi
xung ty le cua CP-decomposition, giam nhieu "ket qua ao theo seed" da xac
nhan la nguyen nhan toan hoc that.
"""

import torch
import torch.nn as nn

from hmr_bilstm import AttentionPooling, ECGFeatureExtractor, RLSTMLoss
from hmr_dsta_lite import BiDSTALite
from dsta_balance_reg import compute_balance_loss

NUM_CLASSES = 5
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
N_IDX, S_IDX, V_IDX, F_IDX = 0, 1, 2, 3


class DSTADualBranch(nn.Module):
    def __init__(
        self,
        hidden_size: int = 96,
        dropout: float = 0.25,
        num_classes: int = NUM_CLASSES,
        cnn_out_channels: int = 64,
        num_layers: int = 2,
        attn_dim: int = 24,
        num_heads: int = 2,
        rank: int = 4,
        rr_dim: int = 5,
        rr_hidden: int = 32,
    ):
        super().__init__()
        # 2 nhanh CNN DOC LAP, KHONG chia se trong so -- moi nhanh tu hoc
        # bo loc phu hop rieng cho dac tinh cua kenh do.
        self.cnn_mlii = ECGFeatureExtractor(
            input_channels=1, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )
        self.cnn_v1 = ECGFeatureExtractor(
            input_channels=1, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )

        # Hop nhat O TANG DAC TRUNG (sau CNN, truoc DSTA-gate) -- dim gap doi
        fused_dim = 2 * cnn_out_channels
        self.birlstm = BiDSTALite(
            fused_dim, hidden_size, num_layers=num_layers, dropout=dropout,
            attn_dim=attn_dim, num_heads=num_heads, rank=rank,
            extra_edge=True, gamma_mode="softmax", blend_mode="per_dim",
        )
        embed_dim = 2 * hidden_size
        self.attention_pool = AttentionPooling(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_dim, rr_hidden), nn.GELU(), nn.Linear(rr_hidden, rr_hidden),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + rr_hidden, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, rr, return_internals=False):
        """x: (B, 187, 2) -- kenh 0=MLII, kenh 1=phu (V1...). rr: (B, rr_dim)."""
        x_mlii = x[..., 0:1]  # (B,187,1)
        x_v1 = x[..., 1:2]    # (B,187,1)

        f_mlii = self.cnn_mlii(x_mlii)  # (B, T', cnn_out_channels)
        f_v1 = self.cnn_v1(x_v1)        # (B, T', cnn_out_channels)
        f_fused = torch.cat([f_mlii, f_v1], dim=-1)  # (B, T', 2*cnn_out_channels)

        h_seq, r_fwd, r_bwd = self.birlstm(f_fused)
        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_embed = self.dropout(self.layer_norm(h_pooled))

        rr_enc = self.rr_encoder(rr)
        combined = torch.cat([h_embed, rr_enc], dim=-1)
        logits = self.classifier(combined)

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "attention_weights": attn_weights}
        return logits


class DualBranchLoss(nn.Module):
    """RLSTMLoss (focal + smoothness, giu nguyen) + balance regularizer nho
    cho DSTA-gate (pha vo doi xung ty le CP-decomposition)."""

    def __init__(self, class_weights=None, focal_gamma=1.5, lambda_smooth=0.003, lambda_balance=0.01):
        super().__init__()
        self.task_loss_fn = RLSTMLoss(lambda_smooth=lambda_smooth, class_weights=class_weights,
                                       use_focal=True, focal_gamma=focal_gamma)
        self.lambda_balance = lambda_balance

    def forward(self, logits, targets, r_fwd, r_bwd, model):
        total, comps = self.task_loss_fn(logits, targets, r_fwd, r_bwd)
        l_balance = compute_balance_loss(model)
        total = total + self.lambda_balance * l_balance
        comps["balance"] = l_balance.detach()
        comps["total"] = total.detach()
        return total, comps


def class_weights_from(y_train, num_classes=NUM_CLASSES):
    import numpy as np
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (num_classes * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


# =============================================================================
#  SMOKE TEST -- CPU, dummy data. Kiem tra: shape dung, gradient sach, VA
#  kiem tra hop nhat KHONG lam mat thong tin (2 nhanh nhan input KHAC nhau
#  phai cho output KHAC nhau -- tranh loi vo tinh dung chung 1 nhanh 2 lan).
# =============================================================================
def _smoke_test():
    torch.manual_seed(42)
    B, T = 16, 187
    x = torch.randn(B, T, 2)
    rr = torch.randn(B, 5)
    y = torch.randint(0, 5, (B,))

    model = DSTADualBranch(hidden_size=96, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    logits, internals = model(x, rr, return_internals=True)
    print(f"logits: {tuple(logits.shape)}")

    # Kiem tra 2 nhanh CNN thuc su DOC LAP (trong so khac nhau ngay tu dau,
    # khong vo tinh chia se) -- so sanh 1 vai tham so dau tien
    same_weights = torch.equal(
        model.cnn_mlii.conv1.weight, model.cnn_v1.conv1.weight)
    print(f"2 nhanh CNN co trung trong so ngau nhien (ky vong KHONG): {same_weights}")
    assert not same_weights, "LOI: 2 nhanh dang dung chung 1 module!"

    # Kiem tra hop nhat co dung ca 2 kenh: doi rieng kenh V1, logit phai doi
    model.eval()
    with torch.no_grad():
        logits_a, _ = model(x, rr, return_internals=True)
        x2 = x.clone()
        x2[..., 1] = torch.randn(B, T)  # chi doi kenh V1
        logits_b, _ = model(x2, rr, return_internals=True)
    diff = (logits_a - logits_b).abs().max().item()
    print(f"Doi rieng kenh V1 -> |thay doi logit| = {diff:.6f} (ky vong > 0, khong duoc = 0)")
    assert diff > 1e-6, "LOI: doi kenh V1 khong anh huong gi den logit -- nhanh V1 khong duoc dung!"

    cw = torch.tensor([0.5, 5.0, 2.0, 10.0, 20.0])
    criterion = DualBranchLoss(class_weights=cw, lambda_balance=0.01)
    model.train()
    logits, internals = model(x, rr, return_internals=True)
    loss, comps = criterion(logits, y, internals["r_fwd"], internals["r_bwd"], model)
    print("loss comps:", {k: round(v.item(), 4) for k, v in comps.items()})
    loss.backward()

    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print(f"Params khong nhan grad: {n_none}  |  grad=NaN: {n_nan}")

    grad_mlii = sum(p.grad.abs().sum().item() for p in model.cnn_mlii.parameters() if p.grad is not None)
    grad_v1 = sum(p.grad.abs().sum().item() for p in model.cnn_v1.parameters() if p.grad is not None)
    print(f"Gradient nhanh MLII: {grad_mlii:.4f}  |  Gradient nhanh V1: {grad_v1:.4f}  (ca 2 phai > 0)")

    ok = (not same_weights and diff > 1e-6 and n_none == 0 and n_nan == 0
          and grad_mlii > 0 and grad_v1 > 0)
    print("Smoke test PASSED" if ok else "Smoke test FAILED")


if __name__ == "__main__":
    _smoke_test()
