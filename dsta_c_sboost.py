"""
Co che "chi duoc cong ve phia S": thay vi noi RR vao embedding dung chung
(cach da chung minh bang toan hoc la lam hong F -- xem phan tich neighbor-
purity: F giong N toi 98.9% trong khong gian RR+template, chi 66.3% trong
khong gian raw waveform), backbone/classifier chinh GIU NGUYEN y het "C khong
RR" (F1-F tot nhat tung do duoc = 0.4322) -- RR KHONG duoc phep cham vao
logit N/V/F duoi bat ky hinh thuc nao.

Mot nhanh RR nho, doc lap, chi tao ra 1 SO KHONG AM (qua softplus) duoc CONG
THEM vao DUY NHAT logit cua S truoc softmax:

    logits = classifier_main(h_embed)          # giong het "C khong RR"
    s_boost = softplus(rr_boost_head(rr))       # >= 0, khong bao gio am
    logits[:, S_IDX] += s_boost                 # CHI cong cho S

Vi logit_N, logit_V, logit_F HOAN TOAN khong bi cham vao, ty le tuong doi
giua chung (logit_N - logit_F, logit_N - logit_V, ...) khong doi -- ve mat
toan hoc, co che nay KHONG THE lam F bi keo ve phia N nhieu hon so voi khi
khong co RR (chi anh huong gian tiep qua chuan hoa softmax, dong deu cho ca
N/V/F, khong uu tien N). Day la diem khac biet co ban so voi moi bien the
da thu truoc do (noi RR vao embedding dung chung cho phep no anh huong TU DO
len moi logit, bao gom ca logit N va F).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr_bilstm import AttentionPooling, ECGFeatureExtractor, RLSTMLoss
from hmr_dsta_lite import BiDSTALite

NUM_CLASSES = 4  # N,S,V,F (Q loai bo, giong cac thi nghiem truoc)
N_IDX, S_IDX, V_IDX, F_IDX = 0, 1, 2, 3


class DSTACSBoost(nn.Module):
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
        rr_dim: int = 3,
        use_sboost: bool = True,
    ):
        super().__init__()
        self.use_sboost = use_sboost
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

        # Classifier CHINH -- giong HET "C khong RR", KHONG nhan RR
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_CLASSES),
        )

        # Nhanh S-boost DOC LAP -- chi nhan RR, khong nhan h_embed (tranh
        # canh tranh gradient nguoc ve backbone -- backbone chi duoc toi uu
        # boi classifier chinh, dung nhu "C khong RR")
        self.rr_boost_head = nn.Sequential(
            nn.Linear(rr_dim, 16), nn.GELU(), nn.Linear(16, 1),
        )

    def forward(self, x, rr, return_internals=False):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features)
        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_embed = self.dropout(self.layer_norm(h_pooled))

        logits = self.classifier(h_embed)

        if self.use_sboost:
            s_boost = F.softplus(self.rr_boost_head(rr)).squeeze(-1)  # (B,), >=0
            boost_onehot = torch.zeros_like(logits)
            boost_onehot[:, S_IDX] = s_boost
            logits = logits + boost_onehot
        else:
            s_boost = torch.zeros(x.size(0), device=x.device)

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "s_boost": s_boost,
                             "attention_weights": attn_weights}
        return logits


def class_weights_from(y_train, num_classes=NUM_CLASSES):
    import numpy as np
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
    B, T, D = 16, 187, 1
    x = torch.randn(B, T, D)
    rr = torch.randn(B, 3)
    y = torch.randint(0, 4, (B,))

    model = DSTACSBoost(input_size=D, hidden_size=96, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    logits, internals = model(x, rr, return_internals=True)
    print(f"logits: {tuple(logits.shape)}  s_boost: {tuple(internals['s_boost'].shape)}  "
          f"s_boost range=[{internals['s_boost'].min().item():.4f}, {internals['s_boost'].max().item():.4f}]")
    assert (internals["s_boost"] >= 0).all(), "S-BOOST BI AM -- SAI!"

    cw = torch.tensor([0.5, 5.0, 2.0, 10.0])
    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=cw, use_focal=True, focal_gamma=1.5)
    loss, _ = criterion(logits, y, internals["r_fwd"], internals["r_bwd"])
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print(f"Params khong nhan grad: {n_none}  |  grad=NaN: {n_nan}")

    # Kiem tra co the KHONG the anh huong logit N/V/F: dat rr_boost_head ve
    # gia tri lon, xem logit N/V/F co doi khong (chi logit S doi).
    # PHAI dung model.eval() -- neu khong dropout ngau nhien giua 2 lan
    # forward se gia mao thanh "sai lech" (bay da gap voi alpha_head truoc do).
    model.eval()
    with torch.no_grad():
        logits_before, _ = model(x, rr, return_internals=True)
        for p in model.rr_boost_head.parameters():
            p.add_(10.0)  # ep s_boost tang manh
        logits_after, internals2 = model(x, rr, return_internals=True)
    diff = (logits_after - logits_before)
    diff_non_s = diff[:, [N_IDX, V_IDX, F_IDX]].abs().max().item()
    diff_s = diff[:, S_IDX].abs().max().item()
    print(f"Sau khi ep s_boost tang: |thay doi logit N/V/F| toi da = {diff_non_s:.8f} (ky vong = 0)")
    print(f"                          |thay doi logit S| toi da = {diff_s:.4f} (ky vong > 0)")

    ok = n_none == 0 and n_nan == 0 and diff_non_s < 1e-6 and diff_s > 1e-6
    print("Smoke test PASSED" if ok else "Smoke test FAILED")


if __name__ == "__main__":
    _smoke_test()
