"""
Kien truc cay phan tang MOI, dung theo dung can cu y van (khong con doan mo):

  Tang 1: N vs {S,V,F}   (P(N) = sigmoid, dac trung = h_embed + template-corr)
  Tang 2: S vs V vs F    (3-lop, dac trung = h_embed + RR-ratio(3))

Ly do thiet ke (xem bao cao da tong hop tu y van trong hoi thoai):
  - N tach truoc vi day la thiet ke chuan da co trong y van cho bai toan AAMI
    5-lop (cascade N-first, ~89% nhip la N, giai quyet duoc phan lon o tang 1
    voi do tin cay cao truoc khi dua phan con lai vao bo phan loai chi tiet).
  - Dac trung template-corr (tuong quan hinh dang voi nhip N chuan) dat DUNG
    cho tang N-vs-rest -- vi ban chat no do "giong N hay khong", KHONG phai
    dac trung tot de phan biet NOI BO S/V/F (da kiem chung: nhet chung vao 1
    classifier 5-lop lam F sup do vi canh tranh gradient voi RR).
  - RR-ratio (thoi gian) dat o tang 2 vi S la van de THOI GIAN (nhip den som,
    P-wave bat thuong), khong phai hinh dang -- y van xac nhan QRS cua S
    "khong rong, hinh dang giong het nhip binh thuong".
  - F khong con la 1 "lop co dac trung rieng" o tang 2 nay -- ve nguyen tac,
    F la nhip HOA TRON that su giua N va V (khong phai doan mo), nen viec
    tach no khoi V trong pham vi {S,V,F} van la bai toan kho, se can bo sung
    sau bang huong tong hop-alpha (dsta_c_synth_fusion.py) neu tang 2 van yeu
    cho F -- CHUA lam trong file nay, de tranh doi nhieu bien 1 luc.

Huan luyen theo dung chain rule xac suat (hierarchical cross-entropy), KHONG
train 2 dau doc lap roi ghep hau ky:
    P(N)  = sigmoid(logit_n)
    P(S)  = (1-P(N)) * softmax(logits_svf)[S]
    P(V)  = (1-P(N)) * softmax(logits_svf)[V]
    P(F)  = (1-P(N)) * softmax(logits_svf)[F]
  loss = -log P(y_true) -- neu y_true=N chi co gradient qua tang 1 (tang 2
  KHONG nhan gradient tu mau N, tu dong dung nhu the theo dung chain rule,
  khong can mask thu cong).
  + 1 so hang phu BCE rieng cho tang 1 (trong so lambda_aux) de "phat nang
  hon loi o tang cao" nhu y van khuyen nghi (parent probability on dinh hoa
  du doan con).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr_bilstm import AttentionPooling, ECGFeatureExtractor

try:
    from hmr_dsta_lite import BiDSTALite
except ImportError:
    BiDSTALite = None


N_IDX, S_IDX, V_IDX, F_IDX, Q_IDX = 0, 1, 2, 3, 4
SVF_ORDER = [S_IDX, V_IDX, F_IDX]  # thu tu index trong logits_svf: 0=S,1=V,2=F


class DSTAHierNFirst(nn.Module):
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
        template_dim: int = 1,
        rr_dim: int = 3,
        detach_stage2_embed: bool = False,
    ):
        super().__init__()
        self.detach_stage2_embed = detach_stage2_embed
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

        self.template_encoder = nn.Sequential(
            nn.Linear(template_dim, 16), nn.GELU(), nn.Linear(16, 16),
        )
        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_dim, 32), nn.GELU(), nn.Linear(32, 32),
        )

        self.stage1_head = nn.Sequential(
            nn.Linear(embed_dim + 16, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.stage2_head = nn.Sequential(
            nn.Linear(embed_dim + 32, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),  # thu tu: S, V, F
        )

    def forward(self, x, template_feat, rr_feat, return_internals=False):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features)
        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_embed = self.dropout(self.layer_norm(h_pooled))

        t_feat = self.template_encoder(template_feat)
        rr_feat_enc = self.rr_encoder(rr_feat)

        h_embed_s2 = h_embed.detach() if self.detach_stage2_embed else h_embed
        logit_n = self.stage1_head(torch.cat([h_embed, t_feat], dim=-1)).squeeze(-1)
        logits_svf = self.stage2_head(torch.cat([h_embed_s2, rr_feat_enc], dim=-1))

        if return_internals:
            return logit_n, logits_svf, {"r_fwd": r_fwd, "r_bwd": r_bwd, "attention_weights": attn_weights}
        return logit_n, logits_svf


def hier_log_probs(logit_n, logits_svf):
    """Tra ve log P(N), log P(S), log P(V), log P(F) theo dung chain rule."""
    log_p_n = F.logsigmoid(logit_n)
    log_p_notn = F.logsigmoid(-logit_n)
    log_softmax_svf = F.log_softmax(logits_svf, dim=-1)
    log_p_s = log_p_notn + log_softmax_svf[:, 0]
    log_p_v = log_p_notn + log_softmax_svf[:, 1]
    log_p_f = log_p_notn + log_softmax_svf[:, 2]
    return log_p_n, log_p_s, log_p_v, log_p_f


def hier_predict(logit_n, logits_svf):
    """Tra ve nhan du doan 4-lop (0=N,1=S,2=V,3=F) tu argmax P(class)."""
    log_p_n, log_p_s, log_p_v, log_p_f = hier_log_probs(logit_n, logits_svf)
    stacked = torch.stack([log_p_n, log_p_s, log_p_v, log_p_f], dim=-1)  # da la log, argmax khong doi
    return stacked.argmax(-1)


class HierLoss(nn.Module):
    """Hierarchical cross-entropy (chain rule) + focal modulation + BCE phu
    cho tang 1 (phat nang hon loi phan loai N-vs-rest, dung nhu y van
    khuyen nghi) + smoothness loss (giu dung quy uoc cu)."""

    def __init__(self, class_weights_4=None, focal_gamma=1.5, lambda_aux_stage1=0.3,
                 lambda_smooth=0.003, bce_pos_weight=None):
        super().__init__()
        self.register_buffer("class_weights_4", class_weights_4 if class_weights_4 is not None
                              else torch.ones(4))
        self.focal_gamma = focal_gamma
        self.lambda_aux_stage1 = lambda_aux_stage1
        self.lambda_smooth = lambda_smooth
        self.bce_pos_weight = bce_pos_weight

    @staticmethod
    def _smooth(r_seq):
        if r_seq.size(1) < 2:
            return r_seq.sum() * 0.0
        diff = r_seq[:, 1:, :] - r_seq[:, :-1, :]
        return (diff ** 2).sum(dim=-1).mean()

    def forward(self, logit_n, logits_svf, y4, r_fwd, r_bwd):
        """y4: nhan da anh xa ve 4 lop (0=N,1=S,2=V,3=F), KHONG con Q."""
        log_p_n, log_p_s, log_p_v, log_p_f = hier_log_probs(logit_n, logits_svf)
        log_p_stack = torch.stack([log_p_n, log_p_s, log_p_v, log_p_f], dim=-1)  # (B,4)

        logp_true = log_p_stack.gather(1, y4.unsqueeze(1)).squeeze(1)  # (B,)
        pt = logp_true.exp().clamp(min=1e-6, max=1.0)
        focal_w = (1.0 - pt) ** self.focal_gamma
        cw = self.class_weights_4[y4]
        l_main = (-focal_w * cw * logp_true).mean()

        y_is_n = (y4 == N_IDX).float()
        l_aux_stage1 = F.binary_cross_entropy_with_logits(
            logit_n, y_is_n, pos_weight=self.bce_pos_weight,
        )

        l_smooth = (self._smooth(r_fwd) + self._smooth(r_bwd)) / 2.0

        total = l_main + self.lambda_aux_stage1 * l_aux_stage1 + self.lambda_smooth * l_smooth
        comps = {
            "main": l_main.detach(), "aux_stage1": l_aux_stage1.detach(),
            "smooth": l_smooth.detach(), "total": total.detach(),
        }
        return total, comps


# =============================================================================
#  SMOKE TEST -- CPU, dummy data
# =============================================================================
def _smoke_test():
    torch.manual_seed(42)
    B, T, D = 16, 187, 1
    x = torch.randn(B, T, D)
    template_feat = torch.randn(B, 1)
    rr_feat = torch.randn(B, 3)
    y4 = torch.randint(0, 4, (B,))

    model = DSTAHierNFirst(input_size=D, hidden_size=96, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    logit_n, logits_svf, internals = model(x, template_feat, rr_feat, return_internals=True)
    print(f"logit_n: {tuple(logit_n.shape)}  logits_svf: {tuple(logits_svf.shape)}")

    log_p_n, log_p_s, log_p_v, log_p_f = hier_log_probs(logit_n, logits_svf)
    probs_sum = (log_p_n.exp() + log_p_s.exp() + log_p_v.exp() + log_p_f.exp())
    print(f"tong xac suat 4 lop (phai ~1.0): min={probs_sum.min().item():.6f} max={probs_sum.max().item():.6f}")
    assert torch.allclose(probs_sum, torch.ones(B), atol=1e-4), "XAC SUAT KHONG CHUAN HOA DUNG!"

    preds = hier_predict(logit_n, logits_svf)
    print(f"preds: {preds.tolist()}")

    cw = torch.tensor([0.5, 5.0, 2.0, 10.0])
    criterion = HierLoss(class_weights_4=cw, lambda_aux_stage1=0.3)
    loss, comps = criterion(logit_n, logits_svf, y4, internals["r_fwd"], internals["r_bwd"])
    print("loss components: " + "  ".join(f"{k}={v.item():.4f}" for k, v in comps.items()))
    loss.backward()

    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print(f"Params khong nhan grad: {n_none}  |  grad=NaN: {n_nan}")

    # Kiem tra dung chain-rule: mau N KHONG duoc lam anh huong gradient len stage2_head
    model.zero_grad()
    y4_all_n = torch.zeros(B, dtype=torch.long)  # gia lap TOAN BO la N
    logit_n2, logits_svf2, internals2 = model(x, template_feat, rr_feat, return_internals=True)
    loss2, _ = criterion(logit_n2, logits_svf2, y4_all_n, internals2["r_fwd"], internals2["r_bwd"])
    loss2.backward()
    stage2_grad_norm = sum(p.grad.abs().sum().item() for p in model.stage2_head.parameters() if p.grad is not None)
    print(f"Gradient stage2_head khi TOAN BO mau la N (ky vong ~0): {stage2_grad_norm:.8f}")

    ok = (n_none == 0 and n_nan == 0 and stage2_grad_norm < 1e-5)
    print("Smoke test PASSED" if ok else "Smoke test FAILED")


if __name__ == "__main__":
    _smoke_test()
