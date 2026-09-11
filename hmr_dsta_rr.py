"""
ResMemECG + DSTA-Lite + RR-interval TRONG context z_t (không phải ghép hời hợt)
================================================================================

Khác với `hmr_bilstm_rr.py` (RR chỉ nối vào ngay trước classifier cuối, sau
khi toàn bộ CNN + BiLSTM + attention pooling đã chạy xong -- cổng gate KHÔNG
BAO GIỜ thấy RR): file này đưa RR-interval (3 đặc trưng: pre_rr_ratio,
post_rr_ratio, local_rr_ratio -- xem validation/preprocess_aami_rr.py) vào
thẳng vector ngữ cảnh z_t của DSTALiteGate, thứ điều khiển gamma_s/gamma_d/
lambda (mức độ chú ý dồn vào từng cạnh transition) -- TẠI MỌI BƯỚC THỜI GIAN
trong 187 bước, không chỉ 1 lần ở cuối.

Vì sao đây là điểm ghép hợp lý (không phải chỗ khác trong model):
  - RR là 3 số hằng số cho cả beat (không đổi theo t) -- ghép vào input thô
    x_t (187 điểm sóng) sẽ không hợp lý vì CNN dùng conv1d trượt theo thời
    gian, không có "kênh hằng số" tự nhiên để đặt RR vào.
  - z_t đã là 1 vector tổng hợp trạng thái tại bước t, dùng để lái các trọng
    số attention -- đây đúng nghĩa "ngữ cảnh" mà 1 tín hiệu tĩnh (RR) nên
    tham gia vào, để attention có thể "biết" nhịp này có sớm bất thường hay
    không khi quyết định trọng số cho từng cạnh, ở MỌI bước, không chỉ dùng
    RR làm 1 input phụ trợ ở tầng classifier cuối như bản cũ.
  - context_norm (LayerNorm) áp SAU khi trộn RR với phần ngữ cảnh động (biến
    đổi theo t) -- nên ảnh hưởng tương đối của RR không cố định giữa các
    bước, nó tương tác qua trọng số học được của context_proj.

Mọi phần khác giữ nguyên 100% so với `hmr_dsta_lite.py` (biến thể
C_perdim_5edge_softmax: blend_mode="per_dim", extra_edge=True,
gamma_mode="softmax" -- biến thể tốt nhất trong ablation) -- chỉ thêm rr_dim
optional (mặc định 0 = hành vi y hệt bản gốc, backward-compatible).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr_bilstm import ECGFeatureExtractor, AttentionPooling


class DSTALiteGateRR(nn.Module):
    STATE_NAMES = ("h_prev", "c_prev", "h_proposal", "c_proposal")
    QUERY_INDICES = (2, 3)

    def __init__(
        self,
        hidden_size: int,
        attn_dim: int = 24,
        num_heads: int = 2,
        rank: int = 4,
        context_dim: Optional[int] = None,
        extra_edge: bool = True,
        gamma_mode: str = "softmax",
        rr_dim: int = 0,
    ) -> None:
        super().__init__()
        if gamma_mode not in ("softmax", "sigmoid"):
            raise ValueError(f"gamma_mode phai la 'softmax' hoac 'sigmoid', nhan: {gamma_mode}")
        self.extra_edge = extra_edge
        self.gamma_mode = gamma_mode
        self.rr_dim = rr_dim

        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim={attn_dim} phai chia het cho num_heads={num_heads}")

        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.rank = rank
        self.head_dim = attn_dim // num_heads
        self.context_dim = context_dim or attn_dim

        self.hidden_state_proj = nn.Linear(hidden_size, attn_dim, bias=False)
        self.cell_state_proj = nn.Linear(hidden_size, attn_dim, bias=False)
        self.role_embedding = nn.Parameter(torch.zeros(2, attn_dim))
        self.time_embedding = nn.Parameter(torch.zeros(2, attn_dim))
        self.token_norm = nn.LayerNorm(attn_dim)

        self.q_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(attn_dim, attn_dim, bias=False)

        # --- DIEM GHEP RR: + rr_dim vao context_input_dim ---
        context_input_dim = 4 * attn_dim + 2 * hidden_size + rr_dim
        self.context_proj = nn.Linear(context_input_dim, self.context_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)

        self.gamma_s_proj = nn.Linear(self.context_dim, num_heads * rank)
        self.gamma_d_proj = nn.Linear(self.context_dim, num_heads * rank)
        self.lambda_proj = nn.Linear(self.context_dim, num_heads)

        self.factor_a = nn.Parameter(torch.empty(num_heads, rank, self.head_dim))
        self.factor_b = nn.Parameter(torch.empty(num_heads, rank, self.head_dim))
        self.factor_c = nn.Parameter(torch.empty(num_heads, rank, self.head_dim))

        src_list = [0, 1, 0, 2]
        dst_list = [2, 3, 1, 3]
        if extra_edge:
            src_list.append(3)
            dst_list.append(2)
        self.register_buffer("transition_sources", torch.tensor(src_list, dtype=torch.long), persistent=False)
        self.register_buffer("transition_targets", torch.tensor(dst_list, dtype=torch.long), persistent=False)
        self.num_transitions = len(src_list)

        self.value_proj = nn.Linear(3 * attn_dim, attn_dim, bias=False)
        self.output_proj = nn.Linear(2 * attn_dim, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        scale = 1.0 / math.sqrt(self.head_dim)
        nn.init.normal_(self.factor_a, mean=0.0, std=scale)
        nn.init.normal_(self.factor_b, mean=0.0, std=scale)
        nn.init.normal_(self.factor_c, mean=0.0, std=scale)

    def _build_state_tokens(self, h_prev, c_prev, h_proposal, c_proposal):
        h_prev_tok = self.hidden_state_proj(h_prev) + self.role_embedding[0] + self.time_embedding[0]
        c_prev_tok = self.cell_state_proj(c_prev) + self.role_embedding[1] + self.time_embedding[0]
        h_prop_tok = self.hidden_state_proj(h_proposal) + self.role_embedding[0] + self.time_embedding[1]
        c_prop_tok = self.cell_state_proj(c_proposal) + self.role_embedding[1] + self.time_embedding[1]
        states = torch.stack([h_prev_tok, c_prev_tok, h_prop_tok, c_prop_tok], dim=1)
        return self.token_norm(states)

    def _build_context(self, states, h_prev, c_prev, h_proposal, c_proposal, rr=None):
        B = states.size(0)
        flat_states = states.reshape(B, -1)
        delta_h = h_proposal - h_prev
        delta_c = c_proposal - c_prev
        parts = [flat_states, delta_h, delta_c]
        if self.rr_dim > 0:
            if rr is None:
                raise ValueError("DSTALiteGateRR duoc khoi tao voi rr_dim>0 nhung forward khong nhan rr")
            parts.append(rr)  # (B, rr_dim) -- hang so theo t, tham gia vao z_t o MOI buoc
        context_input = torch.cat(parts, dim=-1)
        return self.context_norm(self.context_proj(context_input))

    def forward(
        self,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        h_proposal: torch.Tensor,
        c_proposal: torch.Tensor,
        rr: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B = h_prev.size(0)
        H, R, Dh = self.num_heads, self.rank, self.head_dim

        states = self._build_state_tokens(h_prev, c_prev, h_proposal, c_proposal)

        query_states = states[:, self.QUERY_INDICES, :]
        q = self.q_proj(query_states).view(B, 2, H, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(states).view(B, 4, H, Dh).permute(0, 2, 1, 3)

        z_t = self._build_context(states, h_prev, c_prev, h_proposal, c_proposal, rr=rr)

        if self.gamma_mode == "softmax":
            gamma_s = F.softmax(self.gamma_s_proj(z_t).view(B, H, R), dim=-1)
            gamma_d = F.softmax(self.gamma_d_proj(z_t).view(B, H, R), dim=-1)
        else:
            gamma_s = torch.sigmoid(self.gamma_s_proj(z_t).view(B, H, R))
            gamma_d = torch.sigmoid(self.gamma_d_proj(z_t).view(B, H, R))
        directional_mix = torch.sigmoid(self.lambda_proj(z_t))

        chi = torch.einsum("bhid,hrd->bhir", q, self.factor_a)
        xi = torch.einsum("bhjd,hrd->bhjr", k, self.factor_b)
        zeta = torch.einsum("bhjd,hrd->bhjr", k, self.factor_c)

        src, dst = self.transition_sources, self.transition_targets
        xi_src, xi_dst = xi[:, :, src, :], xi[:, :, dst, :]
        zeta_src, zeta_dst = zeta[:, :, src, :], zeta[:, :, dst, :]

        symmetric = 0.5 * (xi_src * zeta_dst + xi_dst * zeta_src)
        directional = 0.5 * (xi_src * zeta_dst - xi_dst * zeta_src)

        relational = (
            gamma_s.unsqueeze(2) * symmetric
            + directional_mix.unsqueeze(-1).unsqueeze(-1) * gamma_d.unsqueeze(2) * directional
        )

        scores = torch.einsum("bhir,bher->bhie", chi, relational) / math.sqrt(max(R, 1))
        attention = F.softmax(scores, dim=-1)

        state_src = states[:, src, :]
        state_dst = states[:, dst, :]
        interaction = state_src * state_dst
        transition_features = torch.cat([state_src, state_dst, interaction], dim=-1)
        values = self.value_proj(transition_features).view(B, -1, H, Dh).permute(0, 2, 1, 3)

        outputs = torch.einsum("bhie,bhed->bhid", attention, values)
        h_out, c_out = outputs[:, :, 0, :], outputs[:, :, 1, :]
        current_out = torch.cat([h_out, c_out], dim=-1).reshape(B, 2 * self.attn_dim)

        gate_logits = self.output_norm(self.output_proj(current_out))
        r_t = torch.sigmoid(gate_logits)

        diagnostics: Dict[str, torch.Tensor] = {}
        if return_diagnostics:
            diagnostics = {"attention": attention, "gamma_s": gamma_s, "gamma_d": gamma_d,
                            "directional_mix": directional_mix}
        return r_t, diagnostics


class PerDimStreamBlend(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: Optional[int] = None) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(hidden_size // 2, 8)
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 2 * hidden_size),
        )

    def forward(self, c_keep_norm: torch.Tensor, c_add_norm: torch.Tensor) -> torch.Tensor:
        B = c_keep_norm.size(0)
        stream_context = torch.cat([c_keep_norm, c_add_norm, c_keep_norm * c_add_norm], dim=-1)
        logits = self.mlp(stream_context).view(B, self.hidden_size, 2)
        weights = F.softmax(logits, dim=-1)
        return weights[..., 0], weights[..., 1]


class DSTALiteCellRR(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        attn_dim: int = 24,
        num_heads: int = 2,
        rank: int = 4,
        context_dim: Optional[int] = None,
        extra_edge: bool = True,
        gamma_mode: str = "softmax",
        blend_mode: str = "per_dim",
        blend_bottleneck_dim: Optional[int] = None,
        rr_dim: int = 0,
    ) -> None:
        super().__init__()
        if blend_mode not in ("scalar", "per_dim"):
            raise ValueError(f"blend_mode phai la 'scalar' hoac 'per_dim', nhan: {blend_mode}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blend_mode = blend_mode

        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)

        self.dsta_gate = DSTALiteGateRR(
            hidden_size=hidden_size, attn_dim=attn_dim, num_heads=num_heads,
            rank=rank, context_dim=context_dim, extra_edge=extra_edge, gamma_mode=gamma_mode,
            rr_dim=rr_dim,
        )

        if blend_mode == "scalar":
            self.W_alpha = nn.Linear(2 * hidden_size, 2)
        else:
            self.stream_blend = PerDimStreamBlend(hidden_size, bottleneck_dim=blend_bottleneck_dim)

        self.W_beta = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.last_r_t = None
        self.last_c_keep = None
        self.last_c_add = None

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2 and "dsta_gate" not in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name and "dsta_gate" not in name:
                nn.init.zeros_(param)
                if "W_x" in name or "W_h" in name:
                    n = param.size(0) // 4
                    with torch.no_grad():
                        param[n:2 * n].fill_(1.0)
                if "W_beta.bias" in name:
                    with torch.no_grad():
                        param.fill_(-5.0)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
                rr: Optional[torch.Tensor] = None):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_size, dim=1)
        i_t, f_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        c_proposal = f_t * c_prev + i_t * g_t
        h_proposal = o_t * torch.tanh(c_proposal)

        r_t, _ = self.dsta_gate(h_prev, c_prev, h_proposal, c_proposal, rr=rr)

        c_keep = (f_t + r_t * (1.0 - f_t)) * c_prev
        c_add = (1.0 - r_t) * (i_t * g_t)

        c_keep_norm = F.layer_norm(c_keep, (c_keep.shape[-1],))
        c_add_norm = F.layer_norm(c_add, (c_add.shape[-1],))
        if self.blend_mode == "scalar":
            combined = torch.cat([c_keep_norm, c_add_norm], dim=-1)
            alpha = F.softmax(self.W_alpha(combined), dim=-1)
            keep_weight, add_weight = alpha[:, 0:1], alpha[:, 1:2]
        else:
            keep_weight, add_weight = self.stream_blend(c_keep_norm, c_add_norm)
        c_rmc = keep_weight * c_keep + add_weight * c_add

        beta = torch.sigmoid(self.W_beta(h_prev))
        c_t = beta * c_rmc + (1.0 - beta) * c_proposal

        h_t = self.dropout(o_t * torch.tanh(c_t))

        self.last_r_t = r_t
        self.last_c_keep = c_keep
        self.last_c_add = c_add

        return h_t, c_t


class DSTALiteLayerRR(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, **dsta_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DSTALiteCellRR(input_size, hidden_size, dropout=dropout, **dsta_kwargs)

    def forward(self, x, h0=None, c0=None, rr=None):
        B, T, _ = x.size()
        device = x.device
        h = torch.zeros(B, self.hidden_size, device=device) if h0 is None else h0
        c = torch.zeros(B, self.hidden_size, device=device) if c0 is None else c0

        h_outputs, r_outputs = [], []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c, rr=rr)  # rr hang so, dua lai moi buoc t
            h_outputs.append(h)
            r_outputs.append(self.cell.last_r_t)

        h_seq = torch.stack(h_outputs, dim=1)
        r_seq = torch.stack(r_outputs, dim=1)
        return h_seq, (h, c), r_seq


class BiDSTALiteRR(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1, **dsta_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = input_size if i == 0 else 2 * hidden_size
            self.fwd_layers.append(DSTALiteLayerRR(layer_in, hidden_size, dropout=dropout, **dsta_kwargs))
            self.bwd_layers.append(DSTALiteLayerRR(layer_in, hidden_size, dropout=dropout, **dsta_kwargs))

        self.inter_dropout = nn.Dropout(dropout) if num_layers > 1 else None

    def forward(self, x, rr=None):
        h_seq = x
        r_fwd = r_bwd = None
        for i in range(self.num_layers):
            h_f, _, r_fwd = self.fwd_layers[i](h_seq, rr=rr)
            x_rev = torch.flip(h_seq, dims=[1])
            h_b, _, r_bwd = self.bwd_layers[i](x_rev, rr=rr)
            h_b = torch.flip(h_b, dims=[1])
            r_bwd = torch.flip(r_bwd, dims=[1])

            h_seq = torch.cat([h_f, h_b], dim=-1)
            if self.inter_dropout is not None and i < self.num_layers - 1:
                h_seq = self.inter_dropout(h_seq)

        return h_seq, r_fwd, r_bwd


class DSTALiteClassifierRR(nn.Module):
    """
    Bien the C (per_dim, extra_edge=True, gamma_mode="softmax" -- tot nhat
    trong ablation) + RR-interval dua vao z_t o moi buoc thoi gian.
    rr_dim=0 (mac dinh) => forward khong can rr, hanh vi giong het
    DSTALiteClassifier goc (backward-compatible, khong anh huong job dang
    chay dung hmr_dsta_lite.py).
    """

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
        extra_edge: bool = True,
        gamma_mode: str = "softmax",
        blend_mode: str = "per_dim",
        blend_bottleneck_dim: Optional[int] = None,
        rr_dim: int = 3,
    ):
        super().__init__()
        self.rr_dim = rr_dim
        self.cnn = ECGFeatureExtractor(
            input_channels=input_size, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )
        self.birlstm = BiDSTALiteRR(
            cnn_out_channels, hidden_size, num_layers=num_layers, dropout=dropout,
            attn_dim=attn_dim, num_heads=num_heads, rank=rank,
            extra_edge=extra_edge, gamma_mode=gamma_mode,
            blend_mode=blend_mode, blend_bottleneck_dim=blend_bottleneck_dim,
            rr_dim=rr_dim,
        )
        self.attention_pool = AttentionPooling(2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, rr=None, return_internals=False):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features, rr=rr)

        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_pooled = self.dropout(self.layer_norm(h_pooled))
        logits = self.classifier(h_pooled)

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "attention_weights": attn_weights}
        return logits


# =============================================================================
#  SMOKE TEST -- CPU, dummy data, KHONG dung GPU (dang ban job confusion-margin)
# =============================================================================
def _smoke_test() -> None:
    from hmr_bilstm import RLSTMLoss

    torch.manual_seed(42)
    B, T, D, NUM_CLASSES, RR_DIM = 16, 187, 1, 5, 3
    HIDDEN = 96

    x = torch.randn(B, T, D)
    rr = torch.randn(B, RR_DIM)  # gia lap [pre_rr_ratio, post_rr_ratio, local_rr_ratio]
    y = torch.randint(0, NUM_CLASSES, (B,))

    print("=" * 70)
    print("1) rr_dim=0 -- phai giong het DSTALiteClassifier goc (backward-compat)")
    print("=" * 70)
    model_no_rr = DSTALiteClassifierRR(input_size=D, hidden_size=HIDDEN, num_classes=NUM_CLASSES,
                                        num_layers=2, rr_dim=0)
    n_no_rr = sum(p.numel() for p in model_no_rr.parameters())
    print(f"  Params (rr_dim=0): {n_no_rr:,}  (doi chieu voi bien the C goc: 603,118)")
    logits0 = model_no_rr(x, rr=None)
    assert logits0.shape == (B, NUM_CLASSES)
    print(f"  Forward khong rr: OK, logits shape {tuple(logits0.shape)}")

    print("\n" + "=" * 70)
    print("2) rr_dim=3 -- RR dua vao z_t moi buoc thoi gian")
    print("=" * 70)
    model_rr = DSTALiteClassifierRR(input_size=D, hidden_size=HIDDEN, num_classes=NUM_CLASSES,
                                     num_layers=2, rr_dim=RR_DIM)
    n_rr = sum(p.numel() for p in model_rr.parameters())
    print(f"  Params (rr_dim=3): {n_rr:,}  (chenh lech vs khong RR: +{n_rr - n_no_rr:,}"
          f" -- chi la context_proj to hon 1 chut, khong dang ke)")

    logits, internals = model_rr(x, rr=rr, return_internals=True)
    assert logits.shape == (B, NUM_CLASSES)
    assert internals["r_fwd"].shape[0] == B and internals["r_fwd"].shape[2] == HIDDEN
    print(f"  Forward co rr: OK, logits shape {tuple(logits.shape)}, r_fwd shape {tuple(internals['r_fwd'].shape)}"
          f"  (T={T} -> {internals['r_fwd'].shape[1]} sau 2 lan MaxPool1d(2) cua CNN)")

    print("\n" + "=" * 70)
    print("3) Gradient check -- rr phai nhan duoc gradient qua context_proj")
    print("=" * 70)
    rr_req = rr.clone().requires_grad_(True)
    logits2, internals2 = model_rr(x, rr=rr_req, return_internals=True)
    criterion = RLSTMLoss(lambda_smooth=0.003)
    loss, comp = criterion(logits2, y, internals2["r_fwd"], internals2["r_bwd"])
    loss.backward()

    print(f"  Loss: {loss.item():.4f} (task={comp['task'].item():.4f}, smooth={comp['smooth'].item():.4f})")
    print(f"  rr.grad la None? {rr_req.grad is None}")
    if rr_req.grad is not None:
        print(f"  rr.grad abs sum: {rr_req.grad.abs().sum().item():.6f}  (>0 => RR thuc su anh huong loss)")

    n_none_grad = sum(1 for p in model_rr.parameters() if p.requires_grad and p.grad is None)
    n_nan_grad = sum(1 for p in model_rr.parameters() if p.grad is not None and torch.isnan(p.grad).any())
    print(f"  Params khong nhan grad: {n_none_grad}  |  Params grad=NaN: {n_nan_grad}")
    status = "OK" if (rr_req.grad is not None and rr_req.grad.abs().sum() > 0
                      and n_none_grad == 0 and n_nan_grad == 0) else "LOI"
    print(f"  => {status}")

    print("\n" + "=" * 70)
    print("4) Kiem tra RR thuc su lam thay doi r_t (khong bi 'nuot' boi LayerNorm)")
    print("=" * 70)
    model_rr.eval()
    with torch.no_grad():
        _, internals_a = model_rr(x, rr=rr, return_internals=True)
        rr_diff = rr + 5.0  # doi manh gia tri RR
        _, internals_b = model_rr(x, rr=rr_diff, return_internals=True)
    r_diff = (internals_a["r_fwd"] - internals_b["r_fwd"]).abs().mean().item()
    print(f"  |r_fwd(rr) - r_fwd(rr+5)| trung binh: {r_diff:.6f}  (>0 ro rang => RR thuc su lam doi gate)")


if __name__ == "__main__":
    _smoke_test()
