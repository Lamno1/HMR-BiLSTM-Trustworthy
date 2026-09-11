"""
ResMemECG + DSTA-Lite: trimmed Directional State-Transition Triadic Attention
================================================================================

Chỉ thay MỘT chỗ duy nhất so với hmr_bilstm.py: cách tính residual gate r_t.
Toàn bộ phần còn lại (4 cổng LSTM chuẩn, keep/add softmax blend, hybrid beta
blend, CNN frontend, attention pooling, classifier) GIỮ NGUYÊN 100% so với
hmr_bilstm.py -- để việc so sánh sau này là single-variable (chỉ đổi r_t).

r_t cũ (hmr_bilstm.py):
    m_t = (W_c c_{t-1}) ⊙ (W_h h_{t-1})
    r_t = sigmoid(LayerNorm(W_c c_{t-1} + W_h h_{t-1} + m_t))
    -> chỉ nhìn quá khứ (c_{t-1}, h_{t-1}), chưa từng thấy đề xuất của LSTM.

r_t mới (DSTA-Lite, DSTALiteGate):
    r_t = DSTA[(h_{t-1}, c_{t-1}) -> (h_bar_t, c_bar_t)]
    -> gate nhìn thấy cả trạng thái đề xuất (h_bar_t, c_bar_t) trước khi quyết
       định giữ/thêm, qua attention có hướng trên 4 transition (thay vì 8) với
       phân rã symmetric/antisymmetric (Sjk, Djk).

Hai điểm cắt so với bản DSTA đầy đủ (TEN_EM_LA_GI.py), không đánh đổi biểu
diễn -- phần bị cắt vốn dĩ dư thừa/không dùng đến trong bản gốc:
  1. Query chỉ tính cho 2 state (h_bar_t, c_bar_t) thay vì cả 4 -- bản gốc
     tính Q cho cả 4 state rồi CHỈ DÙNG output của 2 state này, còn lại bỏ đi.
     -> cắt ~50% compute của attention mỗi bước thời gian.
  2. Transition-value bỏ thành phần delta = s_k - s_j (tổ hợp tuyến tính của
     s_j, s_k đã có sẵn trong phép nối, nên 1 lớp Linear vốn đã tự học được
     nó) -- chỉ giữ [s_j, s_k, s_j⊙s_k] (phần phi tuyến, không suy ra được
     từ tổ hợp tuyến tính).

Cộng thêm cắt "phần vỏ": attn_dim/rank/num_heads nhỏ, tập transition rút
từ 8 xuống 4 cạnh có cơ sở vật lý rõ ràng (2 cạnh cũ->đề xuất, 1 cạnh
hướng của m_t gốc, 1 cạnh giữa 2 đề xuất) -- để ngân sách tham số nằm
gần 505,038 (baseline hiện tại) thay vì phình lên >1M như DSTA đầy đủ.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr_bilstm import ECGFeatureExtractor, AttentionPooling


# =============================================================================
#  DSTA-LITE GATE -- thay thế cho phần tính r_t (m_t + LayerNorm) cũ
# =============================================================================
class DSTALiteGate(nn.Module):
    """
    4 recurrent state: h_prev(0), c_prev(1), h_proposal(2), c_proposal(3).
    Query CHỈ tính cho index {2, 3} (h_proposal, c_proposal) -- đúng phần duy
    nhất được dùng ở output, nên không cần tính Q cho index {0, 1} rồi bỏ đi.
    Key/Value vẫn phủ đủ 4 state (transition vẫn cần state cũ làm nguồn).

    4 transition có hướng (thay vì 8 của bản đầy đủ):
        0 -> 2 : h_prev      -> h_proposal
        1 -> 3 : c_prev      -> c_proposal
        0 -> 1 : h_prev      -> c_prev       (đúng hướng của m_t gốc)
        2 -> 3 : h_proposal  -> c_proposal
    """

    STATE_NAMES = ("h_prev", "c_prev", "h_proposal", "c_proposal")
    TRANSITION_NAMES = (
        "h_prev -> h_proposal",
        "c_prev -> c_proposal",
        "h_prev -> c_prev",
        "h_proposal -> c_proposal",
    )
    TRANSITION_NAMES_EXTRA = TRANSITION_NAMES + ("c_proposal -> h_proposal",)
    QUERY_INDICES = (2, 3)  # chỉ h_proposal, c_proposal làm query

    def __init__(
        self,
        hidden_size: int,
        attn_dim: int = 24,
        num_heads: int = 2,
        rank: int = 4,
        context_dim: Optional[int] = None,
        extra_edge: bool = False,
        gamma_mode: str = "softmax",
    ) -> None:
        """
        extra_edge: nếu True, thêm lại cạnh 3->2 (c_proposal -> h_proposal) --
            khôi phục CÓ CHỌN LỌC cặp cạnh hai chiều gần quyết định keep/add
            nhất từ bản DSTA đầy đủ (TEN_EM_LA_GI.py), thay vì bỏ hết như
            bản Lite gốc. Cạnh 0<->1 (prev-side) vẫn không khôi phục vì ít
            liên quan trực tiếp đến quyết định r_t hơn cạnh 2<->3.
        gamma_mode: "softmax" (mặc định, giữ nguyên bản Lite gốc -- gamma_s/
            gamma_d bắt buộc dương, tổng=1 theo rank) hoặc "sigmoid" (mỗi
            rank có cổng độc lập qua sigmoid, không ràng buộc tổng=1 --
            thêm 1 bậc tự do cho phép "phủ định" một rank channel).
        """
        super().__init__()
        if gamma_mode not in ("softmax", "sigmoid"):
            raise ValueError(f"gamma_mode phải là 'softmax' hoặc 'sigmoid', nhận: {gamma_mode}")
        self.extra_edge = extra_edge
        self.gamma_mode = gamma_mode

        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim={attn_dim} phải chia hết cho num_heads={num_heads}")

        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.rank = rank
        self.head_dim = attn_dim // num_heads
        self.context_dim = context_dim or attn_dim

        # --- State tokenization ---
        self.hidden_state_proj = nn.Linear(hidden_size, attn_dim, bias=False)
        self.cell_state_proj = nn.Linear(hidden_size, attn_dim, bias=False)
        self.role_embedding = nn.Parameter(torch.zeros(2, attn_dim))   # 0=hidden, 1=cell
        self.time_embedding = nn.Parameter(torch.zeros(2, attn_dim))   # 0=prev,   1=proposal
        self.token_norm = nn.LayerNorm(attn_dim)

        # --- Q (chỉ 2 state) / K (cả 4 state) ---
        self.q_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(attn_dim, attn_dim, bias=False)

        # --- Transition context z_t: 4 state token + delta_h + delta_c ---
        context_input_dim = 4 * attn_dim + 2 * hidden_size
        self.context_proj = nn.Linear(context_input_dim, self.context_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)

        # --- Dynamic gate cho symmetric / antisymmetric / directional mix ---
        self.gamma_s_proj = nn.Linear(self.context_dim, num_heads * rank)
        self.gamma_d_proj = nn.Linear(self.context_dim, num_heads * rank)
        self.lambda_proj = nn.Linear(self.context_dim, num_heads)

        # --- Low-rank triadic factor (CP decomposition) ---
        self.factor_a = nn.Parameter(torch.empty(num_heads, rank, self.head_dim))  # query
        self.factor_b = nn.Parameter(torch.empty(num_heads, rank, self.head_dim))  # source
        self.factor_c = nn.Parameter(torch.empty(num_heads, rank, self.head_dim))  # target

        # --- 4-cạnh (hoặc 5 nếu extra_edge=True) transition graph ---
        src_list = [0, 1, 0, 2]
        dst_list = [2, 3, 1, 3]
        if extra_edge:
            src_list.append(3)  # c_proposal -> h_proposal (chiều ngược của cạnh 2->3)
            dst_list.append(2)
        self.register_buffer(
            "transition_sources", torch.tensor(src_list, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "transition_targets", torch.tensor(dst_list, dtype=torch.long), persistent=False
        )
        self.num_transitions = len(src_list)

        # --- Transition value: [s_j, s_k, s_j⊙s_k] -- ĐÃ BỎ delta = s_k - s_j ---
        self.value_proj = nn.Linear(3 * attn_dim, attn_dim, bias=False)

        # --- Output: gộp 2 query (h_proposal, c_proposal) x H head -> r_t ---
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
        states = torch.stack([h_prev_tok, c_prev_tok, h_prop_tok, c_prop_tok], dim=1)  # (B,4,A)
        return self.token_norm(states)

    def _build_context(self, states, h_prev, c_prev, h_proposal, c_proposal):
        B = states.size(0)
        flat_states = states.reshape(B, -1)
        delta_h = h_proposal - h_prev
        delta_c = c_proposal - c_prev
        context_input = torch.cat([flat_states, delta_h, delta_c], dim=-1)
        return self.context_norm(self.context_proj(context_input))

    def forward(
        self,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        h_proposal: torch.Tensor,
        c_proposal: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B = h_prev.size(0)
        H, R, Dh, E = self.num_heads, self.rank, self.head_dim, self.num_transitions

        states = self._build_state_tokens(h_prev, c_prev, h_proposal, c_proposal)  # (B,4,A)

        # Query: CHỈ 2 state (h_proposal, c_proposal)
        query_states = states[:, self.QUERY_INDICES, :]                      # (B,2,A)
        q = self.q_proj(query_states).view(B, 2, H, Dh).permute(0, 2, 1, 3)  # (B,H,2,Dh)

        # Key: cả 4 state
        k = self.k_proj(states).view(B, 4, H, Dh).permute(0, 2, 1, 3)        # (B,H,4,Dh)

        z_t = self._build_context(states, h_prev, c_prev, h_proposal, c_proposal)

        if self.gamma_mode == "softmax":
            gamma_s = F.softmax(self.gamma_s_proj(z_t).view(B, H, R), dim=-1)
            gamma_d = F.softmax(self.gamma_d_proj(z_t).view(B, H, R), dim=-1)
        else:  # "sigmoid" -- mỗi rank có cổng độc lập, không ràng buộc tổng=1
            gamma_s = torch.sigmoid(self.gamma_s_proj(z_t).view(B, H, R))
            gamma_d = torch.sigmoid(self.gamma_d_proj(z_t).view(B, H, R))
        directional_mix = torch.sigmoid(self.lambda_proj(z_t))  # (B,H)

        # Low-rank triadic projections
        chi = torch.einsum("bhid,hrd->bhir", q, self.factor_a)   # (B,H,2,R)
        xi = torch.einsum("bhjd,hrd->bhjr", k, self.factor_b)    # (B,H,4,R)
        zeta = torch.einsum("bhjd,hrd->bhjr", k, self.factor_c)  # (B,H,4,R)

        src, dst = self.transition_sources, self.transition_targets
        xi_src, xi_dst = xi[:, :, src, :], xi[:, :, dst, :]      # (B,H,E,R)
        zeta_src, zeta_dst = zeta[:, :, src, :], zeta[:, :, dst, :]

        symmetric = 0.5 * (xi_src * zeta_dst + xi_dst * zeta_src)     # S_jk = S_kj
        directional = 0.5 * (xi_src * zeta_dst - xi_dst * zeta_src)   # D_jk = -D_kj

        relational = (
            gamma_s.unsqueeze(2) * symmetric
            + directional_mix.unsqueeze(-1).unsqueeze(-1) * gamma_d.unsqueeze(2) * directional
        )  # (B,H,E,R)

        scores = torch.einsum("bhir,bher->bhie", chi, relational) / math.sqrt(max(R, 1))  # (B,H,2,E)
        attention = F.softmax(scores, dim=-1)

        state_src = states[:, src, :]   # (B,E,A)
        state_dst = states[:, dst, :]   # (B,E,A)
        interaction = state_src * state_dst
        transition_features = torch.cat([state_src, state_dst, interaction], dim=-1)  # (B,E,3A) -- không có delta
        values = self.value_proj(transition_features).view(B, E, H, Dh).permute(0, 2, 1, 3)  # (B,H,E,Dh)

        outputs = torch.einsum("bhie,bhed->bhid", attention, values)  # (B,H,2,Dh)
        h_out, c_out = outputs[:, :, 0, :], outputs[:, :, 1, :]       # (B,H,Dh) mỗi cái
        current_out = torch.cat([h_out, c_out], dim=-1).reshape(B, 2 * self.attn_dim)

        gate_logits = self.output_norm(self.output_proj(current_out))
        r_t = torch.sigmoid(gate_logits)

        diagnostics: Dict[str, torch.Tensor] = {}
        if return_diagnostics:
            diagnostics = {
                "attention": attention,
                "gamma_s": gamma_s,
                "gamma_d": gamma_d,
                "directional_mix": directional_mix,
            }
        return r_t, diagnostics


class PerDimStreamBlend(nn.Module):
    """
    Trộn keep/add THEO TỪNG CHIỀU HIDDEN, khôi phục từ `stream_attention` của
    bản DSTA đầy đủ (TEN_EM_LA_GI.py -- DSTAResidualMemoryCell), thay vì trộn
    bằng 1 cặp số toàn cục như W_alpha (bản Lite gốc / hmr_bilstm.py).

    MLP 2 lớp (Linear -> GELU -> Linear) điều kiện theo [keep, add, keep*add]
    ra 2*hidden_size logits -> reshape (hidden_size, 2) -> softmax theo chiều
    keep/add cho TỪNG chiều hidden độc lập.

    bottleneck_dim thu nhỏ so với bản gốc (mặc định hidden_size, ở đây dùng
    hidden_size//2) để giữ ngân sách tham số gần baseline hơn.
    """

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
        stream_context = torch.cat(
            [c_keep_norm, c_add_norm, c_keep_norm * c_add_norm], dim=-1
        )
        logits = self.mlp(stream_context).view(B, self.hidden_size, 2)
        weights = F.softmax(logits, dim=-1)
        return weights[..., 0], weights[..., 1]  # keep_weight, add_weight -- mỗi cái (B, hidden_size)


# =============================================================================
#  DSTA-LITE CELL -- giống hệt RLSTMCell của hmr_bilstm.py, chỉ đổi r_t
#  (+ tuỳ chọn blend_mode="per_dim" để khôi phục stream_attention của bản gốc)
# =============================================================================
class DSTALiteCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        attn_dim: int = 24,
        num_heads: int = 2,
        rank: int = 4,
        context_dim: Optional[int] = None,
        extra_edge: bool = False,
        gamma_mode: str = "softmax",
        blend_mode: str = "scalar",
        blend_bottleneck_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if blend_mode not in ("scalar", "per_dim"):
            raise ValueError(f"blend_mode phải là 'scalar' hoặc 'per_dim', nhận: {blend_mode}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blend_mode = blend_mode

        # 4 cổng LSTM chuẩn -- giống hệt RLSTMCell
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)

        # Thay m_t/LayerNorm cũ bằng DSTA-Lite
        self.dsta_gate = DSTALiteGate(
            hidden_size=hidden_size, attn_dim=attn_dim, num_heads=num_heads,
            rank=rank, context_dim=context_dim, extra_edge=extra_edge, gamma_mode=gamma_mode,
        )

        # Keep/Add blend: "scalar" = W_alpha toàn cục (bản Lite gốc),
        # "per_dim" = PerDimStreamBlend khôi phục từ bản DSTA đầy đủ.
        if blend_mode == "scalar":
            self.W_alpha = nn.Linear(2 * hidden_size, 2)
        else:
            self.stream_blend = PerDimStreamBlend(hidden_size, bottleneck_dim=blend_bottleneck_dim)

        # Hybrid blend gate beta -- giống hệt RLSTMCell
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
                        param[n:2 * n].fill_(1.0)  # forget-gate bias = 1
                if "W_beta.bias" in name:
                    with torch.no_grad():
                        param.fill_(-5.0)  # sigmoid(-5) ~= 0.007, gần như tắt lúc đầu train

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_size, dim=1)
        i_t, f_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        # "LSTM path" -- không phụ thuộc r_t, tính trước để làm proposal cho DSTA-Lite
        c_proposal = f_t * c_prev + i_t * g_t
        h_proposal = o_t * torch.tanh(c_proposal)

        r_t, _ = self.dsta_gate(h_prev, c_prev, h_proposal, c_proposal)

        c_keep = (f_t + r_t * (1.0 - f_t)) * c_prev
        c_add = (1.0 - r_t) * (i_t * g_t)

        c_keep_norm = F.layer_norm(c_keep, (c_keep.shape[-1],))
        c_add_norm = F.layer_norm(c_add, (c_add.shape[-1],))
        if self.blend_mode == "scalar":
            combined = torch.cat([c_keep_norm, c_add_norm], dim=-1)
            alpha = F.softmax(self.W_alpha(combined), dim=-1)
            keep_weight, add_weight = alpha[:, 0:1], alpha[:, 1:2]  # (B,1) broadcast qua mọi chiều
        else:
            keep_weight, add_weight = self.stream_blend(c_keep_norm, c_add_norm)  # (B,H) mỗi chiều riêng
        c_rmc = keep_weight * c_keep + add_weight * c_add

        beta = torch.sigmoid(self.W_beta(h_prev))
        c_t = beta * c_rmc + (1.0 - beta) * c_proposal  # c_proposal đóng vai trò c_lstm

        h_t = self.dropout(o_t * torch.tanh(c_t))

        self.last_r_t = r_t
        self.last_c_keep = c_keep
        self.last_c_add = c_add

        return h_t, c_t


# =============================================================================
#  LAYER + BIDIRECTIONAL (mirror RLSTMLayer / BiRLSTM của hmr_bilstm.py)
# =============================================================================
class DSTALiteLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, **dsta_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DSTALiteCell(input_size, hidden_size, dropout=dropout, **dsta_kwargs)

    def forward(self, x, h0=None, c0=None):
        B, T, _ = x.size()
        device = x.device
        h = torch.zeros(B, self.hidden_size, device=device) if h0 is None else h0
        c = torch.zeros(B, self.hidden_size, device=device) if c0 is None else c0

        h_outputs, r_outputs = [], []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            h_outputs.append(h)
            r_outputs.append(self.cell.last_r_t)

        h_seq = torch.stack(h_outputs, dim=1)
        r_seq = torch.stack(r_outputs, dim=1)
        return h_seq, (h, c), r_seq


class BiDSTALite(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1, **dsta_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = input_size if i == 0 else 2 * hidden_size
            self.fwd_layers.append(DSTALiteLayer(layer_in, hidden_size, dropout=dropout, **dsta_kwargs))
            self.bwd_layers.append(DSTALiteLayer(layer_in, hidden_size, dropout=dropout, **dsta_kwargs))

        self.inter_dropout = nn.Dropout(dropout) if num_layers > 1 else None

    def forward(self, x):
        h_seq = x
        r_fwd = r_bwd = None
        for i in range(self.num_layers):
            h_f, _, r_fwd = self.fwd_layers[i](h_seq)
            x_rev = torch.flip(h_seq, dims=[1])
            h_b, _, r_bwd = self.bwd_layers[i](x_rev)
            h_b = torch.flip(h_b, dims=[1])
            r_bwd = torch.flip(r_bwd, dims=[1])

            h_seq = torch.cat([h_f, h_b], dim=-1)
            if self.inter_dropout is not None and i < self.num_layers - 1:
                h_seq = self.inter_dropout(h_seq)

        return h_seq, r_fwd, r_bwd


# =============================================================================
#  CLASSIFIER -- CNN + DSTA-Lite BiLSTM + Attention Pooling (mirror RLSTMClassifier)
# =============================================================================
class DSTALiteClassifier(nn.Module):
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
        extra_edge: bool = False,
        gamma_mode: str = "softmax",
        blend_mode: str = "scalar",
        blend_bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.cnn = ECGFeatureExtractor(
            input_channels=input_size, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )
        self.birlstm = BiDSTALite(
            cnn_out_channels, hidden_size, num_layers=num_layers, dropout=dropout,
            attn_dim=attn_dim, num_heads=num_heads, rank=rank,
            extra_edge=extra_edge, gamma_mode=gamma_mode,
            blend_mode=blend_mode, blend_bottleneck_dim=blend_bottleneck_dim,
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

    def forward(self, x, return_internals=False):
        features = self.cnn(x)
        h_seq, r_fwd, r_bwd = self.birlstm(features)

        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_pooled = self.dropout(self.layer_norm(h_pooled))
        logits = self.classifier(h_pooled)

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "attention_weights": attn_weights}
        return logits


# =============================================================================
#  TEST TRÊN DỮ LIỆU NHỎ (dummy data, chưa phải MIT-BIH thật)
# =============================================================================
if __name__ == "__main__":
    import time
    from hmr_bilstm import RLSTMClassifier, RLSTMLoss

    torch.manual_seed(42)
    B, T, D, NUM_CLASSES = 8, 187, 1, 5
    HIDDEN = 96
    NUM_LAYERS = 2  # đúng cấu hình thật train.py đang dùng cho ResMemECG

    x = torch.randn(B, T, D)
    y = torch.randint(0, NUM_CLASSES, (B,))
    criterion = RLSTMLoss(lambda_smooth=0.003)

    print("=" * 70)
    print("1) Đếm tham số -- so với baseline hmr_bilstm.py (505,038)")
    print("=" * 70)

    baseline = RLSTMClassifier(input_size=D, hidden_size=HIDDEN, num_classes=NUM_CLASSES, num_layers=NUM_LAYERS)
    n_baseline = sum(p.numel() for p in baseline.parameters())
    print(f"  Baseline (RLSTMClassifier, m_t gốc):        {n_baseline:>10,d}")

    model = DSTALiteClassifier(
        input_size=D, hidden_size=HIDDEN, num_classes=NUM_CLASSES, num_layers=NUM_LAYERS,
        attn_dim=24, num_heads=2, rank=4,
    )
    n_dsta_lite = sum(p.numel() for p in model.parameters())
    print(f"  DSTA-Lite (attn_dim=24, heads=2, rank=4):    {n_dsta_lite:>10,d}")
    print(f"  Chênh lệch:                                  {n_dsta_lite - n_baseline:>+10,d}"
          f"  ({100 * (n_dsta_lite / n_baseline - 1):+.1f}%)")

    print()
    print("=" * 70)
    print("2) Forward + backward smoke test (dummy data)")
    print("=" * 70)

    logits, internals = model(x, return_internals=True)
    print(f"  Logits shape: {tuple(logits.shape)}")
    print(f"  r_fwd shape:  {tuple(internals['r_fwd'].shape)}")

    loss, comp = criterion(logits, y, internals["r_fwd"], internals["r_bwd"])
    print(f"  Loss (task+smooth): {loss.item():.4f}  (task={comp['task'].item():.4f}, "
          f"smooth={comp['smooth'].item():.4f})")
    loss.backward()

    n_nan_grad = sum(
        1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any()
    )
    n_none_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"  Params với grad=NaN: {n_nan_grad}  |  Params không nhận grad: {n_none_grad}")
    print("  -> OK, không NaN, backward chạy được." if n_nan_grad == 0 else "  -> LỖI: có NaN trong grad!")

    print()
    print("=" * 70)
    print("3) So sánh tốc độ forward (ước lượng thô, cùng thiết bị, CPU/GPU tuỳ máy)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline.to(device)
    model.to(device)
    x_dev = x.to(device)

    def time_forward(m, n_iters=5):
        m.eval()
        with torch.no_grad():
            # warmup
            for _ in range(2):
                m(x_dev)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_iters):
                m(x_dev)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.time() - t0) / n_iters

    t_baseline = time_forward(baseline)
    t_dsta_lite = time_forward(model)
    print(f"  Device: {device}")
    print(f"  Baseline forward:   {t_baseline * 1000:.2f} ms/batch (B={B})")
    print(f"  DSTA-Lite forward:  {t_dsta_lite * 1000:.2f} ms/batch (B={B})")
    print(f"  Chậm hơn:           {t_dsta_lite / t_baseline:.2f}x")

    print()
    print("=" * 70)
    print("4) Kiểm tra β khởi động gần LSTM thuần (bias=-5)")
    print("=" * 70)
    beta_sample = torch.sigmoid(model.birlstm.fwd_layers[0].cell.W_beta.bias)
    print(f"  Mean sigmoid(W_beta.bias) lúc init: {beta_sample.mean().item():.4f} (kỳ vọng ~0.0067)")
