"""
Modern architecture benchmark — self-contained model zoo.

Backbones (all consume ECG beats shaped (B, T, 1), T=187):
  - bilstm           : CNN + ResMemECG (paper baseline, re-implemented here
                        standalone so this folder has zero import dependency
                        on the parent project's hmr_bilstm.py)
  - cnn_transformer  : CNN + TransformerEncoder
  - cnn_mamba        : CNN + bidirectional minimal selective-SSM (S6/Mamba)
  - pure_transformer : patch-embed raw waveform + TransformerEncoder (no CNN)

Kept deliberately independent of ../hmr_bilstm.py per the isolation requirement
for this experiment (see README.md).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  SHARED COMPONENTS
# =============================================================================
class ECGFeatureExtractor(nn.Module):
    """2-layer 1D-CNN front-end, identical shape contract to the paper baseline."""

    def __init__(self, input_channels=1, output_channels=64, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        return x.transpose(1, 2)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, h_seq):
        scores = self.attention(h_seq)
        weights = F.softmax(scores, dim=1)
        pooled = (h_seq * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def make_classifier_head(in_dim, hidden_size, num_classes, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, num_classes),
    )


# =============================================================================
#  BASELINE — CNN + ResMemECG (standalone re-implementation for this benchmark)
# =============================================================================
class RLSTMCell(nn.Module):
    """Hybrid Memory Path BiLSTM cell — same equations as ../hmr_bilstm.py."""

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h_rmc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.W_alpha = nn.Linear(2 * hidden_size, 2)
        self.W_beta = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.last_r_t = None
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if any(k in name for k in ["W_h", "W_c", "W_h_rmc"]):
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "W_x" in name or "W_h" in name:
                    n = param.size(0) // 4
                    with torch.no_grad():
                        param[n:2 * n].fill_(1.0)
                if "W_beta.bias" in name:
                    with torch.no_grad():
                        param.fill_(-5.0)

    def forward(self, x_t, h_prev, c_prev):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_size, dim=1)
        i_t, f_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        Wc_c = self.W_c(c_prev)
        Wh_h = self.W_h_rmc(h_prev)
        m_t = Wc_c * Wh_h
        r_t = torch.sigmoid(self.layer_norm(Wc_c + Wh_h + m_t))

        c_keep = (f_t + r_t * (1.0 - f_t)) * c_prev
        c_add = (1.0 - r_t) * (i_t * g_t)

        c_keep_norm = F.layer_norm(c_keep, (c_keep.shape[-1],))
        c_add_norm = F.layer_norm(c_add, (c_add.shape[-1],))
        scores = self.W_alpha(torch.cat([c_keep_norm, c_add_norm], dim=-1))
        alpha = F.softmax(scores, dim=-1)
        c_rmc = alpha[:, 0:1] * c_keep + alpha[:, 1:2] * c_add

        c_lstm = f_t * c_prev + i_t * g_t
        beta = torch.sigmoid(self.W_beta(h_prev))
        c_t = beta * c_rmc + (1.0 - beta) * c_lstm

        h_t = self.dropout(o_t * torch.tanh(c_t))
        self.last_r_t = r_t
        return h_t, c_t


class BiRLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        for i in range(num_layers):
            layer_in = input_size if i == 0 else 2 * hidden_size
            self.fwd_cells.append(RLSTMCell(layer_in, hidden_size, dropout))
            self.bwd_cells.append(RLSTMCell(layer_in, hidden_size, dropout))
        self.inter_dropout = nn.Dropout(dropout) if num_layers > 1 else None

    def _run_direction(self, cell, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        outs = []
        for t in range(T):
            h, c = cell(x[:, t], h, c)
            outs.append(h)
        return torch.stack(outs, dim=1)

    def forward(self, x):
        h_seq = x
        for i in range(self.num_layers):
            h_fwd = self._run_direction(self.fwd_cells[i], h_seq)
            h_bwd = torch.flip(self._run_direction(self.bwd_cells[i], torch.flip(h_seq, dims=[1])), dims=[1])
            h_seq = torch.cat([h_fwd, h_bwd], dim=-1)
            if self.inter_dropout is not None and i < self.num_layers - 1:
                h_seq = self.inter_dropout(h_seq)
        return h_seq


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=96, dropout=0.25, num_classes=5,
                 cnn_out_channels=64, num_layers=2):
        super().__init__()
        self.cnn = ECGFeatureExtractor(input_size, cnn_out_channels, dropout * 0.5)
        self.birlstm = BiRLSTM(cnn_out_channels, hidden_size, num_layers, dropout)
        self.attention_pool = AttentionPooling(2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = make_classifier_head(2 * hidden_size, hidden_size, num_classes, dropout)

    def forward(self, x):
        feats = self.cnn(x)
        h_seq = self.birlstm(feats)
        pooled, _ = self.attention_pool(h_seq)
        pooled = self.dropout(self.layer_norm(pooled))
        return self.classifier(pooled)


# =============================================================================
#  CNN + TRANSFORMER
# =============================================================================
class CNNTransformerClassifier(nn.Module):
    def __init__(self, input_size=1, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.25, num_classes=5, cnn_out_channels=64):
        super().__init__()
        self.cnn = ECGFeatureExtractor(input_size, cnn_out_channels, dropout * 0.5)
        self.input_proj = nn.Linear(cnn_out_channels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = AttentionPooling(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = make_classifier_head(d_model, d_model // 2, num_classes, dropout)

    def forward(self, x):
        feats = self.cnn(x)
        h = self.pos_enc(self.input_proj(feats))
        h = self.encoder(h)
        pooled, _ = self.attention_pool(h)
        pooled = self.dropout(self.layer_norm(pooled))
        return self.classifier(pooled)


# =============================================================================
#  CNN + MAMBA (minimal selective-SSM, pure PyTorch — no CUDA kernel dependency)
# =============================================================================
class MinimalSSM(nn.Module):
    """Simplified S6 selective-scan mixer (Gu & Dao, 2023), loop-based.

    Input-dependent B_t/C_t/delta_t (the "selection" mechanism), diagonal
    negative-real state matrix A, zero-order-hold discretization. Not the
    official mamba-ssm CUDA kernel — a from-scratch PyTorch scan, used because
    mamba-ssm/causal-conv1d do not install cleanly on Windows.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T]
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, T, d_inner)

        B_ssm, C_ssm, dt_raw = torch.split(self.x_proj(x_conv), [self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))  # (B, T, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            delta_t = delta[:, t]  # (B, d_inner)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            Bx = delta_t.unsqueeze(-1) * B_ssm[:, t].unsqueeze(1) * x_conv[:, t].unsqueeze(-1)
            h = A_bar * h + Bx
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(self.dropout(y))


class MambaBlock(nn.Module):
    """Pre-norm residual wrapper around MinimalSSM (unidirectional)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = MinimalSSM(d_model, d_state, d_conv, expand, dropout)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class BiMambaLayer(nn.Module):
    """Forward + backward MambaBlock, concatenated and projected back to d_model."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.bwd = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h_fwd = self.fwd(x)
        h_bwd = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.dropout(self.merge(torch.cat([h_fwd, h_bwd], dim=-1)))


class CNNMambaClassifier(nn.Module):
    def __init__(self, input_size=1, d_model=128, d_state=16, num_layers=4,
                 dropout=0.25, num_classes=5, cnn_out_channels=64):
        super().__init__()
        self.cnn = ECGFeatureExtractor(input_size, cnn_out_channels, dropout * 0.5)
        self.input_proj = nn.Linear(cnn_out_channels, d_model)
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, dropout=dropout) for _ in range(num_layers)
        ])
        self.attention_pool = AttentionPooling(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = make_classifier_head(d_model, d_model // 2, num_classes, dropout)

    def forward(self, x):
        feats = self.cnn(x)
        h = self.input_proj(feats)
        for layer in self.layers:
            h = layer(h)
        pooled, _ = self.attention_pool(h)
        pooled = self.dropout(self.layer_norm(pooled))
        return self.classifier(pooled)


# =============================================================================
#  CNN + xLSTM (sLSTM variant — Beck et al., 2024, "xLSTM: Extended LSTM")
# =============================================================================
class SLSTMCell(nn.Module):
    """Scalar-memory xLSTM cell: exponential input gate + sigmoid forget gate,
    with the paper's log-space stabilizer (m_t) and normalizer (n_t) state to
    prevent the exponential gate from over/underflowing.

    This is the single-head scalar-memory ("sLSTM") variant from the xLSTM
    paper, not the full multi-head sLSTM + matrix-memory mLSTM block stack —
    chosen because it is the variant structurally closest to a direct
    "modernized LSTM gating" comparison against ResMemECG's own custom gating.
    """

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x_t, h_prev, c_prev, n_prev, m_prev):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_pre, f_pre, o_pre, z_pre = torch.split(gates, self.hidden_size, dim=1)

        log_i = i_pre                     # i_t = exp(i_pre)  -> log_i = i_pre
        log_f = F.logsigmoid(f_pre)       # f_t = sigmoid(f_pre)
        o_t = torch.sigmoid(o_pre)
        z_t = torch.tanh(z_pre)

        m_t = torch.maximum(log_f + m_prev, log_i)
        i_stable = torch.exp(log_i - m_t)
        f_stable = torch.exp(log_f + m_prev - m_t)

        c_t = f_stable * c_prev + i_stable * z_t
        n_t = f_stable * n_prev + i_stable

        h_t = self.dropout(o_t * (c_t / (n_t + 1e-6)))
        return h_t, c_t, n_t, m_t


class BiSLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        for i in range(num_layers):
            layer_in = input_size if i == 0 else 2 * hidden_size
            self.fwd_cells.append(SLSTMCell(layer_in, hidden_size, dropout))
            self.bwd_cells.append(SLSTMCell(layer_in, hidden_size, dropout))
        self.inter_dropout = nn.Dropout(dropout) if num_layers > 1 else None

    def _run_direction(self, cell, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        n = torch.zeros(B, self.hidden_size, device=x.device)
        m = torch.zeros(B, self.hidden_size, device=x.device)
        outs = []
        for t in range(T):
            h, c, n, m = cell(x[:, t], h, c, n, m)
            outs.append(h)
        return torch.stack(outs, dim=1)

    def forward(self, x):
        h_seq = x
        for i in range(self.num_layers):
            h_fwd = self._run_direction(self.fwd_cells[i], h_seq)
            h_bwd = torch.flip(self._run_direction(self.bwd_cells[i], torch.flip(h_seq, dims=[1])), dims=[1])
            h_seq = torch.cat([h_fwd, h_bwd], dim=-1)
            if self.inter_dropout is not None and i < self.num_layers - 1:
                h_seq = self.inter_dropout(h_seq)
        return h_seq


class CNNXLSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=96, dropout=0.25, num_classes=5,
                 cnn_out_channels=64, num_layers=2):
        super().__init__()
        self.cnn = ECGFeatureExtractor(input_size, cnn_out_channels, dropout * 0.5)
        self.bislstm = BiSLSTM(cnn_out_channels, hidden_size, num_layers, dropout)
        self.attention_pool = AttentionPooling(2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = make_classifier_head(2 * hidden_size, hidden_size, num_classes, dropout)

    def forward(self, x):
        feats = self.cnn(x)
        h_seq = self.bislstm(feats)
        pooled, _ = self.attention_pool(h_seq)
        pooled = self.dropout(self.layer_norm(pooled))
        return self.classifier(pooled)


# =============================================================================
#  PURE TRANSFORMER (no CNN — patch-embed raw waveform)
# =============================================================================
class PureTransformerClassifier(nn.Module):
    def __init__(self, input_size=1, d_model=128, nhead=4, num_layers=6,
                 dim_feedforward=256, dropout=0.25, num_classes=5, patch_size=11):
        super().__init__()
        # 187 = 11 * 17 for this dataset's fixed beat length; stride=kernel_size
        # patchifies the raw waveform directly, no convolutional feature extraction.
        self.patch_embed = nn.Conv1d(input_size, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = AttentionPooling(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = make_classifier_head(d_model, d_model // 2, num_classes, dropout)

    def forward(self, x):
        patches = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)  # (B, num_patches, d_model)
        h = self.pos_enc(patches)
        h = self.encoder(h)
        pooled, _ = self.attention_pool(h)
        pooled = self.dropout(self.layer_norm(pooled))
        return self.classifier(pooled)


# =============================================================================
#  REGISTRY
# =============================================================================
def build_model(model_type: str, num_classes: int = 5, dropout: float = 0.25, **kwargs):
    if model_type == "bilstm":
        return BiLSTMClassifier(num_classes=num_classes, dropout=dropout, **kwargs)
    elif model_type == "cnn_transformer":
        return CNNTransformerClassifier(num_classes=num_classes, dropout=dropout, **kwargs)
    elif model_type == "cnn_mamba":
        return CNNMambaClassifier(num_classes=num_classes, dropout=dropout, **kwargs)
    elif model_type == "cnn_xlstm":
        return CNNXLSTMClassifier(num_classes=num_classes, dropout=dropout, **kwargs)
    elif model_type == "pure_transformer":
        return PureTransformerClassifier(num_classes=num_classes, dropout=dropout, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D = 4, 187, 1
    x = torch.randn(B, T, D)
    y = torch.randint(0, 5, (B,))

    for name in ["bilstm", "cnn_transformer", "cnn_mamba", "cnn_xlstm", "pure_transformer"]:
        model = build_model(name)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:<18} logits={tuple(logits.shape)}  params={n_params:,}  loss={loss.item():.4f}  backward OK")
