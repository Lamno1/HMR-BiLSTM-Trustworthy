"""
hmr_bilstm_rr.py
=================
Prototype: HMR-BiLSTM full architecture + RR-interval side input.

Reuses the exact CNN / BiRLSTM / AttentionPooling building blocks from
hmr_bilstm_ablation.py (use_rmc=True, use_hybrid=True, use_cnn=True,
use_attention=True, use_interaction=True — i.e. the "full" variant) so this
is a controlled, isolated test of "does adding RR help", not a new
architecture. Only the classifier head changes: it concatenates the 3
RR-interval features to the pooled sequence representation.
"""

import torch
import torch.nn as nn

from hmr_bilstm_ablation import ECGFeatureExtractor, AttentionPooling, BiRLSTM


class RLSTMClassifierRR(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 96,
        dropout: float = 0.25,
        num_classes: int = 5,
        cnn_out_channels: int = 64,
        num_layers: int = 2,
        n_rr_features: int = 3,
    ):
        super().__init__()

        self.cnn = ECGFeatureExtractor(
            input_channels=input_size,
            output_channels=cnn_out_channels,
            dropout=dropout * 0.5,
        )

        self.birlstm = BiRLSTM(
            cnn_out_channels, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_rmc=True,
            use_hybrid=True,
            use_interaction=True,
        )

        self.attention_pool = AttentionPooling(2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size + n_rr_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, rr, return_internals=False):
        features = self.cnn(x)
        outputs = self.birlstm(features)
        h_seq = outputs[0]
        r_fwd, r_bwd = outputs[2], outputs[3]

        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_pooled = self.layer_norm(h_pooled)
        h_pooled = self.dropout(h_pooled)

        combined = torch.cat([h_pooled, rr], dim=-1)
        logits = self.classifier(combined)

        if return_internals:
            internals = {
                "r_fwd": r_fwd,
                "r_bwd": r_bwd,
                "attention_weights": attn_weights,
            }
            return logits, internals
        return logits
