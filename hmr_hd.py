"""
hmr_hd.py  —  HMR-BiLSTM with a differentiable HIERARCHICAL decision head.
========================================================================
Motivation (measured, see memory project_fclass_root_cause_diagnostics):
  The official model has AUC_F = 0.959 but F1-F = 0.241. The discriminative
  information is already there; it is lost at the FLAT 5-way softmax where N's
  ~90% prior wins the close calls (F->N 45%). This module keeps the whole
  proven front end (CNN + BiRLSTM + attention pool) UNCHANGED and only replaces
  the final classifier with a soft two-level decomposition:

      p_abn = sigmoid(abn_head(e))              # P(not-N)
      q     = softmax(sub_head(e))  over [S,V,F,Q]
      P(N)  = 1 - p_abn
      P(c)  = p_abn * q[c]     for c in {S,V,F,Q}

  Fully differentiable, trained end-to-end on the true label — NO hard router
  (the earlier hard-cascade attempt misrouted 26% of F; this one never
  destroys information at the routing boundary). Composed output is returned as
  5-way LOG-PROBS [N,S,V,F,Q]; softmax(log_probs) == probs since they already
  sum to 1, so train_inter_patient.evaluate() works unchanged.

Optional logit adjustment (Menon et al. 2020, "Long-tail learning via logit
adjustment") on the sub-head: pass `logit_adjust` (tensor over [S,V,F,Q]) to
HDLoss — added to sub-logits at TRAIN time only, the provably prior-optimal
replacement for ad-hoc class weights / focal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr_bilstm import (
    ECGFeatureExtractor, BiRLSTM, AttentionPooling, temporal_smoothness_loss,
)


class RLSTMClassifierHD(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 96, dropout: float = 0.25,
                 num_classes: int = 5, cnn_out_channels: int = 64, num_layers: int = 2):
        super().__init__()
        assert num_classes == 5, "HD head composes N + [S,V,F,Q]"
        self.cnn = ECGFeatureExtractor(input_channels=input_size,
                                       output_channels=cnn_out_channels, dropout=dropout * 0.5)
        self.birlstm = BiRLSTM(cnn_out_channels, hidden_size,
                               num_layers=num_layers, dropout=dropout)
        self.attention_pool = AttentionPooling(2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)

        H = 2 * hidden_size
        self.abn_head = nn.Sequential(
            nn.Linear(H, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.sub_head = nn.Sequential(
            nn.Linear(H, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 4),          # S, V, F, Q
        )

    def embed(self, x):
        feats = self.cnn(x)
        outputs = self.birlstm(feats)
        h_seq, r_fwd, r_bwd = outputs[0], outputs[2], outputs[3]
        h_pooled, attn = self.attention_pool(h_seq)
        h_pooled = self.dropout(self.layer_norm(h_pooled))
        return h_pooled, r_fwd, r_bwd, attn

    def forward(self, x, return_internals: bool = False):
        e, r_fwd, r_bwd, attn = self.embed(x)
        abn_logit = self.abn_head(e)                       # (B, 1)
        sub_logits = self.sub_head(e)                      # (B, 4)

        log_p_norm = F.logsigmoid(-abn_logit)             # log P(N)     (B,1)
        log_p_abn = F.logsigmoid(abn_logit)               # log P(abn)   (B,1)
        log_q = F.log_softmax(sub_logits, dim=-1)         # (B,4)
        log_probs = torch.cat([log_p_norm, log_p_abn + log_q], dim=-1)   # (B,5) [N,S,V,F,Q]

        if return_internals:
            return log_probs, {
                "r_fwd": r_fwd, "r_bwd": r_bwd, "attention_weights": attn,
                "abn_logit": abn_logit, "sub_logits": sub_logits, "embedding": e,
            }
        return log_probs


class HDLoss(nn.Module):
    """NLL on the composed hierarchical log-probs + temporal-smoothness reg.

    focal_gamma > 0  : focal modulation (1-pt)^gamma on the composed NLL.
    class_weights     : per-class multiplier on the composed NLL (5-way).
    logit_adjust      : tensor over [S,V,F,Q] added to sub-logits at train time
                        (balanced-softmax / logit adjustment). Mutually the
                        principled alternative to class_weights+focal.
    """

    def __init__(self, lambda_smooth: float = 0.003, class_weights=None,
                 focal_gamma: float = 1.5, logit_adjust=None):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.gamma = focal_gamma
        self.register_buffer("w", class_weights if class_weights is not None else None)
        self.register_buffer("la", logit_adjust if logit_adjust is not None else None)

    def _adjusted_log_probs(self, log_probs):
        """Re-derive composed log-probs with the sub-head logit adjustment applied.
        log_probs came in already composed; we only shift the [S,V,F,Q] part by
        `la` in log-space and re-normalise the sub-distribution."""
        if self.la is None:
            return log_probs
        log_p_norm = log_probs[:, :1]
        log_rest = log_probs[:, 1:]                       # = log P(abn) + log q
        # recover log q, adjust, renormalise
        log_p_abn = torch.logsumexp(log_rest, dim=-1, keepdim=True)
        log_q = log_rest - log_p_abn
        log_q = torch.log_softmax(log_q + self.la, dim=-1)
        return torch.cat([log_p_norm, log_p_abn + log_q], dim=-1)

    def forward(self, log_probs, targets, r_fwd=None, r_bwd=None):
        lp = self._adjusted_log_probs(log_probs)
        ce = -lp.gather(1, targets[:, None]).squeeze(1)
        pt = torch.exp(-ce).clamp(1e-6, 1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce if self.gamma > 0 else ce
        if self.w is not None:
            loss = loss * self.w[targets]
        l_task = loss.mean()

        l_smooth = torch.zeros((), device=log_probs.device)
        if r_fwd is not None:
            l_smooth = l_smooth + temporal_smoothness_loss(r_fwd)
        if r_bwd is not None:
            l_smooth = l_smooth + temporal_smoothness_loss(r_bwd)
        if r_fwd is not None and r_bwd is not None:
            l_smooth = l_smooth / 2.0

        total = l_task + self.lambda_smooth * l_smooth
        return total, {"task": l_task.detach(), "smooth": l_smooth.detach(),
                       "total": total.detach()}


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 16
    x = torch.randn(B, 187, 1)
    y = torch.randint(0, 5, (B,))
    m = RLSTMClassifierHD(num_layers=2)
    lp, internals = m(x, return_internals=True)
    print("log_probs", lp.shape, "sum(exp)=", lp.exp().sum(-1).mean().item())
    assert torch.allclose(lp.exp().sum(-1), torch.ones(B), atol=1e-4)

    crit = HDLoss(class_weights=torch.tensor([0.5, 6.7, 1.2, 25.7, 50.0]))
    loss, comp = crit(lp, y, internals["r_fwd"], internals["r_bwd"])
    loss.backward()
    print("loss", float(loss), "task", float(comp["task"]), "smooth", float(comp["smooth"]))

    la = torch.log(torch.tensor([0.35, 0.58, 0.06, 0.01]))
    crit2 = HDLoss(logit_adjust=la, focal_gamma=0.0)
    lp2, i2 = m(x, return_internals=True)
    loss2, _ = crit2(lp2, y, i2["r_fwd"], i2["r_bwd"])
    loss2.backward()
    print("logit-adjust loss", float(loss2), "OK")
