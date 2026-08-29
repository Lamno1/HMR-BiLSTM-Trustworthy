"""
hmr_multipath.py  —  HMR-MultiPath: the single locked architecture for the F/S
breakthrough. See docs/RARE_CLASS_PROGRESS.md §3 and docs/LOPO_PROTOCOL.md.

    MLII ── HMR trunk (CNN+BiRLSTM+attn) ── e_morph ──┬─ shared [N,S,V,F,Q] logits
    2nd-lead ── lightweight CNN ── e_v1 ──────────────┼─ F-specialist [e_morph,e_v1] -> z_F  (overrides F)
    RR ── rhythm MLP ── e_rhythm ─────────────────────┴─ S-specialist [e_morph,e_rhythm] -> z_S (overrides S)
    e_morph ── conditional GRL record adversary  (sees [GRL(e_morph), onehot(y)])
    [z_N,z_S,z_V,z_F,z_Q] -> softmax -> CE / balanced-softmax

Every module is a constructor flag so the ablation rungs are flag combos:
  rung 0  use_v1=F use_rhythm=F use_grl=F           == plain 5-way head on the trunk
  rung 1  use_v1=T
  rung 2  use_v1=T use_rhythm=T
  rung 3  use_v1=T use_rhythm=T use_grl=T
  rung 4  + loss="balanced"

Design constraints (locked, from the measured failure modes):
  - HMR trunk unchanged — only path that ever held S and F together.
  - 2nd lead feeds ONLY the F head (not both leads through a shared CNN — that
    halved F1-S in prior 3-seed runs).
  - RR feeds ONLY the S head, NOT through the recurrent gate (beta throttled 140x).
  - Separate F / S heads — no shared classifier forced to learn two opposing
    clinical definitions.
  - GRL is class-conditional so it removes record identity *given the class*,
    not the class signal itself.
  - No hierarchical binary gate (all 3 calibration variants of that failed).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr_bilstm import ECGFeatureExtractor, BiRLSTM, AttentionPooling, temporal_smoothness_loss


# --------------------------------------------------------------------------- GRL
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


# --------------------------------------------------------------------------- V1 branch
class Lead2CNN(nn.Module):
    """Lightweight CNN on the 2nd lead -> e_v1. Deliberately small."""

    def __init__(self, out_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, out_dim, 5, padding=2), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, x):                       # x (B,187,1)
        return self.net(x.transpose(1, 2)).squeeze(-1)     # (B,out_dim)


class RhythmMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, rr):                      # rr (B,in_dim)
        return self.net(rr)


# --------------------------------------------------------------------------- model
class HMRMultiPath(nn.Module):
    def __init__(self, hidden_size=96, dropout=0.25, cnn_out_channels=64, num_layers=2,
                 use_v1=False, use_rhythm=False, use_grl=False,
                 n_records=0, rr_dim=3, v1_dim=64, rhythm_dim=32):
        super().__init__()
        self.use_v1, self.use_rhythm, self.use_grl = use_v1, use_rhythm, use_grl
        H = 2 * hidden_size

        # morphology trunk (== RLSTMClassifier front end)
        self.cnn = ECGFeatureExtractor(1, cnn_out_channels, dropout * 0.5)
        self.birlstm = BiRLSTM(cnn_out_channels, hidden_size, num_layers=num_layers, dropout=dropout)
        self.attention_pool = AttentionPooling(H)
        self.layer_norm = nn.LayerNorm(H)
        self.dropout = nn.Dropout(dropout)

        # shared 5-way head from e_morph (fallback for every class)
        self.shared_head = nn.Sequential(
            nn.Linear(H, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 5),
        )

        if use_v1:
            self.l2cnn = Lead2CNN(v1_dim, dropout * 0.5)
            self.f_head = nn.Sequential(
                nn.Linear(H + v1_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, 1))
        if use_rhythm:
            self.rhythm = RhythmMLP(rr_dim, rhythm_dim, dropout * 0.5)
            self.s_head = nn.Sequential(
                nn.Linear(H + rhythm_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, 1))
        if use_grl:
            assert n_records > 0
            self.adv = nn.Sequential(
                nn.Linear(H + 5, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, n_records))

    def embed(self, x):
        out = self.birlstm(self.cnn(x))
        h_seq, r_fwd, r_bwd = out[0], out[2], out[3]
        e, attn = self.attention_pool(h_seq)
        e = self.dropout(self.layer_norm(e))
        return e, r_fwd, r_bwd, attn

    def forward(self, x, x2=None, rr=None, y=None, grl_lambda=0.0, return_internals=False):
        e, r_fwd, r_bwd, attn = self.embed(x)
        logits = self.shared_head(e)                       # (B,5) [N,S,V,F,Q]

        if self.use_v1:
            zf = self.f_head(torch.cat([e, self.l2cnn(x2)], dim=-1)).squeeze(-1)
            logits = logits.clone()
            logits[:, 3] = zf
        if self.use_rhythm:
            zs = self.s_head(torch.cat([e, self.rhythm(rr)], dim=-1)).squeeze(-1)
            logits = logits.clone()
            logits[:, 1] = zs

        adv_logits = None
        if self.use_grl and self.training and y is not None:
            yoh = F.one_hot(y, num_classes=5).float()
            adv_logits = self.adv(torch.cat([grad_reverse(e, grl_lambda), yoh], dim=-1))

        if return_internals:
            return logits, {"r_fwd": r_fwd, "r_bwd": r_bwd, "attn": attn,
                            "adv_logits": adv_logits, "embedding": e}
        return logits


# --------------------------------------------------------------------------- loss
class MultiPathLoss(nn.Module):
    """CE (or balanced-softmax) on the 5-way head + record-adversary CE + smoothness."""

    def __init__(self, class_weights=None, lambda_smooth=0.003, loss="ce",
                 class_prior=None, adv_weight=1.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.loss = loss
        self.adv_weight = adv_weight
        self.register_buffer("w", class_weights if class_weights is not None else None)
        # balanced-softmax: add log prior to logits at train time
        self.register_buffer("logprior", torch.log(class_prior.clamp_min(1e-8))
                             if class_prior is not None else None)

    def forward(self, logits, targets, internals=None, record=None):
        z = logits + self.logprior if (self.loss == "balanced" and self.logprior is not None) else logits
        l_task = F.cross_entropy(z, targets, weight=self.w)

        l_adv = torch.zeros((), device=logits.device)
        if internals is not None and internals.get("adv_logits") is not None and record is not None:
            l_adv = F.cross_entropy(internals["adv_logits"], record)

        l_smooth = torch.zeros((), device=logits.device)
        if internals is not None:
            for r in (internals.get("r_fwd"), internals.get("r_bwd")):
                if r is not None:
                    l_smooth = l_smooth + temporal_smoothness_loss(r)
            if internals.get("r_fwd") is not None and internals.get("r_bwd") is not None:
                l_smooth = l_smooth / 2.0

        total = l_task + self.adv_weight * l_adv + self.lambda_smooth * l_smooth
        return total, {"task": l_task.detach(), "adv": l_adv.detach(), "smooth": l_smooth.detach()}


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 12
    x = torch.randn(B, 187, 1); x2 = torch.randn(B, 187, 1)
    rr = torch.randn(B, 3); y = torch.randint(0, 5, (B,)); rec = torch.randint(0, 20, (B,))
    for cfg in [dict(), dict(use_v1=True),
                dict(use_v1=True, use_rhythm=True),
                dict(use_v1=True, use_rhythm=True, use_grl=True, n_records=20)]:
        m = HMRMultiPath(**cfg)
        m.train()
        logit, intr = m(x, x2, rr, y, grl_lambda=0.1, return_internals=True)
        prior = torch.tensor([0.9, 0.02, 0.05, 0.005, 0.005])
        crit = MultiPathLoss(class_weights=torch.ones(5), class_prior=prior,
                             loss="balanced" if "use_grl" in cfg else "ce")
        loss, comp = crit(logit, y, intr, rec)
        loss.backward()
        n = sum(p.numel() for p in m.parameters())
        print(f"{str(cfg):70s} loss={float(loss):.3f} task={float(comp['task']):.3f} "
              f"adv={float(comp['adv']):.3f} params={n:,}")
