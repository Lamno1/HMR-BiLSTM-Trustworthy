"""
train_matched_recipe.py -- architecture controls under the OFFICIAL ResMemECG recipe.

Answers the supervisor's question "is the gain from Residual Memory Control or from
the training recipe?". Every training detail is imported from train_inter_patient.py
(focal gamma=1.5 + class weights, cosine LR, FGSM eps=0.02 on 30% of each batch,
grad clip 1.0, 45 epochs, patience 15, selection on validation macro-F1 over N/S/V/F);
only the recurrent core changes:

  full   : the unchanged ResMemECG (RMC + hybrid path). Re-run here because the archived
           seed-123/456 checkpoints used early-stopping patience 8, not 15.
  normc  : the ResMemECG cell with the RMC path removed (c_t = f*c + i*g); same
           parameters, initialisation, per-step output dropout and stacking.
  bilstm : nn.LSTM (2 layers, bidirectional, hidden 96) with the same CNN front end,
           attention pooling, LayerNorm, dropout and classifier head as ResMemECG.

Multi-seed ablations, each changing exactly one thing:
  nointer / nohybrid / meanpool : ResMemECG with m_t = 0 / beta = 1 / mean pooling.
  {full,bilstm}_{noadv,ce,nocw} : recipe ablations (no FGSM training / class-weighted
                                  cross-entropy instead of focal / focal without class weights).
  full_nosmooth                 : ResMemECG without the temporal-smoothness penalty.


Resumable per epoch (the machine reboots).

    python train_matched_recipe.py --arch normc --seeds 42 123 456
    python train_matched_recipe.py --arch bilstm --seeds 42 123 456
    python train_matched_recipe.py --arch bilstm --seeds 42 --smoke-batches 3 --out-dir <tmp>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from hmr_bilstm import (RLSTMClassifier, RLSTMCell, RLSTMLoss, ECGFeatureExtractor,
                        AttentionPooling)
from train_inter_patient import (CONFIG, ECGDataset, set_seed, cosine_lr,
                                 train_one_epoch, evaluate)

DATA_DIR = Path("data/processed/splits")
CLASS_NAMES = ["N", "S", "V", "F", "Q"]


class NoRMCCell(RLSTMCell):
    """RLSTMCell with the RMC path removed: c_t = f_t * c_{t-1} + i_t * g_t."""

    def forward(self, x_t, h_prev, c_prev):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_size, dim=1)
        i_t, f_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)
        c_t = f_t * c_prev + i_t * g_t
        h_t = self.dropout(o_t * torch.tanh(c_t))
        zeros = torch.zeros_like(c_t)
        self.last_r_t, self.last_c_keep, self.last_c_add = zeros, zeros, zeros
        return h_t, c_t


def build_normc(cfg):
    model = RLSTMClassifier(input_size=cfg["input_size"], hidden_size=cfg["hidden_size"],
                            dropout=cfg["dropout"], num_classes=cfg["num_classes"],
                            cnn_out_channels=cfg["cnn_out_channels"], num_layers=cfg["num_layers"])
    for layers in (model.birlstm.fwd_layers, model.birlstm.bwd_layers):
        for layer in layers:
            layer.cell.__class__ = NoRMCCell
    # W_c, W_h_rmc, layer_norm, W_alpha, W_beta stay allocated but receive no gradient.
    return model


class BiLSTMControl(nn.Module):
    """ResMemECG with its recurrent trunk replaced by nn.LSTM; everything else identical."""

    def __init__(self, cfg):
        super().__init__()
        h, d = cfg["hidden_size"], cfg["dropout"]
        self.cnn = ECGFeatureExtractor(cfg["input_size"], cfg["cnn_out_channels"], d * 0.5)
        self.lstm = nn.LSTM(cfg["cnn_out_channels"], h, num_layers=cfg["num_layers"],
                            bidirectional=True, dropout=d, batch_first=True)
        self.attention_pool = AttentionPooling(2 * h)
        self.layer_norm = nn.LayerNorm(2 * h)
        self.dropout = nn.Dropout(d)
        self.classifier = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(d),
                                        nn.Linear(h, cfg["num_classes"]))

    def forward(self, x, return_internals=False):
        h_seq, _ = self.lstm(self.cnn(x))
        pooled, attn = self.attention_pool(h_seq)
        logits = self.classifier(self.dropout(self.layer_norm(pooled)))
        if return_internals:
            return logits, {"r_fwd": None, "r_bwd": None, "attention_weights": attn}
        return logits


def build_full(cfg):
    return RLSTMClassifier(input_size=cfg["input_size"], hidden_size=cfg["hidden_size"],
                           dropout=cfg["dropout"], num_classes=cfg["num_classes"],
                           cnn_out_channels=cfg["cnn_out_channels"], num_layers=cfg["num_layers"])


class AblatedRMCCell(RLSTMCell):
    """RLSTMCell with one RMC component switched off; otherwise the original forward."""
    interaction = True
    hybrid = True

    def forward(self, x_t, h_prev, c_prev):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_size, dim=1)
        i_t, f_t, o_t = torch.sigmoid(i_t), torch.sigmoid(f_t), torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)
        Wc_c, Wh_h = self.W_c(c_prev), self.W_h_rmc(h_prev)
        m_t = Wc_c * Wh_h if self.interaction else torch.zeros_like(Wc_c)
        r_t = torch.sigmoid(self.layer_norm(Wc_c + Wh_h + m_t))
        c_keep = (f_t + r_t * (1.0 - f_t)) * c_prev
        c_add = (1.0 - r_t) * (i_t * g_t)
        combined = torch.cat([F.layer_norm(c_keep, (c_keep.shape[-1],)),
                              F.layer_norm(c_add, (c_add.shape[-1],))], dim=-1)
        alpha = F.softmax(self.W_alpha(combined), dim=-1)
        c_rmc = alpha[:, 0:1] * c_keep + alpha[:, 1:2] * c_add
        c_lstm = f_t * c_prev + i_t * g_t
        beta = torch.sigmoid(self.W_beta(h_prev)) if self.hybrid else torch.ones_like(c_rmc)
        c_t = beta * c_rmc + (1.0 - beta) * c_lstm
        h_t = self.dropout(o_t * torch.tanh(c_t))
        self.last_r_t, self.last_c_keep, self.last_c_add = r_t, c_keep, c_add
        self.last_alpha, self.last_beta = alpha, beta
        return h_t, c_t


class NoInteractionCell(AblatedRMCCell):
    interaction = False   # m_t = 0


class NoHybridCell(AblatedRMCCell):
    hybrid = False        # beta = 1: RMC path only, no standard-LSTM fallback


def _swap_cells(model, cls):
    for layers in (model.birlstm.fwd_layers, model.birlstm.bwd_layers):
        for layer in layers:
            layer.cell.__class__ = cls
    return model


class MeanPool(nn.Module):
    def forward(self, h_seq):
        w = torch.full(h_seq.shape[:2], 1.0 / h_seq.size(1), device=h_seq.device)
        return h_seq.mean(dim=1), w


def build_meanpool(cfg):
    model = build_full(cfg)
    model.attention_pool = MeanPool()
    return model


BUILDERS = {"full": build_full, "normc": build_normc, "bilstm": BiLSTMControl,
            "nointer": lambda cfg: _swap_cells(build_full(cfg), NoInteractionCell),
            "nohybrid": lambda cfg: _swap_cells(build_full(cfg), NoHybridCell),
            "meanpool": build_meanpool}

# Ablations: (builder, recipe override). Each changes exactly one thing.
RECIPE_ABLATIONS = {
    "noadv": {"adversarial_training": False},   # no FGSM training
    "ce": {"use_focal_loss": False},            # class-weighted cross-entropy
    "nocw": {"use_class_weights": False},       # focal loss without class weights
    "nosmooth": {"lambda_smooth": 0.0},         # no temporal-smoothness penalty
}
ARCHS = {k: (k, {}) for k in BUILDERS}
ARCHS.update({f"{b}_{r}": (b, o) for b in ("full", "bilstm") for r, o in RECIPE_ABLATIONS.items()
              if not (b == "bilstm" and r == "nosmooth")})


def class_weights(y, k):
    c = np.bincount(y, minlength=k).astype(np.float64)
    c = np.where(c == 0, 1e-9, c)
    return torch.from_numpy(np.clip(c.sum() / (k * c), 0.5, 50.0).astype(np.float32))


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out, ys = [], []
    for X, y in loader:
        out.append(model(X.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def full_metrics(y, logits):
    pred = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    labels = [0, 1, 2, 3]
    out = {
        "accuracy": float((pred == y).mean()),
        "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {},
        "confusion_NSVFQ": confusion_matrix(y, pred, labels=[0, 1, 2, 3, 4]).tolist(),
    }
    p = precision_score(y, pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    for i in labels:
        t = (y == i).astype(int)
        out["per_class"][CLASS_NAMES[i]] = {
            "support": int(t.sum()), "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]),
            "auc": float(roc_auc_score(t, probs[:, i])) if 0 < t.sum() < len(t) else None}
    m4 = y < 4
    p4 = probs[m4][:, :4] / probs[m4][:, :4].sum(1, keepdims=True)
    out["auc_macro_nsvf"] = float(roc_auc_score(y[m4], p4, multi_class="ovr", average="macro",
                                                labels=labels))
    return out


def run_seed(arch, seed, args, loaders, ytr, device, out_dir):
    builder, override = ARCHS[arch]
    cfg = {**CONFIG, **override}
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    model = BUILDERS[builder](cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    cw = class_weights(ytr, cfg["num_classes"]).to(device) if cfg["use_class_weights"] else None
    criterion = RLSTMLoss(lambda_smooth=cfg["lambda_smooth"], class_weights=cw,
                          use_focal=cfg["use_focal_loss"], focal_gamma=cfg["focal_gamma"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / f"{arch}_seed{seed}_state.pt"
    start, best_f1, best_epoch, bad, best_state, history = 1, -1.0, 0, 0, None, []
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, best_f1, best_epoch = st["epoch"] + 1, st["best_f1"], st["best_epoch"]
        bad, best_state, history = st["bad"], st["best_state"], st["history"]
        torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["np_rng"])
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(f"[resume] {arch} seed {seed} from epoch {start}", flush=True)

    epochs = 1 if args.smoke_batches else cfg["epochs"]
    for epoch in range(start, epochs + 1):
        if bad >= cfg["early_stopping_patience"]:
            break
        lr = cosine_lr(epoch - 1, cfg["epochs"], cfg["learning_rate"], cfg["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        loader = train_loader
        if args.smoke_batches:
            loader = [b for _, b in zip(range(args.smoke_batches), train_loader)]
        ts = train_one_epoch(model, loader, opt, criterion, device, cfg["grad_clip"],
                             adv_training=cfg["adversarial_training"],
                             adv_epsilon=cfg["adv_epsilon"], adv_ratio=cfg["adv_ratio"])
        vm = evaluate(model, val_loader, device, cfg["num_classes"])
        if vm["f1_macro"] > best_f1:
            best_f1, best_epoch, bad = vm["f1_macro"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append({"epoch": epoch, "lr": lr, "train_loss": ts["loss"],
                        "val_f1_macro": vm["f1_macro"], "sec": time.time() - t0})
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best_f1": best_f1, "best_epoch": best_epoch, "bad": bad,
                    "best_state": best_state, "history": history,
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   state_path)
        print(f"{arch} s{seed} ep{epoch:2d} lr={lr:.5f} loss={ts['loss']:.4f} "
              f"val_f1={vm['f1_macro']:.4f} best={best_f1:.4f}@{best_epoch} bad={bad} "
              f"{time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    logits, y = predict_logits(model, test_loader, device)
    np.savez_compressed(out_dir / f"{arch}_seed{seed}_test_logits.npz", logits=logits, y=y)
    torch.save({"model_state": best_state, "arch": arch, "seed": seed, "config": cfg},
               out_dir / f"{arch}_seed{seed}_model.pt")
    return {"arch": arch, "seed": seed, "n_params": n_params, "best_epoch": best_epoch,
            "best_val_f1_macro": best_f1, "epochs_run": history[-1]["epoch"],
            "recipe": {k: cfg[k] for k in ["epochs", "early_stopping_patience", "use_focal_loss",
                                           "use_class_weights",
                                           "focal_gamma", "adversarial_training", "adv_epsilon",
                                           "adv_ratio", "grad_clip", "learning_rate", "min_lr",
                                           "weight_decay", "lambda_smooth", "batch_size"]},
            "test": full_metrics(y, logits), "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--out-dir", default="results/matched_recipe")
    ap.add_argument("--smoke-batches", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = CONFIG["batch_size"]
    tr, va, te = (ECGDataset(DATA_DIR / f"inter_{s}.npz") for s in ("train", "val", "test"))
    loaders = (DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0),
               DataLoader(va, batch_size=bs, shuffle=False, num_workers=0),
               DataLoader(te, batch_size=bs, shuffle=False, num_workers=0))
    print(f"device={device} arch={args.arch} train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    for seed in args.seeds:
        rp = out_dir / f"{args.arch}_seed{seed}.json"
        if rp.exists() and not args.smoke_batches:
            print(f"[skip] {rp}"); continue
        res = run_seed(args.arch, seed, args, loaders, tr.y.numpy(), device, out_dir)
        rp.write_text(json.dumps(res, indent=2))
        t = res["test"]
        print(f"[done] {args.arch} s{seed} macro={t['f1_macro']:.4f} "
              + " ".join(f"{c}={t['per_class'][c]['f1']:.3f}" for c in "NSVF"), flush=True)


if __name__ == "__main__":
    main()
