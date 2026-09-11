"""Correctness-locked adversarial evaluation; never overwrites legacy results.

FGSM and PGD-20 run on the complete conventional DS2 test set for LSTM,
BiLSTM, and ResMemECG. Reduced C&W-L2 runs on one fixed stratified subset for
ResMemECG. All attacks use cross-entropy as a common attack objective, keep the
model in eval mode, and verify that parameters and buffers are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import INTER_TEST
from evaluate_fgsm import load_baseline_model
from report_results import load_hmr_bilstm
from robustness.cw_attack import select_stratified_subset

CLASSES = [0, 1, 2, 3]
CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
MODELS = {
    "LSTM": "results/checkpoints/best_lstm.pt",
    "BiLSTM": "results/checkpoints/best_bilstm.pt",
    "ResMemECG": "results/checkpoints/inter_best_rlstm.pt",
}


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(name, device):
    if name == "ResMemECG":
        model, _ = load_hmr_bilstm(MODELS[name], device)
    else:
        model, _ = load_baseline_model(MODELS[name], device)
    model.eval()
    return model


def predict(model, x, device, batch_size=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu())
    return torch.cat(out).numpy()


def metrics(y, clean_logits, adv_logits):
    clean = clean_logits.argmax(1)
    adv = adv_logits.argmax(1)
    initially_correct = clean == y
    flipped_wrong = initially_correct & (adv != y)
    result = {
        "n": int(len(y)),
        "n_clean_correct": int(initially_correct.sum()),
        "clean_accuracy": float(accuracy_score(y, clean)),
        "adv_accuracy": float(accuracy_score(y, adv)),
        "clean_macro_f1_nsvf": float(f1_score(y, clean, labels=CLASSES, average="macro", zero_division=0)),
        "adv_macro_f1_nsvf": float(f1_score(y, adv, labels=CLASSES, average="macro", zero_division=0)),
        "conditional_asr": float(flipped_wrong.sum() / initially_correct.sum()),
        "adversarial_error_rate": float((adv != y).mean()),
        "per_class": {},
    }
    for c in CLASSES:
        mask = y == c
        cc = initially_correct & mask
        result["per_class"][CLASS_NAMES[c]] = {
            "support": int(mask.sum()),
            "clean_recall": float(recall_score(y, clean, labels=[c], average=None, zero_division=0)[0]),
            "adv_recall": float(recall_score(y, adv, labels=[c], average=None, zero_division=0)[0]),
            "conditional_asr": float((flipped_wrong & mask).sum() / cc.sum()) if cc.sum() else None,
        }
    return result


def fgsm(model, x, y, eps):
    z = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(z), y)
    grad = torch.autograd.grad(loss, z)[0]
    lo, hi = x.amin(), x.amax()
    return (x + eps * grad.sign()).clamp(lo, hi).detach()


def pgd(model, x, y, eps, alpha, steps, generator):
    lo, hi = x.amin(), x.amax()
    noise = torch.empty_like(x).uniform_(-eps, eps, generator=generator)
    z = (x + noise).clamp(lo, hi).detach()
    for _ in range(steps):
        z.requires_grad_(True)
        grad = torch.autograd.grad(F.cross_entropy(model(z), y), z)[0]
        z = z.detach() + alpha * grad.sign()
        z = torch.max(torch.min(z, x + eps), x - eps).clamp(lo, hi).detach()
    return z


def run_linf(model, X, y, device, attack, eps=.02, alpha=.005, steps=20, seed=42, batch_size=128):
    clean_logits = predict(model, X, device)
    adv_logits, linf = [], []
    gen = torch.Generator(device=device).manual_seed(seed)
    for i in range(0, len(X), batch_size):
        bx = torch.from_numpy(X[i:i + batch_size]).to(device)
        by = torch.from_numpy(y[i:i + batch_size]).long().to(device)
        z = fgsm(model, bx, by, eps) if attack == "fgsm" else pgd(model, bx, by, eps, alpha, steps, gen)
        linf.extend((z - bx).abs().flatten(1).max(1).values.cpu().numpy().tolist())
        with torch.no_grad():
            adv_logits.append(model(z).cpu())
    result = metrics(y, clean_logits, torch.cat(adv_logits).numpy())
    result.update({"epsilon": eps, "alpha": alpha if attack == "pgd20" else None,
                   "steps": steps if attack == "pgd20" else 1,
                   "linf_max": float(np.max(linf)), "linf_mean": float(np.mean(linf))})
    return result, clean_logits


def cw_binary_search(model, x, y, data_min, data_max, search_steps, opt_steps, lr, kappa):
    batch = len(x)
    lower = torch.zeros(batch, device=x.device)
    upper = torch.full((batch,), 1e10, device=x.device)
    const = torch.full((batch,), 1e-2, device=x.device)
    best_l2 = torch.full((batch,), float("inf"), device=x.device)
    best_adv = x.clone()
    scale = data_max - data_min
    xn = ((x - data_min) / scale).clamp(1e-6, 1 - 1e-6)
    w0 = torch.atanh(2 * xn - 1).detach()
    for _ in range(search_steps):
        w = w0.clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        stage_success = torch.zeros(batch, dtype=torch.bool, device=x.device)
        for _ in range(opt_steps):
            adv = ((torch.tanh(w) + 1) / 2) * scale + data_min
            logits = model(adv)
            onehot = F.one_hot(y, logits.shape[1]).bool()
            true_logit = logits.gather(1, y[:, None]).squeeze(1)
            other = logits.masked_fill(onehot, -torch.inf).max(1).values
            margin = (true_logit - other + kappa).clamp_min(0)
            l2sq = (adv - x).flatten(1).pow(2).sum(1)
            loss = (l2sq + const * margin).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pred = logits.argmax(1)
                success = pred != y
                improved = success & (l2sq < best_l2)
                best_l2[improved] = l2sq[improved]
                best_adv[improved] = adv[improved]
                stage_success |= success
        upper = torch.where(stage_success, torch.minimum(upper, const), upper)
        lower = torch.where(stage_success, lower, torch.maximum(lower, const))
        finite = upper < 1e9
        const = torch.where(finite, (lower + upper) / 2,
                            torch.where(stage_success, const / 10, const * 10))
    return best_adv.detach(), torch.isfinite(best_l2), best_l2.sqrt()


def run_cw(model, X, y, device, seed, n_eval, search_steps, opt_steps, batch_size=32):
    rng = np.random.default_rng(seed)
    Xs, ys, indices = select_stratified_subset(X, y, n_eval, rng)
    clean_logits = predict(model, Xs, device)
    adv_logits, success, norms = [], [], []
    dmin, dmax = float(X.min()), float(X.max())
    for i in range(0, len(Xs), batch_size):
        bx = torch.from_numpy(Xs[i:i + batch_size]).to(device)
        by = torch.from_numpy(ys[i:i + batch_size]).long().to(device)
        z, ok, l2 = cw_binary_search(model, bx, by, dmin, dmax, search_steps, opt_steps, .01, 0.)
        with torch.no_grad(): adv_logits.append(model(z).cpu())
        success.extend(ok.cpu().numpy().tolist()); norms.extend(l2.cpu().numpy().tolist())
        print(f"C&W {i + len(bx)}/{len(Xs)}", flush=True)
    result = metrics(ys, clean_logits, torch.cat(adv_logits).numpy())
    finite_norms = np.asarray(norms)[np.asarray(success)]
    result.update({"subset_indices": indices.tolist(), "requested_n": n_eval,
                   "search_steps": search_steps, "optimization_steps_per_search": opt_steps,
                   "successful_optimized_fraction": float(np.mean(success)),
                   "successful_l2_mean": float(finite_norms.mean()) if len(finite_norms) else None,
                   "successful_l2_median": float(np.median(finite_norms)) if len(finite_norms) else None})
    return result, clean_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["fgsm", "pgd20", "cw", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default="results/robustness_final")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42); torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(INTER_TEST); X = data["X"].astype(np.float32); y = data["y"].astype(np.int64)
    if args.smoke: X, y = X[:256], y[:256]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    result_path = out / ("smoke_results.json" if args.smoke else "results.json")
    results = json.loads(result_path.read_text()) if result_path.exists() else {"protocol_version": "ROBUST_FINAL_V1"}
    phases = [args.phase] if args.phase != "all" else ["fgsm", "pgd20", "cw"]
    for phase in phases:
        names = ["ResMemECG"] if phase == "cw" else list(MODELS)
        results.setdefault(phase, {})
        for name in names:
            if name in results[phase]: continue
            model = load_model(name, device); before = state_hash(model)
            probe_before = predict(model, X[:64], device)
            start = time.time()
            if phase in ("fgsm", "pgd20"):
                r, _ = run_linf(model, X, y, device, phase, batch_size=64 if args.smoke else 128)
            else:
                r, _ = run_cw(model, X, y, device, 42, 20 if args.smoke else 200,
                              2 if args.smoke else 3, 10 if args.smoke else 300)
            probe_after = predict(model, X[:64], device); after = state_hash(model)
            r["runtime_seconds"] = time.time() - start
            r["state_hash_before"] = before; r["state_hash_after"] = after
            r["state_unchanged"] = before == after
            r["clean_probe_max_abs_diff"] = float(np.max(np.abs(probe_before - probe_after)))
            if not r["state_unchanged"] or r["clean_probe_max_abs_diff"] != 0:
                raise RuntimeError(f"Model state changed during {phase}/{name}")
            results[phase][name] = r
            result_path.write_text(json.dumps(results, indent=2))
            print(f"DONE {phase}/{name}: {r['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
