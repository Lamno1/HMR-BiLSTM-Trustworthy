"""Post-hoc analyses of the matched-recipe controls (results/matched_recipe).

Inference only. For every architecture (full ResMemECG, No-RMC, nn.LSTM BiLSTM) and seed:
  * FGSM and PGD-20 (eps 0.02, alpha 0.005) with the correctness-locked attacks of
    validation/run_robustness_final.py. Because every model here was trained with the same
    FGSM recipe, this is the recipe-matched robustness comparison the original Table lacked.
    PGD-20 runs on one fixed stratified subset (all S/V/F/Q beats + 5,000 N) to bound runtime;
    FGSM runs on the full DS2 test set.
  * For the full model, the learned blend gate beta (mean over every cell, time step, unit
    and DS2 beat), i.e. how much the trained network actually uses the RMC path.

    python -m validation.matched_recipe_analysis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_inter_patient import CONFIG  # noqa: E402
from train_matched_recipe import BUILDERS  # noqa: E402
from validation.run_robustness_final import run_linf, state_hash  # noqa: E402

RUNS = ROOT / "results" / "matched_recipe"
OUT = RUNS / "analysis.json"
ARCHS = ("full", "normc", "bilstm")
SEEDS = (42, 123, 456)


def load(arch, seed, device):
    model = BUILDERS[arch](CONFIG.copy()).to(device)
    ck = torch.load(RUNS / f"{arch}_seed{seed}_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    return model.eval()


def pgd_subset(y, n_normal=5000, seed=42):
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(y == 0)
    keep = np.concatenate([np.flatnonzero(y != 0), rng.choice(normal, n_normal, replace=False)])
    return np.sort(keep)


@torch.no_grad()
def mean_beta(model, X, device, bs=512):
    cells = [m for m in model.modules() if hasattr(m, "W_beta")]
    total, count = 0.0, 0

    def hook(mod, _inp, _out):
        nonlocal total, count
        total += mod.last_beta.double().sum().item()
        count += mod.last_beta.numel()

    handles = [c.register_forward_hook(hook) for c in cells]
    for i in range(0, len(X), bs):
        model(torch.from_numpy(X[i:i + bs]).to(device))
    for h in handles:
        h.remove()
    return total / count


def main():
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = False  # cuDNN RNN backward is unavailable in eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(ROOT / "data" / "processed" / "splits" / "inter_test.npz")
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    sub = pgd_subset(y)
    res = json.loads(OUT.read_text()) if OUT.exists() else {"pgd_subset_n": int(len(sub))}
    for arch in ARCHS:
        for seed in SEEDS:
            key = f"{arch}_seed{seed}"
            if key in res or not (RUNS / f"{key}_model.pt").exists():
                continue
            model = load(arch, seed, device)
            before = state_hash(model)
            r = {}
            r["fgsm"], _ = run_linf(model, X, y, device, "fgsm")
            r["pgd20_subset"], _ = run_linf(model, X[sub], y[sub], device, "pgd20")
            if arch == "full":
                r["beta_mean"] = mean_beta(model, X, device)
            if state_hash(model) != before:
                raise RuntimeError(f"model state changed for {key}")
            res[key] = r
            OUT.write_text(json.dumps(res, indent=2))
            print(f"{key}: clean={r['fgsm']['clean_macro_f1_nsvf']:.3f} "
                  f"fgsm={r['fgsm']['adv_macro_f1_nsvf']:.3f} "
                  f"pgd(sub)={r['pgd20_subset']['adv_macro_f1_nsvf']:.3f} "
                  + (f"beta={r['beta_mean']:.3f}" if "beta_mean" in r else ""), flush=True)


if __name__ == "__main__":
    main()
